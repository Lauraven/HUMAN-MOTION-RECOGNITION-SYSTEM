"""
plotter.py — Krepšinio metimo duomenų grafikų modulis.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import butter, filtfilt
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

BUTTER_ORDER = 4
CUTOFF_COORD_HZ = 3.5
CUTOFF_ANGLE_HZ = 5.0

STYLE = {
    'bg': '#0e1117',
    'panel': '#161b22',
    'grid': '#21262d',
    'raw_color': "#ff0000",
    'filt_color': '#3fb950',
    'text': '#c9d1d9',
    'subtext': '#8b949e',
    'tab_active': '#1f6feb',
    'tab_inactive': '#21262d',
    'zone_release': '#3fb950',
}

# Etalonų ribos pagal fazes
ANGLE_ZONES = {
    'elbow': [(150, 180, 'RELEASE', 'zone_release')],
    'release': [(110, 140, 'RELEASE', 'zone_release')],
    'knee': [(140, 180, 'RELEASE', 'zone_release')],
    'ankle': [(115, 130, 'RELEASE', 'zone_release')],
}

COORD_PAIRS = [
    ('l_shoulder', 'Left shoulder'),
    ('r_shoulder', 'Right shoulder'),
    ('l_elbow', 'Left elbow'),
    ('r_elbow', 'Right elbow'),
    ('l_wrist', 'Left wrist'),
    ('r_wrist', 'Right wrist'),
    ('l_hip', 'Left hip'),
    ('r_hip', 'Right hip'),
    ('l_knee', 'Left knee'),
    ('r_knee', 'Right knee'),
]

ANGLE_COLS = [
    ('elbow', 'Elbow angle'),
    ('release', 'Release angle'),
    ('knee', 'Knee angle'),
    ('ankle', 'Ankle angle'),
]


# ── Butterworth filtras ───────────────────────────────────────────────────────
def _butter_lowpass(data, cutoff, fs, order = 4):
    if fs <= 0 or len(data) < order * 3 + 1:
        return data.copy()
    series = pd.Series(data.astype(float))
    nan_mask = series.isna()
    filled = series.interpolate(method = 'linear', limit_direction = 'both').bfill().ffill()
    if filled.isna().all():
        return data.copy()
    nyq = fs / 2.0
    norm_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, norm_cutoff, btype = 'low', analog = False)
    try:
        filtered = filtfilt(b, a, filled.values)
    except Exception:
        return data.copy()
    result = filtered.copy()
    result[nan_mask.values] = np.nan
    return result


def _draw_zones(ax, key, t_end):
    if key not in ANGLE_ZONES:
        return
    for lo, hi, phase_lbl, col_key in ANGLE_ZONES[key]:
        col = STYLE[col_key]
        ax.axhspan(lo, hi, alpha = 0.12, color = col, zorder = 0)
        ax.axhline(lo, color = col, lw = 0.8, linestyle = '--', alpha = 0.5)
        ax.axhline(hi, color = col, lw = 0.8, linestyle = '--', alpha = 0.5)
        ax.text(t_end, hi + 1, phase_lbl,
                color = col, fontsize = 6, ha = 'right', va = 'bottom')


# ── Pagrindinis rinkėjas ──────────────────────────────────────────────────────
class DataPlotter:

    def __init__(self, fps = 30.0, base_output_dir = 'plots'):
        self.fps = fps
        self.base_output_dir = base_output_dir
        self._times = []
        self._landmarks = []
        self._angles = []
        self._shot_counter = 1

    def collect(self, timestamp_s, landmarks, angles):
        self._times.append(timestamp_s)
        self._landmarks.append(dict(landmarks))
        safe = {k: (v if v is not None else float('nan')) for k, v in angles.items()}
        self._angles.append(safe)

    def reset(self):
        self._times.clear()
        self._landmarks.clear()
        self._angles.clear()
        self._shot_counter += 1

    def _build_coord_df(self):
        t = np.array(self._times)
        rows = []
        for lm in self._landmarks:
            row = {}
            for key, _ in COORD_PAIRS:
                val = lm.get(key, ['', ''])
                try:
                    row[key + '_x'] = float(val[0]) if val[0] != '' else np.nan
                    row[key + '_y'] = float(val[1]) if val[1] != '' else np.nan
                except (TypeError, ValueError):
                    row[key + '_x'] = np.nan
                    row[key + '_y'] = np.nan
            rows.append(row)
        raw = pd.DataFrame(rows)
        raw.insert(0, 'time', t)
        filt = raw.copy()
        for col in raw.columns[1:]:
            filt[col] = _butter_lowpass(raw[col].values, CUTOFF_COORD_HZ, self.fps)
        return raw, filt

    def _build_angle_df(self):
        t = np.array(self._times)
        rows = []
        for ang in self._angles:
            row = {}
            for key, _ in ANGLE_COLS:
                try:
                    row[key] = float(ang[key]) if key in ang else np.nan
                except (TypeError, ValueError):
                    row[key] = np.nan
            rows.append(row)
        raw = pd.DataFrame(rows)
        raw.insert(0, 'time', t)
        filt = raw.copy()
        for col in raw.columns[1:]:
            filt[col] = _butter_lowpass(raw[col].values, CUTOFF_ANGLE_HZ, self.fps)
        return raw, filt

    def _make_coord_figure(self, key, label, raw_coord, filt_coord):
        fig = matplotlib.figure.Figure(figsize = (10, 5), facecolor = STYLE['bg'])
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 1)
        fig.subplots_adjust(left = 0.09, right = 0.97, top = 0.92, bottom = 0.10, hspace = 0.06)
        fig.suptitle(label, color = STYLE['text'], fontsize = 11)
        for ax, col, ylabel in zip(axes, [key+'_x', key+'_y'], ['X (px)', 'Y (px)']):
            ax.plot(raw_coord['time'], raw_coord[col],
                    color = STYLE['raw_color'], lw = 1.5, alpha = 0.55, label = 'Raw')
            ax.plot(filt_coord['time'], filt_coord[col],
                    color = STYLE['filt_color'], lw = 1.5, label = 'Filtered')
            ax.set_facecolor(STYLE['panel'])
            ax.set_ylabel(ylabel, color = STYLE['subtext'], fontsize = 8)
            ax.tick_params(colors = STYLE['subtext'], labelsize = 7)
            for spine in ax.spines.values():
                spine.set_edgecolor(STYLE['grid'])
            ax.grid(True, color = STYLE['grid'], linewidth = 0.5, linestyle = '--', alpha = 0.7)
        axes[0].tick_params(labelbottom=False)
        axes[0].legend(fontsize = 7, loc = 'upper right',
                       facecolor = STYLE['panel'], edgecolor = STYLE['grid'],
                       labelcolor = STYLE['text'], framealpha = 0.9)
        axes[1].set_xlabel('Time (s)', color = STYLE['subtext'], fontsize = 8)
        return fig

    def _make_angle_figure(self, key, label, raw_angle, filt_angle):
        fig = matplotlib.figure.Figure(figsize = (10, 3.5), facecolor = STYLE['bg'])
        FigureCanvasAgg(fig)
        ax = fig.subplots(1, 1)
        fig.subplots_adjust(left = 0.09, right = 0.97, top = 0.90, bottom = 0.13)
        fig.suptitle(label, color = STYLE['text'], fontsize = 11)
        t_end = float(raw_angle['time'].iloc[-1]) if len(raw_angle) > 0 else 1.0
        _draw_zones(ax, key, t_end)
        ax.plot(raw_angle['time'], raw_angle[key], color = STYLE['raw_color'], lw = 1.0, alpha = 0.55, label = 'Raw')
        ax.plot(filt_angle['time'], filt_angle[key], color = STYLE['filt_color'], lw = 1.5, label = 'Filtered')
        ax.set_facecolor(STYLE['panel'])
        ax.set_ylabel('Angle (deg)', color = STYLE['subtext'], fontsize = 8)
        ax.set_xlabel('Time (s)', color = STYLE['subtext'], fontsize = 8)
        ax.tick_params(colors = STYLE['subtext'], labelsize = 7)
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE['grid'])
        ax.grid(True, color = STYLE['grid'], linewidth = 0.5, linestyle = '--', alpha = 0.7)
        ax.legend(fontsize = 7, loc = 'upper right', facecolor = STYLE['panel'], edgecolor = STYLE['grid'],
                  labelcolor = STYLE['text'], framealpha = 0.9)
        return fig

    def save(self, output_dir='plots'):
        if not self._times:
            print('[Plotter] No data to save.')
            return []
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out = Path(output_dir) / f'{now}_shotdata'
        out.mkdir(parents=True, exist_ok=True)
        matplotlib.rcParams.update({
            'figure.facecolor': STYLE['bg'],
            'axes.facecolor': STYLE['panel'],
            'text.color': STYLE['text'],
        })
        raw_coord, filt_coord = self._build_coord_df()
        raw_angle, filt_angle = self._build_angle_df()
        saved = []
        for key, label in COORD_PAIRS:
            if not raw_coord[key + '_x'].isna().all():
                fig = self._make_coord_figure(key, label, raw_coord, filt_coord)
                path = out / f'coord_{key}.png'
                fig.savefig(path, dpi = 120, bbox_inches = 'tight', facecolor = STYLE['bg'])
                saved.append(str(path))
        for key, label in ANGLE_COLS:
            if not raw_angle[key].isna().all():
                fig = self._make_angle_figure(key, label, raw_angle, filt_angle)
                path = out / f'angle_{key}.png'
                fig.savefig(path, dpi = 120, bbox_inches = 'tight', facecolor = STYLE['bg'])
                saved.append(str(path))
        print(f'[Plotter] Saved {len(saved)} graphs -> {out.resolve()}')
        return saved

    def show(self):
        plt.show(block=True)


# ── Tabbed langas ─────────────────────────────────────────────────────────────
class _TabbedWindow:

    BTN_H = 0.04
    BTN_PAD = 0.005
    TOP_PAD = 0.07

    def __init__(self, tabs):
        self.tabs = tabs
        self.n = len(tabs)
        self.current_idx = 0

        matplotlib.rcParams.update({
            'figure.facecolor': STYLE['bg'],
            'axes.facecolor': STYLE['panel'],
            'text.color': STYLE['text'],
            'xtick.color': STYLE['subtext'],
            'ytick.color': STYLE['subtext'],
            'font.family': 'monospace',
        })

        self.fig = plt.figure(figsize=(11, 6.5), facecolor=STYLE['bg'])
        try:
            self.fig.canvas.manager.set_window_title('Basketball Shot Analysis')
        except Exception:
            pass

        pad = self.BTN_PAD
        btn_w = (1.0 - pad * (self.n + 1)) / self.n
        y_pos = 1.0 - self.BTN_H - pad

        self.btn_widgets = []
        for i, (label, _) in enumerate(tabs):
            x = pad + i * (btn_w + pad)
            ax_b = self.fig.add_axes([x, y_pos, btn_w, self.BTN_H])
            btn = Button(ax_b, label[:15], color = STYLE['tab_inactive'], hovercolor = STYLE['tab_active'])
            btn.label.set_fontsize(7)
            btn.label.set_color(STYLE['text'])
            btn.on_clicked(lambda _, idx = i: self._switch(idx))
            self.btn_widgets.append(btn)

        self._switch(0)
        plt.show(block=True)

    def _switch(self, idx):
        btn_axes = {b.ax for b in self.btn_widgets}
        for ax in list(self.fig.get_axes()):
            if ax not in btn_axes:
                self.fig.delaxes(ax)

        label, src_fig = self.tabs[idx]
        content_h = 1.0 - self.TOP_PAD
        src_axes  = src_fig.get_axes()
        n = len(src_axes)
        gap = 0.02
        ax_h = (content_h - gap * n) / n

        tab_key = None
        for ak, al in ANGLE_COLS:
            if al == label:
                tab_key = ak
                break

        for j, src_ax in enumerate(src_axes):
            ax_y = content_h - (j + 1) * ax_h - j * gap
            new_ax = self.fig.add_axes([0.08, ax_y, 0.88, ax_h], facecolor = STYLE['panel'])

            xlim = src_ax.get_xlim()
            if tab_key:
                _draw_zones(new_ax, tab_key, xlim[1])

            for line in src_ax.get_lines():
                new_ax.plot(line.get_xdata(), line.get_ydata(), color = line.get_color(), lw = line.get_linewidth(),
                            alpha = line.get_alpha() or 1.0, label = line.get_label(), linestyle = line.get_linestyle())

            new_ax.set_ylabel(src_ax.get_ylabel(), color = STYLE['subtext'], fontsize = 8)
            new_ax.set_ylim(src_ax.get_ylim())
            new_ax.set_xlim(xlim)

            is_last = (j == n - 1)
            if is_last:
                new_ax.set_xlabel('Time (s)', color = STYLE['subtext'], fontsize = 8)
            else:
                new_ax.tick_params(labelbottom=False)
            new_ax.tick_params(colors = STYLE['subtext'], labelsize = 7)
            for spine in new_ax.spines.values():
                spine.set_edgecolor(STYLE['grid'])
            new_ax.grid(True, color = STYLE['grid'], linewidth = 0.5, linestyle='--', alpha=0.7)
            if j == 0:
                new_ax.legend(fontsize = 7, loc = 'upper right',
                              facecolor = STYLE['panel'], edgecolor = STYLE['grid'],
                              labelcolor = STYLE['text'], framealpha = 0.9)

        self.fig.suptitle(label, color = STYLE['text'], fontsize = 11, y = 0.99)

        for i, btn in enumerate(self.btn_widgets):
            btn.ax.set_facecolor(
                STYLE['tab_active'] if i == idx else STYLE['tab_inactive'])
            btn.label.set_fontweight('bold' if i == idx else 'normal')

        self.fig.canvas.draw_idle()
        self.current_idx = idx


# ── show() naudoja TabbedWindow ───────────────────────────────────────────────

def _build_tabs(plotter):
    matplotlib.rcParams.update({
        'figure.facecolor': STYLE['bg'],
        'axes.facecolor': STYLE['panel'],
        'text.color': STYLE['text'],
    })
    raw_coord, filt_coord = plotter._build_coord_df()
    raw_angle, filt_angle = plotter._build_angle_df()
    tabs = []
    for key, label in COORD_PAIRS:
        if not raw_coord[key + '_x'].isna().all():
            fig = plotter._make_coord_figure(key, label, raw_coord, filt_coord)
            tabs.append((label, fig))
    for key, label in ANGLE_COLS:
        if not raw_angle[key].isna().all():
            fig = plotter._make_angle_figure(key, label, raw_angle, filt_angle)
            tabs.append((label, fig))
    return tabs


def _tabbed_show(self):
    if not self._times:
        print('[Plotter] No data collected.')
        return
    tabs = _build_tabs(self)
    if not tabs:
        print('[Plotter] All data empty.')
        return
    _TabbedWindow(tabs)

DataPlotter.show = _tabbed_show