"""
ui.py — Kivy UI for Basketball Shot Analyser.
Layout: [80% camera] [20% dark side panel]
"""

import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle, Color
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.utils import get_color_from_hex


# ── Colors ────────────────────────────────────────────────────────────────────
C_BG     = get_color_from_hex('#080A10')
C_PANEL  = get_color_from_hex('#0D1017')
C_BORDER = get_color_from_hex('#1A2235')
C_ACCENT = get_color_from_hex('#2563EB')
C_DIM    = get_color_from_hex('#374151')
C_TEXT   = get_color_from_hex('#D1D5DB')
C_BRIGHT = get_color_from_hex('#F9FAFB')
C_GOOD   = get_color_from_hex('#22C55E')
C_BAD    = get_color_from_hex('#EF4444')
C_WARN   = get_color_from_hex('#F59E0B')
C_BLUE   = get_color_from_hex('#60A5FA')

PHASE_COLORS = {
    'RESTING': get_color_from_hex('#6B7280'),
    'READY'  : get_color_from_hex('#22C55E'),
    'LOADING': get_color_from_hex('#F59E0B'),
    'RELEASE': get_color_from_hex('#60A5FA'),
}


# ── Camera widget ─────────────────────────────────────────────────────────────

class CameraWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._texture = None
        with self.canvas:
            Color(*C_BG)
            self._bg   = Rectangle(pos=self.pos, size=self.size)
            Color(1, 1, 1, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._upd, size=self._upd)

    def _upd(self, *_):
        self._bg.pos   = self.pos;  self._bg.size   = self.size
        self._rect.pos = self.pos;  self._rect.size = self.size

    def update_frame(self, frame_rgb: np.ndarray):
        h, w = frame_rgb.shape[:2]
        buf = frame_rgb[::-1].copy()
        if self._texture is None or self._texture.size != (w, h):
            self._texture = Texture.create(size=(w, h), colorfmt='rgb')
        self._texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self._rect.texture = self._texture


# ── Section title ─────────────────────────────────────────────────────────────

def make_title(text):
    lbl = Label(
        text=text, size_hint_y=None, height=30,
        font_size='15sp', color=C_ACCENT,
        bold=True, halign='left', valign='middle',
    )
    lbl.bind(size=lambda w, s: setattr(w, 'text_size', s))
    return lbl


def make_divider():
    w = Widget(size_hint_y=None, height=1)
    with w.canvas:
        Color(*C_BORDER)
        w._ln = Rectangle(pos=w.pos, size=(w.width, 1))
    w.bind(pos=lambda wi, p: setattr(wi._ln, 'pos', p),
           width=lambda wi, v: setattr(wi._ln, 'size', (v, 1)))
    return w


def gap(h=6):
    return Widget(size_hint_y=None, height=h)


# ── Info row (label + value) ──────────────────────────────────────────────────

class InfoRow(BoxLayout):
    def __init__(self, lbl_text, **kwargs):
        super().__init__(orientation='horizontal', size_hint_y=None,
                         height=36, spacing=4, **kwargs)
        self._lbl = Label(
            text=lbl_text, size_hint_x=0.55,
            font_size='15sp', color=C_DIM,
            halign='left', valign='middle',
        )
        self._lbl.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self._val = Label(
            text='---', size_hint_x=0.45,
            font_size='16sp', color=C_BRIGHT,
            bold=True, halign='right', valign='middle',
        )
        self._val.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self.add_widget(self._lbl)
        self.add_widget(self._val)

    def set(self, value, color=None):
        self._val.text = str(value)
        if color:
            self._val.color = color


# ── Error badge ───────────────────────────────────────────────────────────────

class ErrorBadge(Label):
    def __init__(self, text, **kwargs):
        super().__init__(
            text=text, size_hint_y=None, height=30,
            font_size='14sp', color=C_BAD,
            bold=True, halign='left', valign='middle',
        )
        self.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self.opacity = 0.0

    def show(self, v):
        self.opacity = 1.0 if v else 0.0


# ── Action button with key hint ───────────────────────────────────────────────

class ControlButton(BoxLayout):
    def __init__(self, label, key_hint, callback, btn_color=None, **kwargs):
        super().__init__(size_hint_y=None, height=38, **kwargs)

        btn = Button(
            text=f'{label}  {key_hint}',
            font_size='13sp',
            bold=True,
            background_normal='',
            background_color=btn_color or C_ACCENT,
            color=C_BRIGHT,
            pos=self.pos,
            size=self.size,
        )
        btn.bind(on_press=lambda _: callback())
        self.bind(pos=lambda *_: setattr(btn, 'pos', self.pos),
                  size=lambda *_: setattr(btn, 'size', self.size))
        self.add_widget(btn)


# ── Side panel ────────────────────────────────────────────────────────────────

class SidePanel(BoxLayout):
    def __init__(self, key_callback, **kwargs):
        super().__init__(orientation='vertical',
                         padding=[14, 16, 14, 16], spacing=3, **kwargs)
        self._key_cb = key_callback

        with self.canvas.before:
            Color(*C_PANEL)
            self._bg = Rectangle(pos=self.pos, size=self.size)
            Color(*C_BORDER)
            self._bdr = Rectangle(pos=self.pos, size=(1, self.height))
        self.bind(pos=self._upd_bg, size=self._upd_bg)

        # ── Header ────────────────────────────────────────────────────
        hdr = Label(
            text='SHOT ANALYSER', size_hint_y=None, height=40,
            font_size='18sp', color=C_BRIGHT,
            bold=True, halign='center', valign='middle',
        )
        hdr.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self.add_widget(hdr)
        self.add_widget(make_divider())
        self.add_widget(gap(8))

        # ── Status ────────────────────────────────────────────────────
        self.add_widget(make_title('STATUS'))
        self._phase_row = InfoRow('PHASE')
        self._cam_row   = InfoRow('CAMERA')
        self.add_widget(self._phase_row)
        self.add_widget(self._cam_row)
        self.add_widget(gap(6))
        self.add_widget(make_divider())
        self.add_widget(gap(6))

        # ── Angles ────────────────────────────────────────────────────
        self.add_widget(make_title('ANGLES'))
        self._elbow_row   = InfoRow('ELBOW')
        self._release_row = InfoRow('RELEASE')
        self._knee_row    = InfoRow('KNEE')
        self.add_widget(self._elbow_row)
        self.add_widget(self._release_row)
        self.add_widget(self._knee_row)
        self.add_widget(gap(6))
        self.add_widget(make_divider())
        self.add_widget(gap(6))

        # ── Score ─────────────────────────────────────────────────────
        self.add_widget(make_title('SCORE'))
        self._good_row = InfoRow('GOOD SHOTS')
        self._bad_row  = InfoRow('BAD SHOTS')
        self.add_widget(self._good_row)
        self.add_widget(self._bad_row)
        self.add_widget(gap(6))
        self.add_widget(make_divider())
        self.add_widget(gap(6))

        # ── Errors ────────────────────────────────────────────────────
        self.add_widget(make_title('ERRORS'))
        self._err_badges = []
        err_defs = [
            'ELBOW FLARE  — elbow flaring out',
            'NO FOLLOW    — no follow-through',
            'KNEE BEND    — incorrect knee angle',
            'REL. LOW     — release angle too low',
        ]
        for txt in err_defs:
            b = ErrorBadge(f'!  {txt}')
            self._err_badges.append(b)
            self.add_widget(b)

        self.add_widget(gap(6))
        self.add_widget(make_divider())
        self.add_widget(gap(6))

        # ── Controls ──────────────────────────────────────────────────
        self.add_widget(make_title('CONTROLS'))

        btns = [
            ('RESET',  '[R]', lambda: key_callback('r'), C_BORDER),
            ('MIRROR', '[F]', lambda: key_callback('f'), C_BORDER),
            ('GRAPHS', '[P]', lambda: key_callback('p'), C_BORDER),
            ('QUIT',   '[Q]', lambda: key_callback('q'),
             get_color_from_hex('#7F1D1D')),
        ]
        for lbl, key, cb, col in btns:
            self.add_widget(ControlButton(lbl, key, cb, btn_color=col))
            self.add_widget(gap(3))

        self.add_widget(Widget())

    def _upd_bg(self, *_):
        self._bg.pos  = self.pos;  self._bg.size  = self.size
        self._bdr.pos = self.pos;  self._bdr.size = (1, self.height)

    def update(self, processor):
        p = processor

        phase = p.last_phase or 'RESTING'
        self._phase_row.set(phase, color=PHASE_COLORS.get(phase, C_TEXT))

        if p.camera_misaligned:
            self._cam_row.set(f'{p.last_offset_angle}  ROTATE', color=C_WARN)
        else:
            self._cam_row.set('OK', color=C_GOOD)

        a = p.last_angles
        def fmt(k):
            v = a.get(k)
            return f'{v}' if v is not None else '---'

        self._elbow_row.set(fmt('elbow'),   color=C_BLUE)
        self._release_row.set(fmt('release'), color=C_BLUE)
        self._knee_row.set(fmt('knee'),     color=C_BLUE)

        self._good_row.set(p.good_count, color=C_GOOD)
        self._bad_row.set(p.bad_count,   color=C_BAD)

        fd = p.feedback_display
        for i, badge in enumerate(self._err_badges):
            badge.show(bool(fd[i] > 0))



# ── Main layout ───────────────────────────────────────────────────────────────

class MainLayout(BoxLayout):
    def __init__(self, key_callback, **kwargs):
        super().__init__(orientation='horizontal', **kwargs)
        with self.canvas.before:
            Color(*C_BG)
            self._bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self._bg, 'pos', self.pos),
                  size=lambda *_: setattr(self._bg, 'size', self.size))

        self.camera_widget = CameraWidget(size_hint_x=0.80)
        self.side_panel    = SidePanel(key_callback=key_callback, size_hint_x=0.20)
        self.add_widget(self.camera_widget)
        self.add_widget(self.side_panel)


# ── App ───────────────────────────────────────────────────────────────────────

class BasketballApp(App):
    def __init__(self, key_callback, **kwargs):
        super().__init__(**kwargs)
        self._key_cb = key_callback
        self.layout  = None

    def build(self):
        Window.clearcolor = C_BG
        Window.title = 'Basketball Shot Analyser'
        Window.maximize()
        self.layout = MainLayout(key_callback=self._key_cb)
        Window.bind(on_key_down=self._on_key)
        return self.layout

    def _on_key(self, window, key, scancode, codepoint, modifier):
        if codepoint:
            self._key_cb(codepoint.lower())

    def update_frame(self, frame_bgr, processor):
        if self.layout is None:
            return
        self.layout.camera_widget.update_frame(frame_bgr)
        self.layout.side_panel.update(processor)

    def schedule_update(self, frame_bgr, processor):
        Clock.schedule_once(lambda dt: self.update_frame(frame_bgr, processor), 0)