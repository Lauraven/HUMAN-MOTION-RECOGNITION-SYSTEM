"""
process_frame.py — Krepšinio metimo analizė.

METO VERTINIMAS:
  Metas užskaitomas kai žmogus praėjo LOADING + RELEASE ir grįžo į READY.
  Klaidos tikrinamos tik aktyvaus metimo metu (ne kai kamera nesulyginta).

  4 klaidos:
  0 — ELBOW FLARE  : alkūnė išsišakoja į šoną (>35° nuo vertikalės)
  1 — NO FOLLOW    : riešas per žemai po paleidimo (nėra follow-through)
  2 — KNEE BEND    : keliai per tiesūs arba per daug sulenkti
  3 — RELEASE LOW  : ranka paleista per žemai
"""

import time

import cv2
import numpy as np
from utils import find_angle, get_landmark_features, draw_text, draw_dotted_line
from good_shot_data import check_angle
from good_shot_data import deviation_score
#from ball import Ball


# ── Fazių kampų ribos ─────────────────────────────────────────────────────────
PHASE_READY_MAX = 70
PHASE_LOADING_MAX = 110
PHASE_RELEASE_MAX = 180

# ── Klaidų slenksčiai ─────────────────────────────────────────────────────────
ELBOW_FLARE_THRESH = 35
FOLLOW_THROUGH_PX = 20
KNEE_MIN = 5
KNEE_MAX = 55
RELEASE_ANGLE_MIN = 60
RELEASE_ANGLE_MAX = 90

# ── Kiti parametrai ───────────────────────────────────────────────────────────
OFFSET_THRESH = 35.0
FEEDBACK_FRAMES = 50

# ── Meto validacijos parametrai ───────────────────────────────────────────────
REQUIRE_LOADING_BEFORE_RELEASE = True
ELBOW_PEAK_MIN = 130
RESTING_ELBOW_DROP = 50
RESTING_TIMEOUT_S = 1.2


class ProcessFrame:

    def __init__(self, flip_frame=False):
        self.flip_frame = flip_frame
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.linetype = cv2.LINE_AA

        self.COLORS = {
            'blue': (0, 127, 255),
            'red': (255, 50,  50),
            'green': (0, 255,  127),
            'light_green': (100, 233, 127),
            'yellow': (255, 255,   0),
            'white': (255, 255, 255),
            'light_blue': (102, 204, 255),
            'orange': (0,  153, 255),
        }

        self.dict_features = {
            'left' : {'shoulder':11,'elbow':13,'wrist':15,
                      'hip':23,'knee':25,'ankle':27,'foot':31},
            'right': {'shoulder':12,'elbow':14,'wrist':16,
                      'hip':24,'knee':26,'ankle':28,'foot':32},
            'nose' : 0,
        }

        self.good_count = 0
        self.bad_count = 0
        self._pending_result = None
        self.last_shot_errors = set()
        self.error_details = {}
        self.last_shot_summary = {}
        self._angle_history = {'elbow': [], 'release': [], 'knee': [], 'ankle': []}
        self._smooth_n = 5
        self._peak_elbow = 0
        self._peak_release = 0
        self._min_wrist = 999
        self._past_peak = False

        self._last_active_time = 0.0
        self._had_loading      = False

        self._reset_shot()
        #self.ball = Ball()

    # ── Resetinimas ───────────────────────────────────────────────────────────

    def _reset_shot(self):
        self.shot_phases_seen = set()
        self.shot_errors = set()
        self.error_details = {}
        self.release_angles = {}
        self._peak_elbow = 0
        self._peak_release = 0
        self._min_wrist = 999
        self._past_peak = False
        self._had_loading = False

        self.feedback_display = np.zeros(4, dtype = np.int64)
        self.last_feedback = [False, False, False, False]

        if not hasattr(self, 'last_landmarks'):
            self.last_landmarks = {}
            self.last_angles = {}
            self.last_phase = None
            self.last_pose_ok = True
            self.last_offset_angle = 0
            self.camera_misaligned = False

    def reset_counters(self):
        self.good_count = 0
        self.bad_count = 0
        self._last_active_time = 0.0
        self._reset_shot()

    # ── Kampų smoothinimas ───────────────────────────────────────────────────
    def _smooth(self, key, value):
        h = self._angle_history[key]
        h.append(value)
        if len(h) > self._smooth_n:
            h.pop(0)
        return int(round(sum(h) / len(h)))

    # ── Fazės nustatymas ─────────────────────────────────────────────────────
    def _get_phase(self, elbow_angle,release_angle, wrist_y = None, hip_y = None):
        if wrist_y is not None and hip_y is not None:
            if wrist_y > hip_y + 20 or (release_angle < 10 and elbow_angle <= PHASE_RELEASE_MAX):
                return 'RESTING'
        if elbow_angle <= PHASE_READY_MAX:
            return 'READY'
        if elbow_angle <= PHASE_LOADING_MAX:
            return 'LOADING'
        if elbow_angle <= PHASE_RELEASE_MAX and release_angle >= 75:
                return 'RELEASE'
        return None

    # ── Klaidų tikrinimas ─────────────────────────────────────────────────────
    def _check_errors(self, phase, elbow_angle, knee_angle, release_angle, wrist_above_elbow):

        flags = [False, False, False, False]

        if phase == 'LOADING':
            ok, status = check_angle('elbow', elbow_angle, phase)
            if not ok:
                flags[0] = True 
                self.error_details[0] = f'Elbow {status}'

        if phase == 'RELEASE' and wrist_above_elbow < -FOLLOW_THROUGH_PX:
            flags[1] = True

        if phase == 'LOADING':
            ok, status = check_angle('knee', knee_angle, phase)
            if not ok:
                flags[2] = True
                self.error_details[2] = f'Knee {status}'

        if phase == 'RELEASE':
            ok, _ = check_angle('shoulder_flexion', release_angle, 'RELEASE')
            if not ok:
                flags[3] = True

        return flags

    # ── Metimo vertinimas ───────────────────────────────────────────────────────

    def _evaluate_shot(self):

        if 'RELEASE' not in self.shot_phases_seen:
            print(f'[DEBUG] skip — no RELEASE in {self.shot_phases_seen}')
            return

        if REQUIRE_LOADING_BEFORE_RELEASE and not self._had_loading:
            print(f'[DEBUG] skip — no LOADING before RELEASE (rankų nuleidimas?)')
            return
        
        if self._peak_elbow < ELBOW_PEAK_MIN:
            print(f'[DEBUG] skip — peak_elbow={self._peak_elbow} < {ELBOW_PEAK_MIN}')
            return

        if self._peak_release < RELEASE_ANGLE_MIN:
            print(f'[DEBUG] skip — peak_release={self._peak_release} < {RELEASE_ANGLE_MIN} (ne tikras metas)')
            return

        # ── Tikras metas — vertinti klaidas ──────────────────────────────────
        ok_elbow, s_elbow = check_angle('elbow', self._peak_elbow, 'RELEASE')
        if not ok_elbow:
            self.shot_errors.add(0)
            self.error_details[0] = f'Elbow peak {s_elbow}'

        ok_release, s_release = check_angle('shoulder_flexion', self._peak_release, 'RELEASE')
        if not ok_release:
            self.shot_errors.add(3)
            self.error_details[3] = f'Release peak {s_release}'

        if self._min_wrist < -FOLLOW_THROUGH_PX:
            self.shot_errors.add(1)
            self.error_details[1] = f'Wrist min {self._min_wrist:+d}px'

        print(f'[DEBUG] eval: peak_elbow = {self._peak_elbow} peak_release = {self._peak_release} '
              f'min_wrist = {self._min_wrist} had_loading = {self._had_loading} errors = {self.shot_errors}')

        if not self.shot_errors:
            self.good_count += 1
            self._pending_result = 'GOOD'
        else:
            self.bad_count += 1
            self._pending_result = 'BAD'

        summary = {}
        for key, val in self.release_angles.items():
            ok, status = check_angle(key, val, 'RELEASE')
            dev = deviation_score(key, val, 'RELEASE')
            summary[key] = {'value': val, 'ok': ok, 'status': status, 'dev': round(dev, 2)}
        summary['_error_details'] = dict(self.error_details)
        self.last_shot_summary = summary


    # ── Pagrindinis metodas ───────────────────────────────────────────────────

    def process(self, frame: np.ndarray, pose):
        if self._pending_result is not None:
            self.shot_just_finished = True
            self.shot_result = self._pending_result
            self._pending_result = None
        else:
            self.shot_just_finished = False
            self.shot_result = None

        frame_height, frame_width = frame.shape[:2]
        keypoints = pose.process(frame)

        if not keypoints.pose_landmarks:
            self.last_phase        = None
            self.camera_misaligned = False
            if self.flip_frame:
                frame = cv2.flip(frame, 1)
            return frame

        ps_lm = keypoints.pose_landmarks

        nose_coord = get_landmark_features(
            ps_lm.landmark, self.dict_features, 'nose', frame_width, frame_height)
        l_shldr, l_elbow, l_wrist, l_hip, l_knee, l_ankle, l_foot = \
            get_landmark_features(ps_lm.landmark, self.dict_features, 'left', frame_width, frame_height)
        r_shldr, r_elbow, r_wrist, r_hip, r_knee, r_ankle, r_foot = \
            get_landmark_features(ps_lm.landmark, self.dict_features, 'right', frame_width, frame_height)

        VIS = 0.5
        lm = ps_lm.landmark
        lf = self.dict_features['left']
        rf = self.dict_features['right']

        def _vis(idx):
            return getattr(lm[idx], 'visibility', 1.0) >= VIS

        def _resolve(li, ri, lc, rc):
            lv, rv = _vis(li), _vis(ri)
            return (lc.tolist() if lv else (rc.tolist() if rv else ['', ''])), \
                   (rc.tolist() if rv else (lc.tolist() if lv else ['', '']))

        ls, rs = _resolve(lf['shoulder'], rf['shoulder'], l_shldr, r_shldr)
        le, re = _resolve(lf['elbow'], rf['elbow'], l_elbow, r_elbow)
        lw, rw = _resolve(lf['wrist'], rf['wrist'], l_wrist, r_wrist)
        lh, rh = _resolve(lf['hip'], rf['hip'], l_hip,   r_hip)
        lk, rk = _resolve(lf['knee'], rf['knee'], l_knee,  r_knee)
        la, ra = _resolve(lf['ankle'], rf['ankle'], l_ankle, r_ankle)

        self.last_landmarks = {
            'l_shoulder': ls, 'r_shoulder': rs,
            'l_elbow': le, 'r_elbow': re,
            'l_wrist': lw, 'r_wrist': rw,
            'l_hip': lh, 'r_hip': rh,
            'l_knee': lk, 'r_knee': rk,
            'l_ankle': la, 'r_ankle': ra,
        }

        offset_angle = find_angle(l_shldr, r_shldr, nose_coord)
        self.last_offset_angle = offset_angle

        if offset_angle > OFFSET_THRESH:
            self.camera_misaligned = True
            self.last_phase = None
            cv2.circle(frame, tuple(nose_coord), 7, self.COLORS['white'],  -1)
            cv2.circle(frame, tuple(l_shldr), 7, self.COLORS['yellow'], -1)
            cv2.circle(frame, tuple(r_shldr), 7, self.COLORS['orange'], -1)
            if self.flip_frame:
                frame = cv2.flip(frame, 1)
            return frame

        self.camera_misaligned = False

        dist_l = abs(int(l_foot[1]) - int(l_shldr[1]))
        dist_r = abs(int(r_foot[1]) - int(r_shldr[1]))

        if dist_l >= dist_r:
            shldr, elbow, wrist, hip, knee, ankle, foot = \
                l_shldr, l_elbow, l_wrist, l_hip, l_knee, l_ankle, l_foot
            mult = -1
        else:
            shldr, elbow, wrist, hip, knee, ankle, foot = \
                r_shldr, r_elbow, r_wrist, r_hip, r_knee, r_ankle, r_foot
            mult = 1

        elbow_angle = self._smooth('elbow', find_angle(shldr, wrist, elbow))
        release_angle = self._smooth('release', find_angle(shldr, np.array([shldr[0], 0]), elbow))
        knee_angle = self._smooth('knee', find_angle(hip, ankle, knee))
        ankle_angle = self._smooth('ankle', find_angle(knee, foot, ankle))
        wrist_above_elbow = int(elbow[1]) - int(wrist[1])

        self.last_angles = {
            'elbow': elbow_angle,
            'release': release_angle,
            'knee': knee_angle,
            'ankle': ankle_angle,
            'wrist_above': wrist_above_elbow,
        }

        # ── Skeleto piešimas ─────────────────────────────────────────
        cv2.ellipse(frame, tuple(elbow), (25, 25), 0,
                    -90, -90 + mult * elbow_angle,
                    self.COLORS['white'], 3, self.linetype)
        cv2.ellipse(frame, tuple(shldr), (30, 30), 0,
                    -90, -90 - mult * release_angle,
                    self.COLORS['white'], 3, self.linetype)
        cv2.ellipse(frame, tuple(knee), (20, 20), 0,
                    -90, -90 - mult * knee_angle,
                    self.COLORS['white'], 3, self.linetype)

        draw_dotted_line(frame, elbow, elbow[1]-60, elbow[1]+20, self.COLORS['blue'])
        draw_dotted_line(frame, shldr, shldr[1]-80, shldr[1]+20, self.COLORS['blue'])
        draw_dotted_line(frame, knee, knee[1]-50, knee[1]+20, self.COLORS['blue'])

        cv2.line(frame, tuple(shldr), tuple(elbow), self.COLORS['light_blue'], 4, self.linetype)
        cv2.line(frame, tuple(elbow), tuple(wrist), self.COLORS['light_blue'], 4, self.linetype)
        cv2.line(frame, tuple(shldr), tuple(hip), self.COLORS['light_blue'], 4, self.linetype)
        cv2.line(frame, tuple(hip), tuple(knee), self.COLORS['light_blue'], 4, self.linetype)
        cv2.line(frame, tuple(knee), tuple(ankle), self.COLORS['light_blue'], 4, self.linetype)
        cv2.line(frame, tuple(ankle), tuple(foot), self.COLORS['light_blue'], 4, self.linetype)

        for pt in [shldr, elbow, wrist, hip, knee, ankle, foot]:
            cv2.circle(frame, tuple(pt), 7, self.COLORS['yellow'], -1, self.linetype)

        flip = self.flip_frame
        ex = (frame_width - int(elbow[0]) + 15) if flip else (int(elbow[0]) + 15)
        sx = (frame_width - int(shldr[0]) + 15) if flip else (int(shldr[0]) + 15)
        kx = (frame_width - int(knee[0])  + 15) if flip else (int(knee[0])  + 15)

        if flip:
            frame = cv2.flip(frame, 1)

        # ── Kampų rodymas ─────────────────────────────────────────────
        txt_col = self.COLORS['light_blue']

        cv2.putText(frame, f'{elbow_angle} deg',
                    (ex, int(elbow[1])),
                    self.font, 0.6, txt_col, 2, self.linetype)
        cv2.putText(frame, f'{release_angle} deg',
                    (sx, int(shldr[1]) + 10),
                    self.font, 0.6, txt_col, 2, self.linetype)
        cv2.putText(frame, f'{knee_angle} deg',
                    (kx, int(knee[1]) + 10),
                    self.font, 0.6, txt_col, 2, self.linetype)

        # ── Fazės nustatymas ir metimo logika ──────────────────────────
        phase = self._get_phase(elbow_angle,release_angle, int(wrist[1]), int(hip[1]))
        now   = time.perf_counter()

        prev_had_release = 'RELEASE' in self.shot_phases_seen

        if phase in ('LOADING', 'RELEASE', 'UNKNOWN') or phase is None:
            if elbow_angle > self._peak_elbow:
                self._peak_elbow = elbow_angle

        # ── Aktyvumo laikmatis ───────────────────────────────────────
        if phase in ('LOADING', 'RELEASE'):
            self._last_active_time = now
            if phase == 'LOADING':
                self._had_loading = True

        # ── RESTING nustatymas — trys keliai ────────────────────────
        elbow_drop = self._peak_elbow - elbow_angle
        time_since_active = now - self._last_active_time if self._last_active_time > 0 else 999

        drop_resting = (
            prev_had_release and
            self._peak_elbow >= ELBOW_PEAK_MIN and
            elbow_drop >= RESTING_ELBOW_DROP and
            phase in ('RELEASE', 'UNKNOWN', 'LOADING', None)
        )
        timeout_resting = (
            prev_had_release and
            self._peak_elbow >= ELBOW_PEAK_MIN and
            time_since_active >= RESTING_TIMEOUT_S
        )

        if drop_resting or timeout_resting:
            phase = 'RESTING'

        if phase is None:
            phase = 'UNKNOWN'

        self.last_phase = phase

        if phase == 'RESTING':
            if prev_had_release:
                self._evaluate_shot()
                self.last_shot_errors = set(self.shot_errors)
            else:
                self.last_shot_errors = set()
            self._reset_shot()
            self._last_active_time = 0.0

        elif phase == 'READY':
            pass

        elif phase in ('LOADING', 'RELEASE'):
            self.shot_phases_seen.add(phase)

            flags = self._check_errors(
                phase, elbow_angle, knee_angle,
                release_angle, wrist_above_elbow)
            self.last_feedback = flags

            for i, f in enumerate(flags):
                if f:
                    self.feedback_display[i] = FEEDBACK_FRAMES

            if phase == 'RELEASE':
                if elbow_angle > self._peak_elbow:
                    self._peak_elbow = elbow_angle
                if release_angle > self._peak_release:
                    self._peak_release = release_angle
                if elbow_angle >= 130 and not self._past_peak:
                    if elbow_angle < self._peak_elbow - 10:
                        self._past_peak = True
                    elif wrist_above_elbow < self._min_wrist:
                        self._min_wrist = wrist_above_elbow
                self.release_angles = {
                    'elbow': elbow_angle,
                    'release': release_angle,
                    'knee': knee_angle,
                }

        self.feedback_display = np.maximum(0, self.feedback_display - 1)
        self.last_pose_ok     = len(self.shot_errors) == 0
        facing_right = (mult == 1)
        #frame = self.ball.update(frame, ps_lm.landmark, phase=phase, facing_right=facing_right, camera_ratio=0.7)
        return frame