"""
KŪRIMO PROCESE. NĖRA IMPLEMENTUOTAS.
ball.py — Krepšinio kamuolio vizualizacija.
Logika:
  RESTING/None — kamuolys nejuda, stovi ore prieš žmogų
  READY/LOADING — kamuolys tarp delnų
  RELEASE — kamuolys paleidžiamas ir skrenda
  rankos žemiau klubo LOADING metu — kamuolys grąžinamas į idle
"""
import cv2
import numpy as np
import time

# Landmark indeksai (Tasks API)
_LS = 11
_RS = 12
_LH = 23
_RH = 24
_LW = 15
_RW = 16
_LE = 13
_RE = 14

BALL_RADIUS = 42 
BALL_COLOR  = (180, 255, 180)
BALL_BORDER = (80, 200, 80)
BALL_ALPHA  = 0.55
BALL_DOT_COLOR = (60, 180, 60)
BALL_DOT_R     = 6

# Fizika
GRAVITY = 1210.0
FLY_DT  = 1 / 30.0

BALL_FRONT_OFFSET_X = 150
BALL_FRONT_OFFSET_Y = -30

WRIST_BALL_OFFSET = BALL_RADIUS

# ── Skrydžio greičio parametrai (px/s) ───────────────────────────────────────
_VY_THRESHOLD = -200.0
_VY_DEFAULT = -700.0
_VY_MIN = -1500.0
_VY_MAX = -600.0
_VX_MIN = 180.0
_VX_MAX = 800.0
_VEL_FRAMES = 7

_FACING_STABLE_FRAMES = 8


class Ball:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self._state    = 'idle'
        self._released = False

        self._hands_down_since: float | None = None
        self._HANDS_DOWN_DELAY = 1.5

        self._wrist_history: list = []

        self._facing_r: bool = True
        self._facing_counter: int = 0

    # ── Stabilus facing ───────────────────────────────────────────────────────

    def _update_facing(self, lm):
        raw = lm[_LS].x < lm[_RS].x
        if raw == self._facing_r:
            self._facing_counter = 0
        else:
            self._facing_counter += 1
            if self._facing_counter >= _FACING_STABLE_FRAMES:
                self._facing_r = raw
                self._facing_counter = 0

    # ── Pozicijų funkcijos ────────────────────────────────────────────────────

    def _wrist_pos(self, lm, w, h):
        if self._facing_r:
            w_pt = np.array([lm[_LW].x * w, lm[_LW].y * h])
            e_pt = np.array([lm[_LE].x * w, lm[_LE].y * h])
        else:
            w_pt = np.array([lm[_RW].x * w, lm[_RW].y * h])
            e_pt = np.array([lm[_RE].x * w, lm[_RE].y * h])

        direction = w_pt - e_pt
        norm = np.linalg.norm(direction)
        if norm > 1e-3:
            direction = direction / norm
        else:
            direction = np.array([1.0 if self._facing_r else -1.0, 0.0])

        return w_pt + direction * WRIST_BALL_OFFSET

    def _idle_pos(self, lm, w, h):
        ls = np.array([lm[_LS].x * w, lm[_LS].y * h])
        rs = np.array([lm[_RS].x * w, lm[_RS].y * h])
        chest = (ls + rs) / 2
        direction = 1 if self._facing_r else -1
        return np.array([
            chest[0] + direction * BALL_FRONT_OFFSET_X,
            chest[1] + BALL_FRONT_OFFSET_Y
        ])

    def _hip_y(self, lm, h):
        return (lm[_LH].y * h + lm[_RH].y * h) / 2

    def _hands_are_down(self, lm, h):
        hip_y = self._hip_y(lm, h)
        return lm[_LW].y * h > hip_y and lm[_RW].y * h > hip_y

    # ── Greičio buferis ───────────────────────────────────────────────────────
    def _push_wrist(self, pos):
        self._wrist_history.append((pos[0], pos[1], time.time()))
        if len(self._wrist_history) > _VEL_FRAMES:
            self._wrist_history.pop(0)

    def _clear_wrist(self):
        self._wrist_history.clear()

    def _compute_velocity(self):
        if len(self._wrist_history) < 2:
            return 0.0, 0.0
        x0, y0, t0 = self._wrist_history[0]
        x1, y1, t1 = self._wrist_history[-1]
        dt = t1 - t0
        if dt < 1e-4:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt

    def _launch_velocity(self):
        raw_vx, raw_vy = self._compute_velocity()

        if raw_vy < _VY_THRESHOLD:
            vy = float(np.clip(raw_vy, _VY_MIN, _VY_MAX))
        else:
            vy = _VY_DEFAULT

        if abs(raw_vx) > _VX_MIN:
            vx = float(np.clip(abs(raw_vx), _VX_MIN, _VX_MAX))
            vx = vx if raw_vx > 0 else -vx
        else:
            vx = _VX_MIN * (1 if self._facing_r else -1)

        return vx, vy

    # ── Pagrindinis update ────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, landmarks,
               phase: str = None,
               facing_right: bool = None,
               camera_ratio: float = 0.7) -> np.ndarray:
        h, w = frame.shape[:2]

        if landmarks is None:
            self._draw(frame)
            return frame

        lm = landmarks
        try:
            if facing_right is not None:
                self._facing_r = facing_right
            else:
                self._update_facing(lm)

            wrist_pos = self._wrist_pos(lm, w, h)
            idle_pos  = self._idle_pos(lm, w, h)
        except (IndexError, AttributeError) as e:
            print(f"[BALL] landmarks klaida: {e}")
            self._draw(frame)
            return frame

        # ── FLYING — fizikos žingsnis ─────────────────────────────────────────
        if self._state == 'flying':
            self.x += self.vx * FLY_DT
            self.y += self.vy * FLY_DT
            self.vy += GRAVITY * FLY_DT
            out = (self.x > w + BALL_RADIUS or
                   self.x < -BALL_RADIUS or
                   self.y < -BALL_RADIUS * 4 or
                   self.y > h + BALL_RADIUS)
            if out:
                self._state = 'waiting_idle'
                self._released = False
                self._hands_down_since = None
                self._clear_wrist()
                print(f"[BALL] WAITING_IDLE after flying")
            self._draw(frame)
            return frame

        # ── WAITING_IDLE — laukiame fiksuotą laiką po metimo ────────────────
        if self._state == 'waiting_idle':
            if self._hands_down_since is None:
                self._hands_down_since = time.time()
            elapsed = time.time() - self._hands_down_since
            if elapsed >= self._HANDS_DOWN_DELAY:
                # Laikas baigėsi — grįžtame į idle ir TĘSIAME šį kadrą
                self._state = 'idle'
                self._released = False
                self._hands_down_since = None
                self.x, self.y = idle_pos[0], idle_pos[1]
                print("[BALL] IDLE (grįžo)")
                # NE return — leidžiame phase logikai veikti šiame kadre
            else:
                # Dar laukiame — nepiešiame
                return frame

        # ── RESTING / None — kamuolys idle ore ───────────────────────────────
        if phase in ('RESTING', None, 'UNKNOWN'):
            self._state    = 'idle'
            self._released = False
            self._hands_down_since = None
            self._clear_wrist()
            self.x, self.y = idle_pos[0], idle_pos[1]

        # ── READY — žmogus pakėlė rankas, kamuolys paimamas ──────────────────
        elif phase == 'READY':
            if self._state == 'idle':
                self._state    = 'held'
                self._released = False
                self._clear_wrist()
            if self._state == 'held':
                self.x = wrist_pos[0]
                self.y = wrist_pos[1]
                self._push_wrist(wrist_pos)
                self._hands_down_since = None

        # ── LOADING — kamuolys laikomas, kyla ────────────────────────────────
        elif phase == 'LOADING':
            if self._state == 'idle':
                self._state    = 'held'
                self._released = False
                self._clear_wrist()

            if self._state == 'held':
                if self._hands_are_down(lm, h):
                    if self._hands_down_since is None:
                        self._hands_down_since = time.time()
                    if time.time() - self._hands_down_since >= self._HANDS_DOWN_DELAY:
                        self._state    = 'idle'
                        self._released = False
                        self._hands_down_since = None
                        self._clear_wrist()
                        self.x, self.y = idle_pos[0], idle_pos[1]
                        print("[BALL] IDLE (rankos žemiau ≥1s)")
                    else:
                        self.x = wrist_pos[0]
                        self.y = wrist_pos[1]
                        self._push_wrist(wrist_pos)
                else:
                    self._hands_down_since = None
                    self.x = wrist_pos[0]
                    self.y = wrist_pos[1]
                    self._push_wrist(wrist_pos)

        # ── RELEASE — paleisti ────────────────────────────────────────────────
        elif phase == 'RELEASE':
            if not self._released and self._state == 'held':
                self._push_wrist(wrist_pos)
                self.vx, self.vy = self._launch_velocity()
                self._state    = 'flying'
                self._released = True
                self._clear_wrist()
                print(f"[BALL] FLYING vx={self.vx:.0f} vy={self.vy:.0f}")

        self._draw(frame)
        return frame

    # ── Piešimas ─────────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray):
        cx = int(self.x)
        cy = int(self.y)
        r  = BALL_RADIUS
        fh, fw = frame.shape[:2]
        if cx + r < 0 or cx - r > fw or cy + r < 0 or cy - r > fh:
            return
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), r, BALL_COLOR, -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), r, BALL_BORDER, 2, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), int(r*0.55), BALL_BORDER, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), BALL_DOT_R, BALL_DOT_COLOR, -1, cv2.LINE_AA)
        gx = cx - int(r * 0.3)
        gy = cy - int(r * 0.3)
        cv2.circle(overlay, (gx, gy), int(r*0.18), (220, 255, 220), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, BALL_ALPHA, frame, 1 - BALL_ALPHA, 0, frame)
        cv2.circle(frame, (cx, cy), r, BALL_BORDER, 2, cv2.LINE_AA)
