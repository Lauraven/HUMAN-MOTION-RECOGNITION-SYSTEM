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

# Landmark indeksai (Tasks API)
_LS = 11
_RS = 12
_LH = 23
_RH = 24
_LI = 19
_RI = 20

BALL_RADIUS = 38
BALL_COLOR = (180, 255, 180)
BALL_BORDER = (80, 200, 80)
BALL_ALPHA = 0.55
BALL_DOT_COLOR = (60, 180, 60)
BALL_DOT_R = 6

GRAVITY = 900.0
FLY_DT = 1 / 30.0

BALL_FRONT_OFFSET_X = 170 
BALL_FRONT_OFFSET_Y = -25 


class Ball:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

        self._state      = 'idle'
        self._prev_wrist = None
        self._released   = False

    def _between_hands(self, lm, w, h):
        lh = np.array([lm[_LI].x * w, lm[_LI].y * h])
        rh = np.array([lm[_RI].x * w, lm[_RI].y * h])
        return (lh + rh) / 2

    def _idle_pos(self, lm, w, h, facing_r):
        ls = np.array([lm[_LS].x * w, lm[_LS].y * h])
        rs = np.array([lm[_RS].x * w, lm[_RS].y * h])
        chest = (ls + rs) / 2
        direction = 1 if facing_r else -1
        return np.array([
            chest[0] + direction * BALL_FRONT_OFFSET_X,
            chest[1] + BALL_FRONT_OFFSET_Y
        ])

    def _hip_y(self, lm, h):
        return (lm[_LH].y * h + lm[_RH].y * h) / 2

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
            facing_r = facing_right if facing_right is not None else (lm[_LS].x < lm[_RS].x)
            mid_hand = self._between_hands(lm, w, h)
            hip_y = self._hip_y(lm, h)
            idle_pos = self._idle_pos(lm, w, h, facing_r)
        except (IndexError, AttributeError) as e:
            print(f"[BALL] landmarks klaida: {e}")
            self._draw(frame)
            return frame

        # ── FLYING — fizikos žingsnis ──────────────────────────────────────────
        if self._state == 'flying':
            self.x += self.vx * FLY_DT
            self.y += self.vy * FLY_DT
            self.vy += GRAVITY * FLY_DT

            out = (self.x > w + BALL_RADIUS or
                   self.x < -BALL_RADIUS or
                   self.y < -BALL_RADIUS * 4 or
                   self.y > h + BALL_RADIUS)

            if out:
                self._state = 'idle'
                self._released = False
                self._prev_wrist = None
                print("[BALL] IDLE (grįžo)")

            self._draw(frame)
            return frame

        # ── RESTING arba nežinoma — kamuolys nejuda ore ───────────────────────
        if phase in ('RESTING', None, 'UNKNOWN'):
            self._state = 'idle'
            self._released = False
            self._prev_wrist = None
            self.x, self.y = idle_pos[0], idle_pos[1]

        # ── READY — kamuolys tarp delnų ───────────────────────────────────────
        elif phase == 'READY':
            self._state = 'held'
            self._released = False
            self.x = mid_hand[0]
            self.y = mid_hand[1]
            self._prev_wrist = mid_hand.copy()

        # ── LOADING — kamuolys tarp delnų, kyla ──────────────────────────────
        elif phase == 'LOADING':
            if mid_hand[1] > hip_y:
                self._state = 'idle'
                self._released = False
                self._prev_wrist = None
                self.x, self.y = idle_pos[0], idle_pos[1]
            else:
                self._state = 'held'
                self.x = mid_hand[0]
                self.y = mid_hand[1]
                self._prev_wrist = mid_hand.copy()

        # ── RELEASE — paleisti kamuolį ────────────────────────────────────────
        elif phase == 'RELEASE':
            if not self._released and self._state == 'held':
                if self._prev_wrist is not None:
                    wx_vel = mid_hand[0] - self._prev_wrist[0]
                    wy_vel = mid_hand[1] - self._prev_wrist[1]
                else:
                    wx_vel = 0.0
                    wy_vel = 0.0

                self.vx = wx_vel / FLY_DT
                self.vy = wy_vel / FLY_DT

                if self.vy > -150:
                    self.vy = -800.0
                if abs(self.vx) < 80:
                    self.vx = 350.0 if facing_r else -350.0

                self._state = 'flying'
                self._released = True
                self._prev_wrist = None
                print(f"[BALL] FLYING vx={self.vx:.0f} vy={self.vy:.0f} facing_right={facing_r}")

        self._draw(frame)
        return frame

    # ── Piešimas ─────────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray):
        cx = int(self.x)
        cy = int(self.y)
        r = BALL_RADIUS
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