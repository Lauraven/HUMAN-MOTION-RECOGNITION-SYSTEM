"""
utils.py — pagalbinės funkcijos.
Naudoja MediaPipe Tasks API (>= 0.10) su pose_landmarker_full.task failu.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mpp
from mediapipe.tasks.python import vision as _mpv
from mediapipe import Image, ImageFormat


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_rounded_rect(img, rect_start, rect_end, corner_width, box_color):
    x1, y1 = rect_start
    x2, y2 = rect_end
    w = corner_width
    cv2.rectangle(img, (x1+w, y1), (x2-w, y1+w), box_color, -1)
    cv2.rectangle(img, (x1+w, y2-w), (x2-w, y2), box_color, -1)
    cv2.rectangle(img, (x1, y1+w), (x1+w, y2-w), box_color, -1)
    cv2.rectangle(img, (x2-w, y1+w), (x2,   y2-w), box_color, -1)
    cv2.rectangle(img, (x1+w, y1+w), (x2-w, y2-w), box_color, -1)
    cv2.ellipse(img, (x1+w, y1+w), (w,w), 0, -90, -180, box_color, -1)
    cv2.ellipse(img, (x2-w, y1+w), (w,w), 0, 0, -90, box_color, -1)
    cv2.ellipse(img, (x1+w, y2-w), (w,w), 0, 90, 180, box_color, -1)
    cv2.ellipse(img, (x2-w, y2-w), (w,w), 0, 0, 90, box_color, -1)
    return img


def draw_dotted_line(frame, lm_coord, start, end, line_color):
    for i in range(start, end+1, 8):
        cv2.circle(frame, (lm_coord[0], i), 2, line_color, -1, lineType=cv2.LINE_AA)
    return frame


def draw_text(img, msg, width = 8, font = cv2.FONT_HERSHEY_SIMPLEX, pos = (0, 0),
              font_scale = 1, font_thickness = 2, text_color = (0, 255, 0),
              text_color_bg = (0, 0, 0), box_offset = (20, 10)):
    offset = box_offset
    x, y = pos
    text_size, _ = cv2.getTextSize(msg, font, font_scale, font_thickness)
    text_w, text_h = text_size
    rec_start = tuple(p - o for p, o in zip(pos, offset))
    rec_end = tuple(m+n-o for m, n, o in zip((x+text_w, y+text_h), offset, (25, 0)))
    img = draw_rounded_rect(img, rec_start, rec_end, width, text_color_bg)
    cv2.putText(img, msg,
                (int(rec_start[0]+6), int(y+text_h+font_scale-1)),
                font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    return text_size


# ── Math ───────────────────────────────────────────────────────────────────────

def find_angle(p1, p2, ref_pt=np.array([0, 0])):
    p1_ref = p1 - ref_pt
    p2_ref = p2 - ref_pt
    denom = np.linalg.norm(p1_ref) * np.linalg.norm(p2_ref)
    if denom == 0:
        return 0
    cos_theta = np.dot(p1_ref, p2_ref) / denom
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return int(int(180 / np.pi) * theta)


# ── Landmarks ─────────────────────────────────────────────────────────────────

def get_landmark_array(pose_landmark, key, frame_width, frame_height):
    return np.array([int(pose_landmark[key].x * frame_width),
                     int(pose_landmark[key].y * frame_height)])


def get_landmark_features(kp_results, dict_features, feature, frame_width, frame_height):
    if feature == 'nose':
        return get_landmark_array(kp_results, dict_features[feature], frame_width, frame_height)
    elif feature in ('left', 'right'):
        f = dict_features[feature]
        return tuple(
            get_landmark_array(kp_results, f[k], frame_width, frame_height)
            for k in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'foot')
        )
    else:
        raise ValueError("feature must be 'nose', 'left', or 'right'")


# ── Pose wrapper (Tasks API) ─────────────────────────────────────────────────

class _NewPoseLandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _NewPoseResult:
    def __init__(self, landmarks_list):
        self.pose_landmarks = _NewPoseLandmarkList(landmarks_list[0]) if landmarks_list else None


class _NewPoseWrapper:
    def __init__(self, model_path = 'pose_landmarker_full.task',
                 min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        options = _mpv.PoseLandmarkerOptions(
            base_options = _mpp.BaseOptions(model_asset_path = model_path),
            running_mode = _mpv.RunningMode.VIDEO,
            min_pose_detection_confidence = min_detection_confidence,
            min_tracking_confidence = min_tracking_confidence,
        )
        self._lm = _mpv.PoseLandmarker.create_from_options(options)
        self._ts = 0

    def process(self, frame_rgb):
        self._ts += 33
        r = self._lm.detect_for_video(
            Image(image_format = ImageFormat.SRGB, data = frame_rgb), self._ts)
        return _NewPoseResult(r.pose_landmarks if r.pose_landmarks else [])


def get_mediapipe_pose(static_image_mode = False, model_complexity = 1,
                       smooth_landmarks = True, min_detection_confidence = 0.5,
                       min_tracking_confidence = 0.5, model_path = 'pose_landmarker_full.task'):
    print('[INFO] MediaPipe: Tasks API — naudoja ' + model_path)
    return _NewPoseWrapper(
        model_path = model_path,
        min_detection_confidence = min_detection_confidence,
        min_tracking_confidence = min_tracking_confidence,
    )