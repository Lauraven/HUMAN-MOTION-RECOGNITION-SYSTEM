"""
Basketball Shot Analyser
========================
Paleidimas:
  python main.py

Valdymas:
  Q — išeiti
  R — atstatyti skaitliukus
  F — įjungti/išjungti veidrodį
  P — rodyti grafikus
"""

import argparse
import os
import logging
import threading
import time
import collections
from datetime import datetime
import cv2
import numpy as np
import queue

from process_frame import ProcessFrame
from data_logger   import DataLogger
from plotter       import DataPlotter
from utils         import get_mediapipe_pose
from ui            import BasketballApp

# Išjungti Kivy debug žinutes prieš importuojant
os.environ['KIVY_NO_ENV_CONFIG'] = '1'
os.environ['KIVY_LOG_LEVEL'] = 'warning'
os.environ['KIVY_NO_ARGS'] = '1'
logging.getLogger('kivy').setLevel(logging.WARNING)
logging.getLogger('kivy.core.text').setLevel(logging.CRITICAL)
logging.getLogger('kivy.core.text.text_pygame').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)


class RealFpsTracker:
    def __init__(self, window=30):
        self._times = collections.deque(maxlen = window)

    def tick(self):
        self._times.append(time.perf_counter())

    def get(self):
        if len(self._times) < 2:
            return 25.0
        elapsed = self._times[-1] - self._times[0]
        return max(1.0, (len(self._times) - 1) / elapsed)

def parse_args():
    p = argparse.ArgumentParser(description='Basketball Shot Analyser')
    p.add_argument('--source',  default=4,     #default for laptops 0
                   help='4 = orbbec camera')
    p.add_argument('--flip',    action='store_true',
                   help='Flip frame horizontally')
    p.add_argument('--no-log',  action='store_true',
                   help='Data not saved')
    return p.parse_args()


def main():
    args   = parse_args()
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] Failed to open source: {source}')
        return

    pose = get_mediapipe_pose()
    processor = ProcessFrame(flip_frame = args.flip)

    session_ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    session_dir = os.path.join('data', f'{session_ts}_shotdata')
    os.makedirs(session_dir, exist_ok=True)
    print(f'[INFO] Session folder: {session_dir}')

    os.makedirs('data', exist_ok=True)
    logger = None if args.no_log else DataLogger(output_dir = 'data')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    plotter   = DataPlotter(fps = fps)

    flip_flag = args.flip
    start_time = time.perf_counter()
    shot_num = 0
    prev_phase = None
    running = True
    real_fps_tracker = RealFpsTracker(window = 30)


    # ── Klavišų callback (Kivy iškviečia iš savo thread'o) ───────────────────
    def on_key(key):
        nonlocal flip_flag, running
        if key == 'q':
            running = False
            cap.release()
            app.stop()
        elif key == 'r':
            processor.reset_counters()
            print('[INFO] Counters reset.')
        elif key == 'f':
            flip_flag = not flip_flag
            processor.flip_frame = flip_flag
            print(f'[INFO] Mirror: {"ON" if flip_flag else "OFF"}')
        elif key == 'p':
            print('[INFO] Showing graphs...')
            threading.Thread(target = plotter.show, daemon = True).start()

    # ── Kamera thread ─────────────────────────────────────────────────────────────
    raw_queue = queue.Queue(maxsize = 1)
    result_queue = queue.Queue(maxsize = 1)

    def capture_loop():
        while running:
            ret, frame = cap.read()
            if not ret:
                print('[INFO] Video source has ended.')
                app.stop()
                break
            try:
                raw_queue.put_nowait(frame)
            except queue.Full:
                pass

    def inference_loop():
        nonlocal shot_num, prev_phase, running

        while running:
            try:
                frame = raw_queue.get(timeout = 1.0)
            except queue.Empty:
                continue

            timestamp_s = time.perf_counter() - start_time
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_rgb = processor.process(frame_rgb, pose)
            real_fps_tracker.tick()

            # ── Meto registravimas ────────────────────────────────────
            cur_phase = processor.last_phase

            if cur_phase == 'LOADING' and prev_phase not in ('LOADING', 'RELEASE'):
                if logger:
                    logger.start_shot()
            if cur_phase == 'RESTING' and prev_phase in ('LOADING', 'RELEASE', 'READY'):
                if logger:
                    logger.end_shot()

            if processor.shot_just_finished:
                shot_num += 1
                if logger:
                    logger.log_shot(shot_num, processor.shot_result, timestamp_s, processor.last_shot_errors)
                    logger.reset_shot_buf()
                print(f'[SHOT #{shot_num}] {processor.shot_result}')

            prev_phase = cur_phase

            if logger and processor.last_landmarks:
                logger.log_frame(
                    timestamp_s = timestamp_s,
                    shot_num = shot_num,
                    phase = processor.last_phase,
                    landmarks = processor.last_landmarks,
                    angles = processor.last_angles,
                    ball_state = 'IDLE',
                    pose_correct = processor.last_pose_ok,
                    feedback_flags = processor.last_feedback,
                    shot_result = processor.shot_result if processor.shot_just_finished else None,
                )

            plotter.collect(timestamp_s, processor.last_landmarks, processor.last_angles)

            try:
                result_queue.put_nowait((processed_rgb, processor))
            except queue.Full:
                pass

    def ui_loop():
        while running:
            try:
                processed_rgb, proc = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            app.schedule_update(processed_rgb, proc)

        cap.release()
        if logger:
            logger.close()

    # ── Klavišų callback (Kivy iškviečia iš savo thread'o) ───────────────────
    def on_key(key):
        nonlocal flip_flag, running
        if key == 'q':
            running = False
            cap.release()
            app.stop()
        elif key == 'r':
            processor.reset_counters()
            print('[INFO] Counters reset.')
        elif key == 'f':
            flip_flag = not flip_flag
            processor.flip_frame = flip_flag
            print(f'[INFO] Mirror: {"ON" if flip_flag else "OFF"}')
        elif key == 'p':
            print('[INFO] Showing graphs...')
            threading.Thread(target=plotter.show, daemon=True).start()

    # ── Paleidimas ────────────────────────────────────────────────────────────
    print('[INFO] System started. Q=quit  R=reset  F=mirror  P=graphs')

    app = BasketballApp(key_callback=on_key)

    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    ui_thread = threading.Thread(target=ui_loop, daemon=True)

    cap_thread.start()
    inference_thread.start()
    ui_thread.start()

    app.run()

    running = False
    cap_thread.join(timeout = 5.0)
    inference_thread.join(timeout = 5.0)
    ui_thread.join(timeout = 5.0)

    if plotter and len(plotter._times) > 0:
        plotter.save(session_dir)
        print(f'[INFO] Graphs saved to {session_dir}/')

if __name__ == '__main__':
    main()