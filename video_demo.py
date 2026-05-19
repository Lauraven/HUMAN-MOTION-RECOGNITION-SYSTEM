"""
video.py — Krepšinio metimo analizė iš video failo su Kivy UI.

Išvestis:
  video_demo/data/basename/basename_analyzed.mp4
  video_demo/data/basename/basketball_TIMESTAMP.xlsx
  video_demo/data/basename/graphs/*.png
"""

import argparse
import os
import sys
import time
import logging
import threading
import cv2
import subprocess, shutil

os.environ['KIVY_NO_ENV_CONFIG'] = '1'
os.environ['KIVY_LOG_LEVEL'] = 'warning'
os.environ['KIVY_NO_ARGS'] = '1'
logging.getLogger('kivy').setLevel(logging.WARNING)
logging.getLogger('kivy.core.text').setLevel(logging.CRITICAL)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.CRITICAL)

from process_frame import ProcessFrame
from data_logger import DataLogger
from plotter import DataPlotter
from utils import get_mediapipe_pose
from ui import BasketballApp

def parse_args():
    p = argparse.ArgumentParser(description='Basketball video analysis')
    p.add_argument('source', help = 'Kelias iki .mp4 failo')
    p.add_argument('--flip', action = 'store_true', help = 'Veidrodinis vaizdas')
    p.add_argument('--out', default = None, help = 'Išvesties failo pavadinimas')
    p.add_argument('--no-preview', action = 'store_true', help = 'Nerodyti peržiūros lango')
    p.add_argument('--no-log', action = 'store_true', help = 'Neišsaugoti Excel duomenų')
    p.add_argument('--no-plots', action = 'store_true', help = 'Nerodyti grafikų po baigimo')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source):
        print(f'[ERROR] Failas nerastas: {args.source}')
        sys.exit(1)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f'[ERROR] Nepavyko atidaryti: {args.source}')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Aplanko struktūra ────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(args.source))[0]
    session_dir = os.path.join('video_demo', 'data', base)
    graphs_dir = os.path.join(session_dir, 'graphs')
    os.makedirs(session_dir, exist_ok = True)
    os.makedirs(graphs_dir, exist_ok = True)

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(session_dir, f'{base}_analyzed.mp4')

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    pose = get_mediapipe_pose()
    processor = ProcessFrame(flip_frame = args.flip)
    logger = None if args.no_log else DataLogger(output_dir = session_dir)
    plotter = DataPlotter(fps = fps)

    flip_flag = args.flip
    shot_num = 0
    prev_phase = None
    running = True
    good_shots = 0
    bad_shots = 0
    frame_idx = 0
    start_time = time.perf_counter()

    print(f'[INFO] Analizuojama: {args.source}')
    print(f'[INFO] Kadrai: {total} | FPS: {fps:.1f} | {width}x{height}')
    print(f'[INFO] Išvestis: {out_path}')

    def on_key(key):
        nonlocal flip_flag, running
        if key == 'q':
            running = False
            cap.release()
            if not args.no_preview:
                app.stop()
        elif key == 'f':
            flip_flag = not flip_flag
            processor.flip_frame = flip_flag
            print(f'[INFO] Veidrodis: {"ĮJUNGTAS" if flip_flag else "IŠJUNGTAS"}')
        elif key == 'p':
            print('[INFO] Rodomi grafikai...')
            threading.Thread(target = plotter.show, daemon = True).start()

    def video_loop():
        nonlocal shot_num, prev_phase, running, good_shots, bad_shots, frame_idx

        while running:
            ret, frame = cap.read()
            if not ret:
                print('[INFO] Video baigėsi.')
                running = False
                if not args.no_preview:
                    app.stop()
                break

            timestamp_s = time.perf_counter() - start_time
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_rgb = processor.process(frame_rgb.copy(), pose)

            cur_phase = processor.last_phase
            if logger:
                if cur_phase == 'LOADING' and prev_phase not in ('LOADING', 'RELEASE'):
                    logger.start_shot()
                if cur_phase == 'RESTING' and prev_phase in ('LOADING', 'RELEASE', 'READY'):
                    logger.end_shot()

            if processor.shot_just_finished:
                shot_num += 1
                result = processor.shot_result
                if result == 'GOOD':
                    good_shots += 1
                else:
                    bad_shots += 1
                if logger:
                    logger.log_shot(shot_num, result, timestamp_s, processor.last_shot_errors)
                    logger.reset_shot_buf()
                print(f'[SHOT #{shot_num}] {result}')

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

            output_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

            if total > 0:
                progress = int(width * frame_idx / total)
                cv2.rectangle(output_bgr, (0, height - 4), (progress, height), (37, 99, 235), -1)

            writer.write(output_bgr)

            if not args.no_preview:
                app.schedule_update(processed_rgb, processor)

            frame_idx += 1
            if frame_idx % 150 == 0:
                pct = frame_idx / total * 100 if total > 0 else 0
                print(f'[INFO] {frame_idx}/{total} kadrų ({pct:.0f}%)')

        cap.release()
        writer.release()
        if logger:
            logger.close()

        if shutil.which('ffmpeg'):
            fixed_path = out_path.replace('.mp4', '_fixed.mp4')
            result = subprocess.run(
                ['ffmpeg', '-y', '-r', str(fps), '-i', out_path,
                 '-c', 'copy', '-r', str(fps), fixed_path],
                capture_output=True
            )
            if result.returncode == 0:
                os.replace(fixed_path, out_path)
                print(f'[INFO] FPS metaduomenys pataisyti ({fps:.1f} FPS)')
            else:
                print(f'[WARN] ffmpeg nepavyko: {result.stderr.decode()[-200:]}')
        else:
            print('[WARN] ffmpeg nerastas — video greitis gali būti neteisingas')

        print(f'\n[REZULTATAI]')
        print(f'  Iš viso metimų: {shot_num}')
        print(f'  GOOD: {good_shots}')
        print(f'  BAD: {bad_shots}')
        print(f'  Video: {out_path}')

        if not args.no_plots and len(plotter._times) > 0:
            plotter.save(graphs_dir)
            print(f'  Grafikai: {graphs_dir}/')
            if args.no_preview:
                print('[INFO] Rodomi grafikai (uždaryk langą kad baigti)...')
                # plotter.show()

    if args.no_preview:
        video_loop()
    else:
        print('[INFO] Sistema paleista. Q = išeiti  F = veidrodis  P = grafikai')
        app = BasketballApp(key_callback = on_key)

        vid_thread = threading.Thread(target = video_loop, daemon = True)
        vid_thread.start()

        app.run()

        running = False
        vid_thread.join(timeout = 5.0)

        if not args.no_plots and len(plotter._times) > 0:
            plotter.save(graphs_dir)

if __name__ == '__main__':
    main()