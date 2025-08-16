import os, sys, cv2, math, time, shutil, re, tempfile, random, pickle
import numpy as np
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms, models

from PIL import Image, ImageDraw
from scipy.signal import savgol_filter, find_peaks

import tensorflow as tf
from tensorflow.python.keras.models import load_model

from ultralytics import YOLO, SAM

# Add custom module path (if needed)
sys.path.append("/home/ubuntu/API")

# Custom module imports
from ball_tracker import BallTracker, combine_three_frames
from ball_analysis import *

from player_tracker import PlayerTracker
from player_analysis import *

from court_tracker import CourtTracker
from game_tracker import Game

from filter_and_tracker_helpers.initialization import *
from filter_and_tracker_helpers.frame_processing import *
from filter_and_tracker_helpers.tracking import *
from filter_and_tracker_helpers.post_processing import *

from heatmap import *

def filter_and_track(input_path, thresh=0.03, sample_rate=1, batch_size=30):
    # === Constants ===
    MINIMAP_SCALE = 10              # Multiplier to convert real-world meters to pixel units for the minimap overlay
    MINIMAP_MARGIN = 20             # Pixel padding around the minimap to avoid rendering elements at the edge
    EXTRA_MARGIN = 2.0              # Additional margin (in meters) added beyond the court size for drawing space
    SMOOTHING_ALPHA = 0.3           # Smoothing factor (0–1) used to dampen jitter in player position projections
    WARMUP_FRAMES = 15              # Number of frames to process before starting main tracking logic (for smoothing/init)
    HEATMAP_UPDATE_INTERVAL = 5     # Frequency (in frames) at which the heatmap is recomputed and updated

    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_tracked{ext}"
  
    # === Timers ===
    timers = {k: 0.0 for k in [
        'init', 'frame_reading', 'frame_filtering', 'frame_processing',
        'warmup_flush', 'drawing_and_writing', 'heatmap_tracking',
        'video_output_writing', 'post_processing', 'cleanup', 'total'
    ]}
    step_timers = {}

    def mark_step(label):
        step_timers[label] = time.perf_counter()

    def stop_step(label):
        if label in step_timers:
            timers[label] += time.perf_counter() - step_timers[label]
            del step_timers[label]

    total_start = time.perf_counter()

    # === Models ===
    model_paths = {
        'player': "yolov8s.pt",
        'ball': "balltrackernet_traced.pt",
        'court': "court10.pt",
        'bounce': "bounce_classifier.pt"
    }

    mark_step('init')
    cap, out, fps, width, height = initialize_video_io(input_path, output_path)

    minimap_out_path = "/tmp/heatmap_players.mp4"
    os.makedirs(os.path.dirname(minimap_out_path), exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    minimap_writer = cv2.VideoWriter(minimap_out_path, fourcc, fps, (width, height))

    ball_tracker, player_tracker, court_tracker = initialize_trackers(
        model_paths['ball'], model_paths['player'], model_paths['court']
    )

    _ = detect_ball(np.zeros((height, width, 3), dtype=np.uint8), ball_tracker)
    _ = detect_players(np.zeros((height, width, 3), dtype=np.uint8), player_tracker)
    _ = detect_court(np.zeros((height, width, 3), dtype=np.uint8), court_tracker)
    stop_step('init')

    # === Heatmap & layout ===
    mini_w = int(11 * MINIMAP_SCALE)
    mini_h = int((23.77 + EXTRA_MARGIN) * MINIMAP_SCALE)
    off_x = (width - mini_w) // 2
    off_y = (height - mini_h) // 2
    GLOBAL_HEATMAP = np.zeros((height, width), dtype=np.float32)
    player_coord_smoother = {}
    heat_color = np.zeros((height, width, 3), dtype=np.uint8)

    early_frame_buffer = []
    top_two_players_dicts = {}
    frame_idx, kept_count = 0, 0
    wrote_early_frames = False

    while True:
        mark_step('frame_reading')
        frames, indices = [], []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize_frame(frame, width, height)
            frames.append(frame)
            indices.append(frame_idx)
            frame_idx += 1
        stop_step('frame_reading')
        if not frames:
            break

        for i, frame in enumerate(frames):
            idx = indices[i]
            mark_step('frame_filtering')
            keep, prob = should_keep_frame(idx, frame, sample_rate, thresh)
            stop_step('frame_filtering')
            if not keep:
                continue

            kept_count += 1
            mark_step('frame_processing')
            ball_pos = detect_ball(frame, ball_tracker)
            all_players, scale_x, scale_y = detect_players(frame, player_tracker)
            keypoints, H = detect_court(frame, court_tracker)
            court_tracker.store_homography(H, idx)
            stop_step('frame_processing')

            if idx < WARMUP_FRAMES:
                early_frame_buffer.append((frame.copy(), keypoints, all_players))
                continue

            if idx == WARMUP_FRAMES and not wrote_early_frames:
                mark_step('warmup_flush')
                for j, (e_frame, e_keypoints, e_players) in enumerate(early_frame_buffer):
                    processed = process_warmup_frames(
                        e_frame, e_keypoints, e_players, idx=j,
                        minimap_scale=MINIMAP_SCALE,
                        minimap_margin=MINIMAP_MARGIN,
                        extra_m=EXTRA_MARGIN,
                        player_tracker=player_tracker
                    )
                    out.write(processed)
                early_frame_buffer.clear()
                wrote_early_frames = True
                stop_step('warmup_flush')

            mark_step('drawing_and_writing')
            frame = draw_fixed_minimap(frame, scale=MINIMAP_SCALE, margin=MINIMAP_MARGIN, extra_m=EXTRA_MARGIN)
            
            player_dict = get_top_two_players(all_players, idx, H)
            print("Top 2 Players")
            print(player_dict)
            print("All Players")
            print(all_players)
            top_two_players_dicts[idx] = player_dict.copy()

            frame = player_tracker.draw_bbox_on_frame(frame, player_dict)
            frame = draw_bbox_with_threshold(frame, player_dict)

            if H is not None:
                coords = []
                for pid, bbox in player_dict.items():
                    p = project_player_to_court(bbox, H)
                    if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2:
                        p = np.array(p, dtype=np.float32)
                        prev = player_coord_smoother.get(pid, p)
                        p = SMOOTHING_ALPHA * p + (1 - SMOOTHING_ALPHA) * prev
                        player_coord_smoother[pid] = p
                        coords.append(p)

                frame = draw_players_on_minimap(frame, coords, scale=MINIMAP_SCALE, margin=MINIMAP_MARGIN, extra_m=EXTRA_MARGIN)

                for x_m, y_m in coords:
                    x_px = int(x_m * MINIMAP_SCALE + off_x)
                    y_px = int(y_m * MINIMAP_SCALE + off_y)
                    if 0 <= x_px < width and 0 <= y_px < height:
                        GLOBAL_HEATMAP[y_px, x_px] += 1.0

                # === Player Heatmap Logic ===
                if idx % HEATMAP_UPDATE_INTERVAL == 0:
                  mark_step('heatmap_tracking')
                  blurred = cv2.GaussianBlur(GLOBAL_HEATMAP, (101, 101), 0)
                  m = blurred.max() or 1.0
                  hm8 = (blurred / m * 255).astype(np.uint8)
                  heat_color = cv2.applyColorMap(hm8, cv2.COLORMAP_HOT)
                  stop_step('heatmap_tracking')

                  mini = np.zeros((height, width, 3), dtype=np.uint8)
                  mini = cv2.addWeighted(mini, 0.0, heat_color, 1.0, 0)
                  mini = draw_fixed_minimap_centered(mini)
                  mini = draw_players_on_minimap_centered(mini, coords)
                  minimap_writer.write(mini)
                
            frame = draw_frame_number(frame, idx)
            stop_step('drawing_and_writing')

            mark_step('video_output_writing')
            out.write(frame)
            stop_step('video_output_writing')

    mark_step('cleanup')
    out.release()
    cap.release()
    minimap_writer.release()

    stop_step('cleanup')

    mark_step('post_processing')
    post_process_ball(ball_tracker)
    zone_stats, return_stats, annotated_path, hits = run_game_tracking(
        ball_tracker=ball_tracker,
        court_tracker=court_tracker,
        bounce_classifier_path=model_paths['bounce'],
        top_two_players_dicts=top_two_players_dicts,
        output_path=output_path,
        fps=fps, width=width, height=height
    )
    stop_step('post_processing')

    stroke_counts = {
    "near": {"forehand": 0, "backhand": 0},
    "far": {"forehand": 0, "backhand": 0}
    }

    for frame_idx in sorted(hits):
        if frame_idx not in top_two_players_dicts or frame_idx >= len(ball_tracker.detections):
            continue

        players = top_two_players_dicts[frame_idx]
        ball_x, ball_y = ball_tracker.detections[frame_idx]

        if ball_x is None or ball_y is None or len(players) != 2:
            continue

        # Identify near and far players by bbox y2 (lower = farther away)
        sorted_players = sorted(players.items(), key=lambda item: item[1][3])
        far_player = sorted_players[0]  # smaller y2
        near_player = sorted_players[1]  # larger y2

        # Compute distance from ball to bbox center
        closest = None
        min_dist = float('inf')
        for role, (_, (x1, y1, x2, y2)) in zip(["far", "near"], [far_player, near_player]):
            if None in [x1, y1, x2, y2]:
                continue
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            dist = (ball_x - mid_x) ** 2 + (ball_y - mid_y) ** 2
            if dist < min_dist:
                min_dist = dist
                closest = (role, mid_x)

        if closest:
            role, mid_x = closest
            kind = "forehand" if ball_x < mid_x else "backhand"
            stroke_counts[role][kind] += 1

    timers['total'] = time.perf_counter() - total_start

    print(f"\n✅ Final annotated video saved at: {annotated_path}")
    print("\n⏲️ Time Summary:")
    for k, t in timers.items():
        print(f"{k:<30} {t:.2f}")

    return annotated_path, stroke_counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter & track tennis video")
    parser.add_argument("input", help="Path to input video")
    parser.add_argument("--thresh", type=float, default=0.03)
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=30)
    args = parser.parse_args()

    out = filter_and_track(args.input, args.thresh, args.sample_rate, args.batch_size)
    if out:
        print(f"✅ Done! Tracked video at {out}")
    else:
        print("❌ Error")
