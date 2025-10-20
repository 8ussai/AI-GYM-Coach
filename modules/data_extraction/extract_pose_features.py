#!/usr/bin/env python3
"""
Extract per-frame features for a given exercise and append rows into outputs/raw/<exercise>_frames.csv.

- Scans videos/<exercise>/
- Runs MediaPipe Pose + YOLO (class_id=1 presence used for squat bar on shoulder)
- Builds features via modules.common.feature_builder.build_features(...)
- Writes CSV rows: video_name, frame_idx, t_s, rep_id, <feature columns...>
- Frame-rate downsampled to config.FPS_TARGET

Usage:
  python modules/data_extraction/extract_pose_features.py --exercise squat
  python modules/data_extraction/extract_pose_features.py --exercise squat --overwrite
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pandas as pd

from modules.common import config
from modules.common.paths import VIDEOS_DIR, get_raw_csv
from modules.common.io_utils import write_csv
from modules.common.feature_builder import build_features
from .mediapipe_runner import PoseRunner
from .yolo_runner import YoloRunner

# Canonical feature ordering for squat (must align with feature_builder output names)
SQUAT_FEATURES: List[str] = [
    "sq_knee_angle_L", "sq_knee_angle_R",
    "sq_torso_incline","sq_pelvis_drop","sq_stance_ratio",
    "sq_elbow_angle_L","sq_elbow_angle_R",
    "sq_bar_present",
    "pose_confidence",
]

BASE_COLUMNS = ["video_name", "frame_idx", "t_s", "rep_id"]  # rep_id filled later by data_processing

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-frame features into a single CSV per exercise.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g. 'squat')")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output CSV instead of appending")
    p.add_argument("--device", type=str, default="", help="YOLO device (e.g. '0' or 'cpu')")
    p.add_argument("--max-videos", type=int, default=0, help="Limit number of videos (0 means all)")
    p.add_argument("--verbose", action="store_true", help="Print per-video progress")
    return p.parse_args()

def list_videos(exercise: str) -> List[Path]:
    vid_dir = VIDEOS_DIR / exercise
    if not vid_dir.exists():
        raise FileNotFoundError(f"Videos folder not found: {vid_dir}")
    videos = [p for p in vid_dir.iterdir() if p.suffix.lower() in {".mp4",".mov",".avi",".mkv"}]
    videos.sort()
    return videos

def build_row(video_name: str, frame_idx: int, t_s: float, feats: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "video_name": video_name,
        "frame_idx": frame_idx,
        "t_s": float(t_s),
        "rep_id": ""  # filled later by derive_reps
    }
    # Enforce consistent feature order and fill missing with NaN
    for f in SQUAT_FEATURES:
        row[f] = feats.get(f, np.nan)
    return row

def process_video(video_path: Path, exercise: str, yolo: YoloRunner, pose: PoseRunner,
                  fps_target: int, verbose: bool = False) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return pd.DataFrame()

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0
    stride = max(1, int(round(native_fps / fps_target)))
    rows: List[Dict[str, Any]] = []

    frame_idx = -1
    kept_idx  = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        kept_idx += 1

        # Timestamp in seconds based on native fps and actual frame index
        t_s = frame_idx / float(native_fps)

        # Pose
        lm = pose.process_bgr(frame)
        if lm is None:
            # No pose; still record a row with NaNs except minimal fields?
            # We'll skip empty to avoid noisy rows.
            continue

        # YOLO detections
        det = yolo.infer_bgr(frame)

        # Build features
        feats = build_features(lm, det, exercise_type=exercise)

        # Assemble row
        row = build_row(video_path.name, kept_idx, t_s, feats)
        rows.append(row)

    cap.release()

    if verbose:
        print(f"[INFO] {video_path.name} -> kept {len(rows)} frames (target_fps={fps_target}, native_fps={native_fps:.2f})")

    if not rows:
        return pd.DataFrame()

    # DataFrame with stable column order
    cols = BASE_COLUMNS + SQUAT_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    return df

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    if exercise != "squat":
        print(f"[ERROR] Only 'squat' is configured right now in feature_builder. Got: {exercise}")
        raise SystemExit(1)

    out_csv = get_raw_csv(exercise)
    mode = "w" if args.overwrite else "a"

    videos = list_videos(exercise)
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    if not videos:
        print(f"[ERROR] No videos found under: {VIDEOS_DIR / exercise}")
        raise SystemExit(2)

    print("======== Data Extraction ========")
    print(f"exercise     : {exercise}")
    print(f"videos dir   : {VIDEOS_DIR / exercise}")
    print(f"num videos   : {len(videos)}")
    print(f"out csv      : {out_csv}")
    print(f"fps target   : {config.FPS_TARGET}")
    print(f"yolo weights : {config.YOLO_WEIGHTS}")
    print("=================================")

    # Initialize runners
    yolo = YoloRunner(device=args.device)
    with PoseRunner() as pose:
        total_rows = 0

        # If overwrite, drop existing file header
        header_needed = True
        if out_csv.exists() and mode == "a":
            # Appending: write header only if file is empty
            header_needed = out_csv.stat().st_size == 0

        for i, vp in enumerate(videos, 1):
            if args.verbose:
                print(f"[INFO] Processing ({i}/{len(videos)}): {vp.name}")

            df = process_video(vp, exercise, yolo, pose, config.FPS_TARGET, verbose=args.verbose)
            if df.empty:
                if args.verbose:
                    print(f"[WARN] No valid frames extracted from {vp.name}")
                continue

            write_csv(df, out_csv, mode=mode, header=header_needed)
            # After first write, all subsequent appends should not include header
            header_needed = False
            mode = "a"
            total_rows += len(df)

        print(f"[SUCCESS] Extraction finished. Total rows written: {total_rows}")
        print(f"[INFO] Output CSV: {out_csv}")

if __name__ == "__main__":
    main()
