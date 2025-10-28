#!/usr/bin/env python3
"""
Extract per-frame features for squat exercise.

✅ Extracts 7 training features + 2 metadata features
✅ YOLO runs independently to detect barbell presence
✅ Output: outputs/raw/squat_frames.csv

Usage:
  python modules/data_extraction/extract_pose_features.py --exercise squat --verbose
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
from modules.common.feature_builder import build_features, SQUAT_ALL_FEATURES
from .mediapipe_runner import PoseRunner
from .yolo_runner import YoloRunner

BASE_COLUMNS = ["video_name", "frame_idx", "t_s", "rep_id"]

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-frame features into CSV.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g. 'squat')")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output CSV")
    p.add_argument("--device", type=str, default="", help="YOLO device (e.g. '0' or 'cpu')")
    p.add_argument("--max-videos", type=int, default=0, help="Limit videos (0=all)")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()

def list_videos(exercise: str) -> List[Path]:
    """✅ List videos with error handling."""
    vid_dir = VIDEOS_DIR / exercise
    
    if not vid_dir.exists():
        print(f"\n❌ ERROR: Videos folder not found!")
        print(f"   Expected: {vid_dir}")
        print(f"   Absolute: {vid_dir.absolute()}")
        print(f"\n💡 Fix: Create folder and add videos:")
        print(f"   mkdir -p {vid_dir}")
        raise SystemExit(1)
    
    supported = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
    videos = [p for p in vid_dir.iterdir() if p.suffix in supported]
    
    if not videos:
        print(f"\n❌ ERROR: No videos found in {vid_dir}")
        print(f"\n📁 Current contents:")
        for item in vid_dir.iterdir():
            print(f"   - {item.name}")
        raise SystemExit(2)
    
    videos.sort()
    print(f"\n✅ Found {len(videos)} video(s)")
    for v in videos[:5]:
        print(f"   - {v.name}")
    if len(videos) > 5:
        print(f"   ... and {len(videos)-5} more")
    
    return videos

def build_row(video_name: str, frame_idx: int, t_s: float, feats: Dict[str, Any]) -> Dict[str, Any]:
    """✅ Build CSV row with enforced column order."""
    row = {
        "video_name": video_name,
        "frame_idx": frame_idx,
        "t_s": float(t_s),
        "rep_id": ""  # filled later by derive_reps
    }
    
    # Add features in canonical order
    for f in SQUAT_ALL_FEATURES:
        row[f] = feats.get(f, np.nan)
    
    return row

def process_video(video_path: Path, exercise: str, yolo: YoloRunner, pose: PoseRunner,
                  fps_target: int, verbose: bool = False) -> pd.DataFrame:
    """✅ Process one video file."""
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return pd.DataFrame()
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Failed to open: {video_path}")
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

        t_s = frame_idx / float(native_fps)

        # ✅ MediaPipe Pose
        lm = pose.process_bgr(frame)
        if lm is None:
            continue  # Skip frames without pose

        # ✅ YOLO detections (independent)
        det = yolo.infer_bgr(frame)

        # ✅ Build features (7 + 2 metadata)
        feats = build_features(lm, det, exercise_type=exercise)

        row = build_row(video_path.name, kept_idx, t_s, feats)
        rows.append(row)

    cap.release()

    if verbose:
        print(f"   ✅ {video_path.name}: {len(rows)} frames (fps={fps_target})")

    if not rows:
        print(f"⚠️  No valid frames: {video_path.name}")
        return pd.DataFrame()

    cols = BASE_COLUMNS + SQUAT_ALL_FEATURES
    df = pd.DataFrame(rows, columns=cols)
    return df

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    
    if exercise != "squat":
        print(f"❌ Only 'squat' is configured. Got: {exercise}")
        raise SystemExit(1)

    out_csv = get_raw_csv(exercise)
    mode = "w" if args.overwrite else "a"

    videos = list_videos(exercise)
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    print("="*60)
    print("🏋️  DATA EXTRACTION - Squat")
    print("="*60)
    print(f"Videos       : {len(videos)}")
    print(f"Output       : {out_csv}")
    print(f"FPS target   : {config.FPS_TARGET}")
    print(f"YOLO weights : {config.YOLO_WEIGHTS}")
    print("="*60)

    # Initialize
    yolo = YoloRunner(device=args.device)
    with PoseRunner() as pose:
        total_rows = 0
        header_needed = True
        
        if out_csv.exists() and mode == "a":
            header_needed = out_csv.stat().st_size == 0

        for i, vp in enumerate(videos, 1):
            if args.verbose:
                print(f"\n[{i}/{len(videos)}] {vp.name}")

            df = process_video(vp, exercise, yolo, pose, config.FPS_TARGET, verbose=args.verbose)
            
            if df.empty:
                continue

            write_csv(df, out_csv, mode=mode, header=header_needed)
            header_needed = False
            mode = "a"
            total_rows += len(df)

        print(f"\n✅ SUCCESS! Total rows: {total_rows}")
        print(f"📁 Output: {out_csv}")

if __name__ == "__main__":
    main()