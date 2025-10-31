from modules.common import config
from modules.common.paths import VIDEOS_DIR, get_raw_csv
from modules.common.io_utils import write_csv
from modules.common.feature_builder import build_features, SQUAT_ALL_FEATURES
from modules.data_extraction.mediapipe_runner import PoseRunner
from modules.data_extraction.yolo_runner import YoloRunner

import numpy as np
import pandas as pd
import cv2
import argparse

from pathlib import Path
from typing import List, Dict, Any, Optional

BASE_COLUMNS = ["video_name", "frame_idx", "t_s", "rep_id"]

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-frame features into CSV.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g. 'squat')")
    p.add_argument("--verbose", action="store_true", help="Print progress")
    return p.parse_args()

def list_videos(exercise: str) -> List[Path]:
    vid_dir = VIDEOS_DIR / exercise
    supported = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV"}
    videos = [p for p in vid_dir.iterdir() if p.suffix in supported]

    return videos

def build_row(video_name: str, frame_idx: int, t_s: float, feats: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "video_name": video_name,
        "frame_idx": frame_idx,
        "t_s": float(t_s),
        "rep_id": ""  
    }
    
    for f in SQUAT_ALL_FEATURES:
        row[f] = feats.get(f, np.nan)
    
    return row

def process_video(video_path: Path, exercise: str, yolo: YoloRunner, pose: PoseRunner,
                  fps_target: int, verbose: bool = False) -> pd.DataFrame:

    cap = cv2.VideoCapture(str(video_path))
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
        lm = pose.process_bgr(frame)
        
        if lm is None:
            continue  # Skip frames without pose

        det = yolo.infer_bgr(frame)
        feats = build_features(lm, det, exercise_type=exercise)
        row = build_row(video_path.name, kept_idx, t_s, feats)
        rows.append(row)

    cap.release()

    cols = BASE_COLUMNS + SQUAT_ALL_FEATURES
    df = pd.DataFrame(rows, columns=cols)

    return df

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    out_csv = get_raw_csv(exercise)
    mode = "w" if args.overwrite else "a"

    videos = list_videos(exercise)
    yolo = YoloRunner(device=args.device)

    with PoseRunner() as pose:
        total_rows = 0
        header_needed = True
        
        if out_csv.exists() and mode == "a":
            header_needed = out_csv.stat().st_size == 0

        for i, vp in enumerate(videos, 1):
            df = process_video(vp, exercise, yolo, pose, config.FPS_TARGET, verbose=args.verbose)
            
            if df.empty:
                continue

            write_csv(df, out_csv, mode=mode, header=header_needed)
            header_needed = False
            mode = "a"
            total_rows += len(df)

if __name__ == "__main__":
    main()