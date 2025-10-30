#!/usr/bin/env python3
"""
Clean and smooth raw frames CSV.

✅ Keeps only 7 training features + base columns
✅ Removes sq_bar_present (YOLO is independent)
✅ Uses pose_confidence for filtering, then drops it

Usage:
  python modules/data_processing/clean_data.py --exercise squat --verbose
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from modules.common.paths import get_raw_csv, get_cleaned_frames
from modules.common.io_utils import read_csv, write_csv
from modules.common.smoothing import smooth_dataframe
from modules.common.feature_builder import SQUAT_TRAINING_FEATURES

BASE_COLS = ["video_name", "frame_idx", "t_s", "rep_id"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean and smooth raw frames CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key")
    p.add_argument("--min-pose-conf", type=float, default=0.5, help="Min pose confidence")
    p.add_argument("--clip-angles", action="store_true", help="Clip angles to physio ranges")
    p.add_argument("--smooth", choices=["ema","mean","none"], default="ema", help="Smoothing")
    p.add_argument("--ema-alpha", type=float, default=0.2, help="EMA alpha")
    p.add_argument("--mean-window", type=int, default=5, help="Mean window")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def validate_and_reorder(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    """✅ Validate columns and keep only training features."""
    if exercise != "squat":
        raise ValueError(f"Unsupported exercise: {exercise}")
    
    # Expected columns
    cols_needed = BASE_COLS + SQUAT_TRAINING_FEATURES + ["pose_confidence"]
    
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # ✅ Drop YOLO column if present
    df = df.drop(columns=["sq_bar_present"], errors='ignore')
    
    return df[cols_needed].copy()

def clip_physiologic_ranges(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    """✅ Clip features to realistic ranges."""
    df = df.copy()
    
    if exercise == "squat":
        # Angles in degrees [0, 180]
        angle_cols = [
            "sq_knee_angle_L", "sq_knee_angle_R",
            "sq_elbow_angle_L", "sq_elbow_angle_R"
        ]
        for c in angle_cols:
            df[c] = df[c].clip(lower=0, upper=180)
        
        # Torso incline [0, 90]
        df["sq_torso_incline"] = df["sq_torso_incline"].clip(lower=0, upper=90)
        
        # Normalized metrics
        df["sq_pelvis_drop"] = df["sq_pelvis_drop"].clip(lower=0, upper=1.5)
        df["sq_stance_ratio"] = df["sq_stance_ratio"].clip(lower=0.1, upper=3.0)
    
    return df

def smooth_signals(df: pd.DataFrame, exercise: str, method: str, 
                  ema_alpha: float, mean_window: int) -> pd.DataFrame:
    """✅ Apply smoothing to training features only."""
    if method == "none":
        return df
    
    df = df.copy()
    
    if exercise == "squat":
        smooth_cols = [
            "sq_knee_angle_L", "sq_knee_angle_R",
            "sq_torso_incline", "sq_pelvis_drop", "sq_stance_ratio",
            "sq_elbow_angle_L", "sq_elbow_angle_R"
        ]
        
        if method == "ema":
            df = smooth_dataframe(df, smooth_cols, method="ema", alpha=ema_alpha)
        else:
            df = smooth_dataframe(df, smooth_cols, method="mean", window=mean_window)
    
    return df

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    
    if exercise != "squat":
        raise SystemExit(f"❌ Only 'squat' is supported. Got: {exercise}")
    
    raw_csv = get_raw_csv(exercise)
    out_csv = get_cleaned_frames(exercise)

    if not raw_csv.exists():
        raise SystemExit(f"❌ Raw CSV not found: {raw_csv}")

    if args.verbose:
        print("="*60)
        print("🧹 CLEANING DATA")
        print("="*60)
        print(f"Exercise     : {exercise}")
        print(f"Raw CSV      : {raw_csv}")
        print(f"Output       : {out_csv}")
        print(f"Min pose conf: {args.min_pose_conf}")
        print(f"Smoothing    : {args.smooth}")
        print("="*60)

    df = read_csv(raw_csv)
    before = len(df)

    # ✅ Filter by pose confidence
    df = df[df["pose_confidence"] >= args.min_pose_conf].copy()
    
    if args.verbose:
        print(f"✅ Dropped {before - len(df)} low-confidence frames")

    # ✅ Validate and remove unnecessary columns
    df = validate_and_reorder(df, exercise)

    if args.clip_angles:
        df = clip_physiologic_ranges(df, exercise)

    df = df.sort_values(by=["video_name", "t_s", "frame_idx"]).reset_index(drop=True)
    df = smooth_signals(df, exercise, args.smooth, args.ema_alpha, args.mean_window)

    # ✅ Drop pose_confidence after filtering
    df = df.drop(columns=["pose_confidence"], errors='ignore')

    # ✅ Final column order: BASE + 7 features
    final_cols = BASE_COLS + SQUAT_TRAINING_FEATURES
    df = df[final_cols]

    write_csv(df, out_csv, mode="w", header=True)
    
    print(f"\n✅ SUCCESS! Cleaned frames: {out_csv}")
    print(f"📊 Rows: {len(df)}")
    print(f"📐 Columns: {list(df.columns)}")

if __name__ == "__main__":
    main()