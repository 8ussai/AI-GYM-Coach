#!/usr/bin/env python3
"""
Clean raw per-frame CSV into a stable, smoothed frames.csv for a given exercise.

Reads:
  outputs/raw/<exercise>_frames.csv

Writes:
  outputs/cleaned/<exercise>/frames.csv

Typical use:
  python -m modules.data_processing.clean_data --exercise squat --verbose
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from modules.common.paths import get_raw_csv, get_cleaned_frames
from modules.common.io_utils import read_csv, write_csv
from modules.common.smoothing import smooth_dataframe

# Base columns present in raw CSV
BASE_COLS = ["video_name", "frame_idx", "t_s", "rep_id"]

# Squat feature columns (must match feature_builder + extractor order)
SQUAT_FEATURES = [
    "sq_knee_angle_L","sq_knee_angle_R",
    "sq_hip_angle_L","sq_hip_angle_R",
    "sq_ankle_angle_L","sq_ankle_angle_R",
    "sq_torso_incline","sq_pelvis_drop","sq_stance_ratio",
    "sq_elbow_angle_L","sq_elbow_angle_R",
    "sq_bar_present",
    "pose_confidence",
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean and smooth raw frames CSV into cleaned/<exercise>/frames.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g., squat)")
    p.add_argument("--min-pose-conf", type=float, default=0.5, help="Drop frames below this pose confidence")
    p.add_argument("--clip-angles", action="store_true", help="Clip angles to physiologic ranges")
    p.add_argument("--smooth", choices=["ema","mean","none"], default="ema", help="Smoothing method")
    p.add_argument("--ema-alpha", type=float, default=0.2, help="EMA alpha when --smooth=ema")
    p.add_argument("--mean-window", type=int, default=5, help="Window when --smooth=mean")
    p.add_argument("--verbose", action="store_true", help="Print progress info")
    p.add_argument("--overwrite", action="store_true", help="Overwrite cleaned frames.csv instead of replacing")
    return p.parse_args()

def expected_features(exercise: str):
    if exercise == "squat":
        return SQUAT_FEATURES
    raise ValueError(f"Unsupported exercise: {exercise}")

def validate_and_reorder(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    feats = expected_features(exercise)
    cols_needed = BASE_COLS + feats
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw CSV: {missing}")
    # Keep exact order; ignore any extra columns silently
    return df[cols_needed].copy()

def clip_physiologic_ranges(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    df = df.copy()
    if exercise == "squat":
        # Angles in degrees
        angle_cols = [
            "sq_knee_angle_L","sq_knee_angle_R",
            "sq_hip_angle_L","sq_hip_angle_R",
            "sq_ankle_angle_L","sq_ankle_angle_R",
            "sq_elbow_angle_L","sq_elbow_angle_R",
        ]
        for c in angle_cols:
            df[c] = df[c].clip(lower=0, upper=180)

        # Torso incline 0..90
        df["sq_torso_incline"] = df["sq_torso_incline"].clip(lower=0, upper=90)

        # Normalized metrics sanity
        df["sq_pelvis_drop"] = df["sq_pelvis_drop"].clip(lower=0, upper=1.5)
        df["sq_stance_ratio"] = df["sq_stance_ratio"].clip(lower=0.1, upper=3.0)

        # bar_present is binary [0,1]
        df["sq_bar_present"] = df["sq_bar_present"].fillna(0).clip(lower=0, upper=1).astype(int)
    return df

def smooth_signals(df: pd.DataFrame, exercise: str, method: str, ema_alpha: float, mean_window: int) -> pd.DataFrame:
    if method == "none":
        return df
    df = df.copy()
    if exercise == "squat":
        smooth_cols = [
            "sq_knee_angle_L","sq_knee_angle_R",
            "sq_hip_angle_L","sq_hip_angle_R",
            "sq_ankle_angle_L","sq_ankle_angle_R",
            "sq_torso_incline",
            "sq_pelvis_drop","sq_stance_ratio",
            "sq_elbow_angle_L","sq_elbow_angle_R",
        ]
        if method == "ema":
            df = smooth_dataframe(df, smooth_cols, method="ema", alpha=ema_alpha)
        else:
            df = smooth_dataframe(df, smooth_cols, method="mean", window=mean_window)
    return df

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    raw_csv = get_raw_csv(exercise)
    out_csv = get_cleaned_frames(exercise)

    if not raw_csv.exists():
        raise SystemExit(f"[ERROR] Raw CSV not found: {raw_csv}")

    if args.verbose:
        print("======== Cleaning Config ========")
        print(f"exercise        : {exercise}")
        print(f"raw csv         : {raw_csv}")
        print(f"cleaned frames  : {out_csv}")
        print(f"min pose conf   : {args.min_pose_conf}")
        print(f"smoothing       : {args.smooth}")
        print("=================================")

    df = read_csv(raw_csv)

    # Filter by pose confidence
    before = len(df)
    df = df[df["pose_confidence"] >= args.min_pose_conf].copy()
    if args.verbose:
        print(f"[INFO] Dropped {before - len(df)} frame(s) by pose_confidence < {args.min_pose_conf}")

    # Ensure schema and order
    df = validate_and_reorder(df, exercise)

    # Clip ranges
    if args.clip_angles:
        df = clip_physiologic_ranges(df, exercise)

    # Sort to ensure time order per video
    df = df.sort_values(by=["video_name", "t_s", "frame_idx"]).reset_index(drop=True)

    # Smooth signals
    df = smooth_signals(df, exercise, args.smooth, args.ema_alpha, args.mean_window)

    # Rep_id stays as-is (empty) for now; derive_reps will fill
    # Save
    mode = "w"  # always write fresh cleaned file
    write_csv(df, out_csv, mode=mode, header=True)
    print(f"[SUCCESS] Cleaned frames written: {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    main()
