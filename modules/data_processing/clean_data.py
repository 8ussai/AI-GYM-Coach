from modules.common.paths import get_raw_csv, get_cleaned_frames
from modules.common.io_utils import read_csv, write_csv
from modules.common.smoothing import smooth_dataframe
from modules.common.feature_builder import SQUAT_TRAINING_FEATURES

import numpy as np
import pandas as pd
import argparse

from pathlib import Path

BASE_COLS = ["video_name", "frame_idx", "t_s", "rep_id"]

def parse_args():
    p = argparse.ArgumentParser(
        description="Clean and smooth raw frames CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key")
    return p.parse_args()

def validate_and_reorder(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    if exercise != "squat":
        raise ValueError(f"Unsupported exercise: {exercise}")
    
    cols_needed = BASE_COLS + SQUAT_TRAINING_FEATURES + ["pose_confidence"]
    
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df.drop(columns=["sq_bar_present"], errors='ignore')
    
    return df[cols_needed].copy()

def clip_physiologic_ranges(df: pd.DataFrame, exercise: str) -> pd.DataFrame:
    df = df.copy()
    
    if exercise == "squat":
        angle_cols = [
            "sq_knee_angle_L", "sq_knee_angle_R",
            "sq_elbow_angle_L", "sq_elbow_angle_R"
        ]
        for c in angle_cols:
            df[c] = df[c].clip(lower=0, upper=180)
        
        df["sq_torso_incline"] = df["sq_torso_incline"].clip(lower=0, upper=90)
        df["sq_pelvis_drop"] = df["sq_pelvis_drop"].clip(lower=0, upper=1.5)
        df["sq_stance_ratio"] = df["sq_stance_ratio"].clip(lower=0.1, upper=3.0)
    
    return df

def smooth_signals(df: pd.DataFrame, exercise: str, method: str, 
                  ema_alpha: float, mean_window: int) -> pd.DataFrame:
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
    df = read_csv(raw_csv)
    before = len(df)
    df = df[df["pose_confidence"] >= args.min_pose_conf].copy()
    df = validate_and_reorder(df, exercise)

    if args.clip_angles:
        df = clip_physiologic_ranges(df, exercise)

    df = df.sort_values(by=["video_name", "t_s", "frame_idx"]).reset_index(drop=True)
    df = smooth_signals(df, exercise, args.smooth, args.ema_alpha, args.mean_window)
    df = df.drop(columns=["pose_confidence"], errors='ignore')
    final_cols = BASE_COLS + SQUAT_TRAINING_FEATURES
    df = df[final_cols]
    write_csv(df, out_csv, mode="w", header=True)
    
if __name__ == "__main__":
    main()