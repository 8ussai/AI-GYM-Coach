#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd

from modules.common.paths import get_cleaned_frames, get_cleaned_reps
from modules.common.io_utils import read_csv, write_csv

SIG_COL = "sq_knee_angle_mean"

@dataclass
class SquatThresholds:
    rest_knee_deg: float = 150.0
    start_knee_deg: float = 135.0
    bottom_knee_deg: float = 100.0
    max_torso_incline: float = 70.0
    max_pelvis_drop: float = 0.25
    min_rep_time_s: float = 0.4
    max_rep_time_s: float = 6.0
    cooldown_s: float = 0.15

def parse_args():
    p = argparse.ArgumentParser(
        description="Derive reps from cleaned frames and write reps.csv; fill rep_id back into frames.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g., squat)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--rest-knee", type=float, default=SquatThresholds.rest_knee_deg)
    p.add_argument("--start-knee", type=float, default=SquatThresholds.start_knee_deg)
    p.add_argument("--bottom-knee", type=float, default=SquatThresholds.bottom_knee_deg)
    p.add_argument("--min-time", type=float, default=SquatThresholds.min_rep_time_s)
    p.add_argument("--max-time", type=float, default=SquatThresholds.max_rep_time_s)
    p.add_argument("--cooldown", type=float, default=SquatThresholds.cooldown_s)
    return p.parse_args()

def load_frames(exercise: str):
    frames_csv = get_cleaned_frames(exercise)
    if not frames_csv.exists():
        raise SystemExit(f"[ERROR] Cleaned frames.csv not found: {frames_csv}")
    df = read_csv(frames_csv)
    return df, frames_csv

def prepare_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[SIG_COL] = df[["sq_knee_angle_L","sq_knee_angle_R"]].mean(axis=1)
    return df

def derive_reps_for_video(dfv: pd.DataFrame, th: SquatThresholds, verbose: bool = False):
    t = dfv["t_s"].to_numpy()
    sig = dfv[SIG_COL].to_numpy()
    torso = dfv["sq_torso_incline"].to_numpy()
    pelvis = dfv["sq_pelvis_drop"].to_numpy()

    reps = []
    rep_ids = np.array([""] * len(dfv), dtype=object)

    state = "IDLE"
    start_i = None
    min_i = None
    last_end_time = -1e9
    rep_counter = 0

    for i in range(len(dfv)):
        ang = sig[i]
        if np.isnan(ang):
            continue
        time = t[i]

        if state == "IDLE":
            if ang <= th.start_knee_deg:
                state = "INREP"; start_i = i; min_i = i
        elif state == "INREP":
            if min_i is None or ang < sig[min_i]:
                min_i = i
            if ang >= th.rest_knee_deg:
                end_i = i
                dur = t[end_i] - t[start_i]
                bottom_angle = sig[min_i] if min_i is not None else np.nan
                if th.min_rep_time_s <= dur <= th.max_rep_time_s and not np.isnan(bottom_angle):
                    rep_counter += 1
                    rid = f"r{rep_counter}"
                    label, reason = label_squat_rep(
                        bottom_angle=bottom_angle,
                        torso_incline=np.nanmax(torso[start_i:end_i+1]),
                        pelvis_drop=np.nanmax(pelvis[start_i:end_i+1]),
                        th=th
                    )
                    reps.append({
                        "rep_id": rid,
                        "start_time_s": float(t[start_i]),
                        "end_time_s": float(t[end_i]),
                        "label": label,
                        "reason": reason
                    })
                    rep_ids[start_i:end_i+1] = rid
                state = "COOLDOWN"; start_i = None; min_i = None
        elif state == "COOLDOWN":
            if time - last_end_time >= th.cooldown_s:
                state = "IDLE"

    return reps, rep_ids

def label_squat_rep(bottom_angle: float, torso_incline: float, pelvis_drop: float, th: SquatThresholds):
    reasons = []
    if not (bottom_angle <= th.bottom_knee_deg):
        reasons.append("low_depth")
    if torso_incline > th.max_torso_incline:
        reasons.append("back_rounding")
    if pelvis_drop > th.max_pelvis_drop:
        reasons.append("asymmetry")
    if len(reasons) == 0:
        return "Correct", ""
    for r in ["low_depth","back_rounding","asymmetry"]:
        if r in reasons:
            return "Incorrect", r
    return "Incorrect", reasons[0]

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()
    if exercise != "squat":
        raise SystemExit(f"[ERROR] Only 'squat' is supported for derive_reps right now.")

    th = SquatThresholds(
        rest_knee_deg=args.rest_knee,
        start_knee_deg=args.start_knee,
        bottom_knee_deg=args.bottom_knee,
        min_rep_time_s=args.min_time,
        max_rep_time_s=args.max_time,
        cooldown_s=args.cooldown
    )

    frames_df, frames_path = load_frames(exercise)
    if args.verbose:
        print("======== Derive Reps Config ========")
        print(f"exercise       : {exercise}")
        print(f"frames.csv     : {frames_path}")
        print(f"rest/start/btm : {th.rest_knee_deg}/{th.start_knee_deg}/{th.bottom_knee_deg}")
        print(f"min/max time   : {th.min_rep_time_s}/{th.max_rep_time_s}")
        print("====================================")

    frames_df = frames_df.sort_values(by=["video_name","t_s","frame_idx"]).reset_index(drop=True)
    frames_df = prepare_signal(frames_df)

    all_reps = []
    rep_id_column = np.array([""] * len(frames_df), dtype=object)

    for vid, dfv in frames_df.groupby("video_name", sort=False):
        reps, rep_ids = derive_reps_for_video(dfv, th, verbose=args.verbose)
        for r in reps:
            all_reps.append({"video_name": vid, **r})
        rep_id_column[dfv.index.to_numpy()] = rep_ids
        if args.verbose:
            print(f"[INFO] {vid}: reps found = {len(reps)}")

    reps_df = pd.DataFrame(all_reps, columns=["video_name","rep_id","start_time_s","end_time_s","label","reason"])
    reps_out = get_cleaned_reps(exercise)
    write_csv(reps_df, reps_out, mode="w", header=True)
    print(f"[SUCCESS] Reps written: {reps_out} (rows={len(reps_df)})")

    frames_df["rep_id"] = rep_id_column
    write_csv(frames_df, frames_path, mode="w", header=True)
    print(f"[SUCCESS] Frames updated with rep_id: {frames_path} (rows={len(frames_df)})")

if __name__ == "__main__":
    main()
