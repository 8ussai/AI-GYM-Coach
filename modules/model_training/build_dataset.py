#!/usr/bin/env python3
# modules/model_training/build_dataset.py
"""
Build per-rep sequences dataset for a given exercise.

Reads:
  - outputs/cleaned/<exercise>/frames.csv
  - outputs/cleaned/<exercise>_reps.csv

Writes:
  - outputs/datasets/<exercise>/{train,val,test}/X.npy, y.npy, rep_ids.json
  - outputs/models/<exercise>_meta.json  (feature_list, mean, std, seq_len, label_map)

Usage:
  python -m modules.model_training.build_dataset --exercise squat --seq-len 96 --val 0.15 --test 0.15 --oversample-correct
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from modules.common.paths import (
    get_cleaned_frames, get_cleaned_reps,
    get_dataset_dir, get_models_dir
)
from modules.common.io_utils import read_csv, write_csv

# Features aligned with your latest schema (after removing hip/ankle angles)
FEATURE_LIST = [
    "sq_knee_angle_L", "sq_knee_angle_R",
    "sq_torso_incline", "sq_pelvis_drop", "sq_stance_ratio",
    "sq_elbow_angle_L", "sq_elbow_angle_R",
    "sq_bar_present",
    "pose_confidence",
]

LABEL_MAP = {"Incorrect": 0, "Correct": 1}

def parse_args():
    p = argparse.ArgumentParser(
        description="Build fixed-length per-rep sequences dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--exercise", type=str, required=True, help="Exercise key (e.g., squat)")
    p.add_argument("--seq-len", type=int, default=96, help="Output sequence length per rep")
    p.add_argument("--val", type=float, default=0.15, help="Validation fraction (by video)")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction (by video)")
    p.add_argument("--oversample-correct", action="store_true",
                   help="Oversample 'Correct' class in TRAIN split to mitigate imbalance")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def _resample_rep(df_rep: pd.DataFrame, seq_len: int) -> np.ndarray:
    """
    Resample a single rep's time series to fixed length using linear interpolation on t_s.
    Returns array [seq_len, F]
    """
    ts = df_rep["t_s"].to_numpy()
    if len(ts) < 2:
        # pad by repeating the same row
        row = df_rep[FEATURE_LIST].iloc[0].to_numpy(dtype=float)
        return np.tile(row, (seq_len, 1))

    t_min, t_max = float(ts.min()), float(ts.max())
    t_target = np.linspace(t_min, t_max, num=seq_len, dtype=float)

    # Build matrix [len(ts), F]
    M = df_rep[FEATURE_LIST].to_numpy(dtype=float)

    # Interpolate per feature independently
    out = np.zeros((seq_len, M.shape[1]), dtype=float)
    for f in range(M.shape[1]):
        y = M[:, f]
        # handle NaNs by simple forward-fill then back-fill
        y = pd.Series(y).ffill().bfill().to_numpy()
        out[:, f] = np.interp(t_target, ts, y)
    return out

def _group_reps(frames: pd.DataFrame, reps: pd.DataFrame, seq_len: int, verbose: bool):
    """
    For each video and rep_id, extract and resample features to fixed length.
    Returns: X [N, seq_len, F], y [N], rep_meta list of dicts
    """
    # Merge labels on (video_name, rep_id)
    reps_keyed = reps.set_index(["video_name", "rep_id"])
    X_list, y_list, meta_list = [], [], []

    # iterate per (video, rep_id)
    for (vid, rid), dfv in frames.groupby(["video_name", "rep_id"], sort=False):
        if not isinstance(rid, str) or not rid:
            continue
        if (vid, rid) not in reps_keyed.index:
            # Guard: if derive_reps and cleaned frames got out of sync
            continue
        label = reps_keyed.loc[(vid, rid), "label"]
        if isinstance(label, pd.Series):
            label = label.iloc[0]
        y_val = LABEL_MAP.get(str(label), None)
        if y_val is None:
            continue

        # keep only this rep frames in time order
        df_rep = dfv.sort_values(by=["t_s", "frame_idx"])
        # resample
        seq = _resample_rep(df_rep, seq_len)
        X_list.append(seq)
        y_list.append(y_val)
        meta_list.append({
            "video_name": vid,
            "rep_id": rid,
            "start_time_s": float(df_rep["t_s"].min()),
            "end_time_s": float(df_rep["t_s"].max()),
            "label": str(label),
        })

    if verbose:
        print(f"[INFO] Built {len(X_list)} rep sequences")

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, seq_len, len(FEATURE_LIST)), dtype=float)
    y = np.array(y_list, dtype=int)
    return X, y, meta_list

def _split_by_video(reps_df: pd.DataFrame, val_frac: float, test_frac: float, verbose: bool):
    vids = reps_df["video_name"].unique().tolist()
    vids.sort()

    n = len(vids)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    n_train = max(0, n - n_test - n_val)
    if n_train <= 0:
        raise ValueError("Invalid split fractions; train set would be empty.")

    # simple deterministic split
    test_vids = vids[:n_test]
    val_vids  = vids[n_test:n_test+n_val]
    train_vids= vids[n_test+n_val:]

    if verbose:
        print(f"[INFO] Split by video: train={len(train_vids)} val={len(val_vids)} test={len(test_vids)}")

    return set(train_vids), set(val_vids), set(test_vids)

def _normalize_train_stats(X_train: np.ndarray):
    """
    Compute mean/std over all time steps and samples per feature [F]
    Returns mean [F], std [F] (std clipped to >=1e-6)
    """
    # reshape to [N*T, F]
    NT, F = X_train.shape[0]*X_train.shape[1], X_train.shape[2]
    flat = X_train.reshape(NT, F)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0)
    std  = np.clip(std, 1e-6, None)
    return mean, std

def _apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean[None, None, :]) / std[None, None, :]

def _oversample_correct(X: np.ndarray, y: np.ndarray, meta: List[Dict]):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, meta  # nothing to do

    # replicate positives to match negatives roughly
    reps_needed = int(np.ceil(len(neg_idx) / max(1, len(pos_idx)))) - 1
    if reps_needed <= 0:
        return X, y, meta

    X_aug = [X]
    y_aug = [y]
    meta_aug = [meta]
    for _ in range(reps_needed):
        X_aug.append(X[pos_idx])
        y_aug.append(y[pos_idx])
        meta_aug.append([meta[i] for i in pos_idx])

    X_new = np.concatenate(X_aug, axis=0)
    y_new = np.concatenate(y_aug, axis=0)
    meta_new = sum(meta_aug, [])
    return X_new, y_new, meta_new

def main():
    args = parse_args()
    exercise = args.exercise.strip().lower()

    frames_csv = get_cleaned_frames(exercise)
    reps_csv = get_cleaned_reps(exercise)
    if not frames_csv.exists() or not reps_csv.exists():
        raise SystemExit(f"[ERROR] Missing cleaned data. frames={frames_csv} reps={reps_csv}")

    print("======== Build Dataset ========")
    print(f"exercise     : {exercise}")
    print(f"frames.csv   : {frames_csv}")
    print(f"reps.csv     : {reps_csv}")
    print(f"seq_len      : {args.seq_len}")
    print(f"val/test frac: {args.val}/{args.test}")
    print(f"oversample   : {args.oversample_correct}")
    print("================================")

    frames = read_csv(frames_csv)
    reps = read_csv(reps_csv)

    # Keep only frames that belong to a rep
    frames = frames[frames["rep_id"].astype(str) != ""].copy()
    frames = frames.sort_values(by=["video_name", "rep_id", "t_s", "frame_idx"]).reset_index(drop=True)

    # Split by video
    train_vids, val_vids, test_vids = _split_by_video(reps, args.val, args.test, args.verbose)

    # Build datasets
    def filt(df, vids): return df[df["video_name"].isin(vids)].copy()

    X_tr, y_tr, meta_tr = _group_reps(filt(frames, train_vids), filt(reps, train_vids), args.seq_len, args.verbose)
    X_va, y_va, meta_va = _group_reps(filt(frames, val_vids),   filt(reps, val_vids),   args.seq_len, args.verbose)
    X_te, y_te, meta_te = _group_reps(filt(frames, test_vids),  filt(reps, test_vids),  args.seq_len, args.verbose)

    # Oversample Correct on train if requested
    if args.oversample_correct:
        X_tr, y_tr, meta_tr = _oversample_correct(X_tr, y_tr, meta_tr)

    # Compute normalization from train only
    if X_tr.shape[0] == 0:
        raise SystemExit("[ERROR] Empty train set after grouping reps.")
    mean, std = _normalize_train_stats(X_tr)

    X_tr = _apply_norm(X_tr, mean, std)
    X_va = _apply_norm(X_va, mean, std) if X_va.size else X_va
    X_te = _apply_norm(X_te, mean, std) if X_te.size else X_te

    # Save datasets
    ds_dir = get_dataset_dir(exercise)
    (ds_dir / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "val").mkdir(parents=True, exist_ok=True)
    (ds_dir / "test").mkdir(parents=True, exist_ok=True)

    np.save(ds_dir / "train" / "X.npy", X_tr)
    np.save(ds_dir / "train" / "y.npy", y_tr)
    (ds_dir / "train" / "rep_ids.json").write_text(json.dumps(meta_tr, indent=2))

    np.save(ds_dir / "val" / "X.npy", X_va)
    np.save(ds_dir / "val" / "y.npy", y_va)
    (ds_dir / "val" / "rep_ids.json").write_text(json.dumps(meta_va, indent=2))

    np.save(ds_dir / "test" / "X.npy", X_te)
    np.save(ds_dir / "test" / "y.npy", y_te)
    (ds_dir / "test" / "rep_ids.json").write_text(json.dumps(meta_te, indent=2))

    # Save meta.json
    models_dir = get_models_dir(exercise)
    models_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "exercise": exercise,
        "feature_list": FEATURE_LIST,
        "label_map": LABEL_MAP,
        "seq_len": int(args.seq_len),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (models_dir / f"{exercise}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[SUCCESS] Dataset saved under: {ds_dir}")
    print(f"[SUCCESS] Meta saved: {models_dir / f'{exercise}_meta.json'}")
    print(f"[INFO] Shapes: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")

if __name__ == "__main__":
    main()
