#!/usr/bin/env python3
# modules/live_analyzer/inference.py

import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tensorflow import keras

from modules.live_analyzer.fsm import SquatThresholds, SIG_COL

class LiveInference:
    def __init__(self, models_dir: Path, exercise: str):
        meta_path = models_dir / f"{exercise}_meta.json"
        model_path = models_dir / f"{exercise}_model.h5"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta not found: {meta_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model = keras.models.load_model(model_path)

        self.features: List[str] = self.meta["feature_list"]
        self.seq_len: int = int(self.meta["seq_len"])
        self.mean = np.array(self.meta["mean"], dtype=float)
        self.std  = np.array(self.meta["std"], dtype=float)
        self.std  = np.clip(self.std, 1e-6, None)

    def build_sequence(self, frames_buf: List[dict], start_t: float, end_t: float) -> np.ndarray:
        seg = [r for r in frames_buf if start_t <= r["t_s"] <= end_t]
        if not seg:
            return None
        seg = sorted(seg, key=lambda r: r["t_s"])
        ts = np.array([r["t_s"] for r in seg], dtype=float)
        t_min, t_max = float(ts.min()), float(ts.max())
        if t_max == t_min:
            # repeat same row
            row = np.array([[r[f] for f in self.features] for r in seg], dtype=float)[0]
            X = np.tile(row[None, :], (self.seq_len, 1))
        else:
            # linear interpolation per feature
            t_target = np.linspace(t_min, t_max, num=self.seq_len, dtype=float)
            M = np.array([[r[f] for f in self.features] for r in seg], dtype=float)
            X = np.zeros((self.seq_len, M.shape[1]), dtype=float)
            for j in range(M.shape[1]):
                y = M[:, j]
                # simple fill for NaNs
                y = np.where(np.isnan(y), np.nanmean(y), y)
                X[:, j] = np.interp(t_target, ts, y)
        # normalize
        X = (X - self.mean[None, :]) / self.std[None, :]
        return X  # [T, F]

    def predict_prob(self, X_seq: np.ndarray) -> float:
        X = X_seq[None, :, :]  # [1, T, F]
        prob = float(self.model.predict(X, verbose=0).reshape(-1)[0])
        return prob
