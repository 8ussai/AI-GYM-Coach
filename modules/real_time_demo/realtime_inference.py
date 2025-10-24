"""
Real-time GRU model inference for rep quality classification.

Loads the trained model and meta.json, then performs inference on completed reps.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tensorflow import keras

from modules.common.paths import get_models_dir


class RealtimeInference:
    def __init__(self, exercise: str = "squat"):
        self.exercise = exercise
        self.model = None
        self.meta = None
        self.feature_list = []
        self.mean = None
        self.std = None
        self.seq_len = 96
        self.label_map = {"Incorrect": 0, "Correct": 1}
        self.label_map_inv = {0: "Incorrect", 1: "Correct"}
        
        self._load_model()

    def _load_model(self):
        models_dir = get_models_dir(self.exercise)
        model_path = models_dir / f"{self.exercise}_model.h5"
        meta_path = models_dir / f"{self.exercise}_meta.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta not found: {meta_path}")
        
        self.model = keras.models.load_model(str(model_path))
        
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        
        self.feature_list = self.meta["feature_list"]
        self.mean = np.array(self.meta["mean"])
        self.std = np.array(self.meta["std"])
        self.seq_len = int(self.meta["seq_len"])
        self.label_map = self.meta.get("label_map", {"Incorrect": 0, "Correct": 1})
        self.label_map_inv = {v: k for k, v in self.label_map.items()}

    def _resample_rep(self, features_sequence: List[Dict]) -> np.ndarray:
        if len(features_sequence) < 2:
            row = np.array([features_sequence[0].get(f, 0.0) for f in self.feature_list], dtype=float)
            row = np.nan_to_num(row, nan=0.0)
            return np.tile(row, (self.seq_len, 1))
        
        ts = np.arange(len(features_sequence), dtype=float)
        t_target = np.linspace(0, len(features_sequence) - 1, num=self.seq_len, dtype=float)
        
        M = np.zeros((len(features_sequence), len(self.feature_list)), dtype=float)
        for i, feat_dict in enumerate(features_sequence):
            for j, feat_name in enumerate(self.feature_list):
                val = feat_dict.get(feat_name, None)
                M[i, j] = 0.0 if (val is None or np.isnan(val)) else float(val)
        
        out = np.zeros((self.seq_len, len(self.feature_list)), dtype=float)
        for f in range(M.shape[1]):
            y = M[:, f]
            y_series = pd.Series(y)
            y_filled = y_series.ffill().bfill()
            if y_filled.isna().all():
                y_filled = y_series.fillna(0.0)
            y_clean = y_filled.to_numpy()
            out[:, f] = np.interp(t_target, ts, y_clean)
        
        out = np.nan_to_num(out, nan=0.0)
        return out

    def classify_rep(self, features_sequence: List[Dict]) -> Dict:
        if not features_sequence:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "probability": 0.5
            }
        
        seq = self._resample_rep(features_sequence)
        
        seq_norm = (seq - self.mean[None, :]) / self.std[None, :]
        
        X = seq_norm[None, :, :]
        
        prob = float(self.model.predict(X, verbose=0)[0, 0])
        
        pred_label = self.label_map_inv.get(1 if prob >= 0.5 else 0, "Unknown")
        
        confidence = prob if prob >= 0.5 else (1 - prob)
        
        return {
            "prediction": pred_label,
            "confidence": confidence,
            "probability": prob
        }
