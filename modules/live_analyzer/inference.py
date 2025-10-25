#!/usr/bin/env python3
# modules/live_analyzer/inference.py
"""
GRU inference engine with adaptive probability interpretation.
Supports both normal and inverted model outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
from tensorflow import keras

from modules.common.paths import get_models_dir
from modules.common.feature_builder import build_features

class GRUInference:
    def __init__(self, exercise: str, confidence_threshold: float = 0.5, 
                 invert_probabilities: bool = False):
        """
        Args:
            exercise: Exercise type (e.g., 'squat')
            confidence_threshold: Threshold for classification (0-1)
            invert_probabilities: Set True if model outputs are inverted
        """
        self.exercise = exercise
        self.confidence_threshold = confidence_threshold
        self.invert_probabilities = invert_probabilities
        
        models_dir = get_models_dir(exercise)
        model_path = models_dir / f"{exercise}_model.h5"
        meta_path = models_dir / f"{exercise}_meta.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        
        self.model = keras.models.load_model(str(model_path))
        
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
        
        self.feature_list = self.meta["feature_list"]
        self.seq_len = self.meta["seq_len"]
        self.mean = np.array(self.meta["mean"], dtype=np.float32)
        self.std = np.array(self.meta["std"], dtype=np.float32)
        
        # Label mapping from meta
        self.label_map = self.meta.get("label_map", {"Incorrect": 0, "Correct": 1})
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        
        # Buffer for sequence building
        self.buffer: List[List[float]] = []
        
        # Statistics for adaptive threshold
        self.prediction_history = []
        self.max_history = 50
        
        print(f"[GRU] Loaded model: {model_path}")
        print(f"[GRU] Features: {len(self.feature_list)}, seq_len: {self.seq_len}")
        print(f"[GRU] Threshold: {self.confidence_threshold}")
        print(f"[GRU] Invert probabilities: {self.invert_probabilities}")
        print(f"[GRU] Label map: {self.label_map}")
    
    def push_frame_features(self, features: Dict[str, Any]):
        """Add features from one frame to buffer."""
        row = []
        for feat_name in self.feature_list:
            val = features.get(feat_name, np.nan)
            if val is None:
                val = np.nan
            row.append(float(val))
        
        self.buffer.append(row)
        
        # Keep buffer size reasonable
        if len(self.buffer) > self.seq_len * 2:
            self.buffer = self.buffer[-self.seq_len:]
    
    def predict_from_buffer(self) -> Optional[Dict[str, Any]]:
        """
        Build sequence from buffer and predict.
        Uses raw model output without inversion by default.
        """
        if len(self.buffer) < self.seq_len:
            return None
        
        # Take last seq_len frames
        seq = np.array(self.buffer[-self.seq_len:], dtype=np.float32)
        
        # Normalize
        seq_norm = (seq - self.mean[None, :]) / self.std[None, :]
        
        # Add batch dimension [1, seq_len, F]
        X = seq_norm[None, :, :]
        
        # Predict - get raw model output
        raw_prob = float(self.model.predict(X, verbose=0)[0, 0])
        
        # Store in history for analysis
        self.prediction_history.append(raw_prob)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # Apply inversion if needed
        prob = (1.0 - raw_prob) if self.invert_probabilities else raw_prob
        
        # Interpret result
        if prob >= self.confidence_threshold:
            predicted_class = 1  # Correct
            label = "Correct"
        else:
            predicted_class = 0  # Incorrect
            label = "Incorrect"
        
        # Calculate confidence
        confidence = abs(prob - 0.5) * 2.0
        
        return {
            "label": label,
            "probability": prob,
            "confidence": confidence,
            "raw_output": raw_prob,
        }
    
    def get_prediction_stats(self) -> Dict[str, float]:
        """Get statistics from recent predictions."""
        if not self.prediction_history:
            return {"mean": 0.5, "std": 0.0, "min": 0.0, "max": 1.0}
        
        hist = np.array(self.prediction_history)
        return {
            "mean": float(hist.mean()),
            "std": float(hist.std()),
            "min": float(hist.min()),
            "max": float(hist.max()),
        }
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.buffer.clear()
    
    def get_form_feedback(self, features: Dict[str, Any], label: str) -> str:
        """
        Generate Arabic feedback based on features and label.
        """
        if label == "Correct":
            return "ممتاز! استمر بنفس الأداء"
        
        # Check common issues for squat
        feedback_parts = []
        
        # Knee depth
        knee_L = features.get("sq_knee_angle_L", 180)
        knee_R = features.get("sq_knee_angle_R", 180)
        
        # Handle None/NaN
        if knee_L is None or (isinstance(knee_L, float) and np.isnan(knee_L)):
            knee_L = 180
        if knee_R is None or (isinstance(knee_R, float) and np.isnan(knee_R)):
            knee_R = 180
            
        knee_avg = (float(knee_L) + float(knee_R)) / 2.0
        
        # Depth check - most important
        if knee_avg > 120:
            feedback_parts.append("انزل أكثر - العمق غير كافي")
        
        # Back angle
        torso_incline = features.get("sq_torso_incline", 0)
        if torso_incline is None or (isinstance(torso_incline, float) and np.isnan(torso_incline)):
            torso_incline = 0
        torso_incline = float(torso_incline)
        
        if torso_incline > 25:
            feedback_parts.append("الظهر مائل كثير - احتفظ بوضع مستقيم")
        
        # Asymmetry
        pelvis_drop = features.get("sq_pelvis_drop", 0)
        if pelvis_drop is None or (isinstance(pelvis_drop, float) and np.isnan(pelvis_drop)):
            pelvis_drop = 0
        pelvis_drop = float(pelvis_drop)
        
        if pelvis_drop > 0.2:
            feedback_parts.append("عدم توازن - وزع الوزن بالتساوي")
        
        # Stance width
        stance = features.get("sq_stance_ratio", 1.0)
        if stance is None or (isinstance(stance, float) and np.isnan(stance)):
            stance = 1.0
        stance = float(stance)
        
        if stance < 0.7:
            feedback_parts.append("القدمين قريبتين - وسّع الوقفة")
        elif stance > 1.8:
            feedback_parts.append("القدمين بعيدتين - ضيّق الوقفة")
        
        if not feedback_parts:
            # Generic feedback when no specific issue detected
            return "الوضعية تحتاج تحسين - راجع التقنية"
        
        # Return top 2 issues
        return " | ".join(feedback_parts[:2])
    
    def analyze_predictions(self) -> str:
        """
        Analyze recent predictions to suggest if inversion is needed.
        """
        if len(self.prediction_history) < 10:
            return "Not enough data yet"
        
        stats = self.get_prediction_stats()
        mean = stats["mean"]
        
        if mean < 0.3:
            return "⚠️ Model outputs are consistently LOW. Try setting invert_probabilities=True"
        elif mean > 0.7:
            return "⚠️ Model outputs are consistently HIGH. Model might be biased."
        else:
            return f"✓ Model outputs seem balanced (mean={mean:.2f})"


def infer_rep_quality(exercise: str, landmarks_sequence: List[Dict[str, Any]], 
                     detections_sequence: Optional[List[Dict[str, Any]]] = None,
                     confidence_threshold: float = 0.5,
                     invert_probabilities: bool = False) -> Dict[str, Any]:
    """
    Legacy function for batch inference on a complete rep sequence.
    """
    engine = GRUInference(exercise, confidence_threshold, invert_probabilities)
    
    for i, lm in enumerate(landmarks_sequence):
        det = detections_sequence[i] if detections_sequence else None
        features = build_features(lm, det, exercise)
        engine.push_frame_features(features)
    
    result = engine.predict_from_buffer()
    if result is None:
        return {
            "label": "Unknown",
            "probability": 0.0,
            "confidence": 0.0,
            "reason": "Insufficient frames"
        }
    
    # Get last frame features for feedback
    if landmarks_sequence:
        last_lm = landmarks_sequence[-1]
        last_det = detections_sequence[-1] if detections_sequence else None
        last_features = build_features(last_lm, last_det, exercise)
        result["reason"] = engine.get_form_feedback(last_features, result["label"])
    
    return result