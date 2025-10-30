#!/usr/bin/env python3
# modules/live_analyzer/inference.py
"""
GRU inference engine with support for new .keras models.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf

from modules.common.paths import get_models_dir
from modules.common.feature_builder import build_features


class GRUInference:
    def __init__(self, exercise: str, confidence_threshold: float = 0.5):
        """
        Args:
            exercise: Exercise type (e.g., 'squat')
            confidence_threshold: Threshold for classification (0-1)
        """
        self.exercise = exercise
        self.confidence_threshold = confidence_threshold
        
        models_dir = get_models_dir(exercise)
        
        # ✅ دعم كل من .keras و .h5
        model_path_keras = models_dir / f"{exercise}_tcn_gru.keras"
        model_path_h5 = models_dir / f"{exercise}_model.h5"
        
        if model_path_keras.exists():
            model_path = model_path_keras
            print(f"[GRU] Using new .keras model")
        elif model_path_h5.exists():
            model_path = model_path_h5
            print(f"[GRU] Using legacy .h5 model")
        else:
            raise FileNotFoundError(
                f"Model not found. Expected:\n"
                f"  - {model_path_keras}\n"
                f"  - {model_path_h5}"
            )
        
        # ✅ Metadata من dataset بدل model
        meta_path = get_models_dir(exercise).parent.parent / "datasets" / exercise / "squat_metadata.json"
        
        if not meta_path.exists():
            # Fallback: ابحث عن meta قديم
            meta_path = models_dir / f"{exercise}_meta.json"
        
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        # Load model
        self.model = tf.keras.models.load_model(str(model_path))
        
        # Load metadata
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        
        # ✅ استخراج المعلومات من metadata الجديد
        config = self.meta.get("config", {})
        stats = self.meta.get("dataset_stats", {})
        norm = self.meta.get("normalization_params", {})
        
        self.feature_list = config.get("feature_cols", self.meta.get("feature_names", []))
        self.seq_len = stats.get("sequence_length", config.get("sequence_length", 40))
        self.mean = np.array(norm.get("mean", []), dtype=np.float32)
        self.std = np.array(norm.get("std", []), dtype=np.float32)
        
        # ✅ Label mapping
        label_map = self.meta.get("label_mapping", {})
        # تحويل المفاتيح من string إلى int
        self.inv_label_map = {int(k): v for k, v in label_map.items()}
        self.num_classes = len(self.inv_label_map)
        
        # Buffer for sequence building
        self.buffer: List[List[float]] = []
        
        # Statistics
        self.prediction_history = []
        self.max_history = 50
        
        print(f"[GRU] ✅ Model loaded: {model_path.name}")
        print(f"[GRU] Features: {len(self.feature_list)}, seq_len: {self.seq_len}")
        print(f"[GRU] Classes: {self.num_classes} → {self.inv_label_map}")
        print(f"[GRU] Threshold: {self.confidence_threshold:.2%}")
    
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
        Returns dict with label, probability, confidence.
        """
        if len(self.buffer) < self.seq_len:
            return None
        
        # Take last seq_len frames
        seq = np.array(self.buffer[-self.seq_len:], dtype=np.float32)
        
        # Normalize
        seq_norm = (seq - self.mean[None, :]) / (self.std[None, :] + 1e-8)
        
        # Add batch dimension [1, seq_len, F]
        X = seq_norm[None, :, :]
        
        # ✅ Predict - دعم multi-class
        probs = self.model.predict(X, verbose=0)[0]  # shape: (num_classes,)
        
        # ✅ تحديد الكلاس الأعلى
        predicted_class = int(np.argmax(probs))
        max_prob = float(probs[predicted_class])
        
        # ✅ تحويل الـ class ID إلى label
        label = self.inv_label_map.get(predicted_class, "Unknown")
        
        # Store in history
        self.prediction_history.append(max_prob)
        if len(self.prediction_history) > self.max_history:
            self.prediction_history.pop(0)
        
        # ✅ Calculate confidence (distance from uniform distribution)
        uniform_prob = 1.0 / self.num_classes
        confidence = max(0.0, max_prob - uniform_prob) / (1.0 - uniform_prob)
        
        return {
            "label": label,
            "probability": max_prob,
            "confidence": confidence,
            "class_id": predicted_class,
            "all_probs": probs.tolist()
        }
    
    def clear_buffer(self):
        """Clear the frame buffer."""
        self.buffer.clear()
    
    def get_form_feedback(self, features: Dict[str, Any], label: str) -> str:
        """
        Generate Arabic feedback based on features and label.
        """
        # ✅ إذا كان Label = "Correct"
        if label.lower() in ["correct", "صحيح", "جيد"]:
            return "ممتاز! استمر بنفس الأداء"
        
        # ✅ إذا كان Label أحد الأخطاء المعروفة، استخدمه مباشرة
        error_feedback = {
            "low_depth": "انزل أكثر - العمق غير كافي",
            "back_rounding": "الظهر مائل كثير - احتفظ بوضع مستقيم",
            "asymmetry": "عدم توازن - وزع الوزن بالتساوي",
            "stance_width": "وسّع أو ضيّق الوقفة حسب الراحة"
        }
        
        if label in error_feedback:
            return error_feedback[label]
        
        # ✅ إذا label غير معروف، حلل الـ features يدوياً
        feedback_parts = []
        
        # Knee depth
        knee_L = features.get("sq_knee_angle_L", 180)
        knee_R = features.get("sq_knee_angle_R", 180)
        
        if knee_L is None or (isinstance(knee_L, float) and np.isnan(knee_L)):
            knee_L = 180
        if knee_R is None or (isinstance(knee_R, float) and np.isnan(knee_R)):
            knee_R = 180
        
        knee_avg = (float(knee_L) + float(knee_R)) / 2.0
        
        # Depth check
        if knee_avg > 120:
            feedback_parts.append("انزل أكثر - العمق غير كافي")
        
        # Back angle
        torso = features.get("sq_torso_incline", 0)
        if torso is None or (isinstance(torso, float) and np.isnan(torso)):
            torso = 0
        torso = float(torso)
        
        if torso > 25:
            feedback_parts.append("الظهر مائل كثير")
        
        # Asymmetry
        pelvis = features.get("sq_pelvis_drop", 0)
        if pelvis is None or (isinstance(pelvis, float) and np.isnan(pelvis)):
            pelvis = 0
        pelvis = float(pelvis)
        
        if pelvis > 0.2:
            feedback_parts.append("عدم توازن")
        
        # Stance
        stance = features.get("sq_stance_ratio", 1.0)
        if stance is None or (isinstance(stance, float) and np.isnan(stance)):
            stance = 1.0
        stance = float(stance)
        
        if stance < 0.7 or stance > 1.8:
            feedback_parts.append("عدّل عرض الوقفة")
        
        if not feedback_parts:
            return "الوضعية تحتاج تحسين"
        
        return " | ".join(feedback_parts[:2])


def infer_rep_quality(exercise: str, 
                     landmarks_sequence: List[Dict[str, Any]], 
                     detections_sequence: Optional[List[Dict[str, Any]]] = None,
                     confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Legacy function for batch inference on a complete rep sequence.
    """
    engine = GRUInference(exercise, confidence_threshold)
    
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
    
    # Get feedback
    if landmarks_sequence:
        last_lm = landmarks_sequence[-1]
        last_det = detections_sequence[-1] if detections_sequence else None
        last_features = build_features(last_lm, last_det, exercise)
        result["reason"] = engine.get_form_feedback(last_features, result["label"])
    
    return result