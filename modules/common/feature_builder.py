from typing import Dict, Any, Optional, Iterable
import numpy as np
from modules.common.config import YOLO_DUMBELL_CLASS_ID, YOLO_BARBELL_CLASS_ID

from .geometry import calculate_angle, distance, angle_from_vertical

def _norm_scale(landmarks: Dict[str, Any]) -> float:
    try:
        return max(distance(landmarks["shoulder_L"], landmarks["shoulder_R"]), 1e-6)
    except Exception:
        return 1.0

def _mean_conf(keys: Iterable[str], lm: Dict[str, Any]) -> float:
    vals = []
    for k in keys:
        conf_key = f"{k}_conf"
        if conf_key in lm and lm[conf_key] is not None:
            vals.append(float(lm[conf_key]))
    return float(np.mean(vals)) if vals else 1.0

def _has_class(detections: Dict[str, Any], class_id: int, min_conf: float = 0.3) -> int:
    if detections is None:
        return 0
    items = detections.get("yolo", [])
    for obj in items:
        try:
            if int(obj.get("cls", -1)) == class_id and float(obj.get("conf", 0)) >= min_conf:
                return 1
        except Exception:
            continue
    return 0

def build_features(landmarks: Dict[str, Any], detections: Optional[Dict[str, Any]], exercise_type: str) -> Dict[str, Any]:
    """
    ✅ Build features for given exercise.
    
    For squat: Returns 7 main features + 2 metadata (bar_present, pose_confidence)
    - The 7 features are used for training
    - bar_present is logged but NOT used in model
    - pose_confidence is used for data cleaning only
    """
    feats: Dict[str, Any] = {}

    if exercise_type == "squat":
        L = landmarks
        scale = _norm_scale(L)

        # ============ 7 TRAINING FEATURES ============
        # 1-2: Knee angles
        feats["sq_knee_angle_L"] = calculate_angle(L["hip_L"], L["knee_L"], L["ankle_L"])
        feats["sq_knee_angle_R"] = calculate_angle(L["hip_R"], L["knee_R"], L["ankle_R"])

        # 3: Torso incline
        shoulder_mid = ((L["shoulder_L"][0]+L["shoulder_R"][0])/2, (L["shoulder_L"][1]+L["shoulder_R"][1])/2)
        hip_mid      = ((L["hip_L"][0]+L["hip_R"][0])/2,         (L["hip_L"][1]+L["hip_R"][1])/2)
        feats["sq_torso_incline"] = angle_from_vertical(shoulder_mid, hip_mid)

        # 4: Pelvis drop (normalized)
        feats["sq_pelvis_drop"] = abs(L["hip_L"][1] - L["hip_R"][1]) / scale

        # 5: Stance ratio (normalized)
        stance = distance(L["ankle_L"], L["ankle_R"])
        feats["sq_stance_ratio"] = stance / scale

        # 6-7: Elbow angles
        feats["sq_elbow_angle_L"] = calculate_angle(L["shoulder_L"], L["elbow_L"], L["wrist_L"])
        feats["sq_elbow_angle_R"] = calculate_angle(L["shoulder_R"], L["elbow_R"], L["wrist_R"])

        # ============ METADATA (NOT FOR TRAINING) ============
        # Bar presence via YOLO (logged but excluded in build_dataset.py)
        feats["sq_bar_present"] = _has_class(detections or {}, class_id=YOLO_BARBELL_CLASS_ID, min_conf=0.3)

        # Pose confidence (used for cleaning in clean_data.py)
        used_keys = [
            "shoulder_L","shoulder_R","hip_L","hip_R",
            "knee_L","knee_R","ankle_L","ankle_R",
            "foot_index_L","foot_index_R","elbow_L","elbow_R","wrist_L","wrist_R"
        ]
        feats["pose_confidence"] = _mean_conf(used_keys, L)

    return feats

# ✅ Define canonical feature list for consistency
SQUAT_TRAINING_FEATURES = [
    "sq_knee_angle_L", "sq_knee_angle_R",
    "sq_torso_incline", "sq_pelvis_drop", "sq_stance_ratio",
    "sq_elbow_angle_L", "sq_elbow_angle_R"
]

SQUAT_METADATA_FEATURES = [
    "sq_bar_present",
    "pose_confidence"
]

SQUAT_ALL_FEATURES = SQUAT_TRAINING_FEATURES + SQUAT_METADATA_FEATURES