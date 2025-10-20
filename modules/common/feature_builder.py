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
    feats: Dict[str, Any] = {}

    if exercise_type == "squat":
        L = landmarks
        scale = _norm_scale(L)

        # Knees
        feats["sq_knee_angle_L"] = calculate_angle(L["hip_L"], L["knee_L"], L["ankle_L"])
        feats["sq_knee_angle_R"] = calculate_angle(L["hip_R"], L["knee_R"], L["ankle_R"])

        # Torso incline
        shoulder_mid = ((L["shoulder_L"][0]+L["shoulder_R"][0])/2, (L["shoulder_L"][1]+L["shoulder_R"][1])/2)
        hip_mid      = ((L["hip_L"][0]+L["hip_R"][0])/2,         (L["hip_L"][1]+L["hip_R"][1])/2)
        feats["sq_torso_incline"] = angle_from_vertical(shoulder_mid, hip_mid)

        # Pelvis drop (normalized)
        feats["sq_pelvis_drop"] = abs(L["hip_L"][1] - L["hip_R"][1]) / scale

        # Stance ratio (normalized)
        stance = distance(L["ankle_L"], L["ankle_R"])
        feats["sq_stance_ratio"] = stance / scale

        # Elbows
        feats["sq_elbow_angle_L"] = calculate_angle(L["shoulder_L"], L["elbow_L"], L["wrist_L"])
        feats["sq_elbow_angle_R"] = calculate_angle(L["shoulder_R"], L["elbow_R"], L["wrist_R"])

        # Bar presence via YOLO class 1
        feats["sq_bar_present"] = _has_class(detections or {}, class_id=YOLO_BARBELL_CLASS_ID, min_conf=0.3)

        used_keys = [
            "shoulder_L","shoulder_R","hip_L","hip_R",
            "knee_L","knee_R","ankle_L","ankle_R",
            "foot_index_L","foot_index_R","elbow_L","elbow_R","wrist_L","wrist_R"
        ]
        feats["pose_confidence"] = _mean_conf(used_keys, L)

    return feats
