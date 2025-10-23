# modules/runtime/services/feature_service.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
from modules.common.feature_builder import build_features

MP_IDX = {
    "nose": 0,
    "eye_inner_L": 1, "eye_L": 2, "eye_outer_L": 3,
    "eye_inner_R": 4, "eye_R": 5, "eye_outer_R": 6,
    "ear_L": 7, "ear_R": 8,
    "mouth_L": 9, "mouth_R": 10,
    "shoulder_L": 11, "shoulder_R": 12,
    "elbow_L": 13, "elbow_R": 14,
    "wrist_L": 15, "wrist_R": 16,
    "pinky_L": 17, "pinky_R": 18,
    "index_L": 19, "index_R": 20,
    "thumb_L": 21, "thumb_R": 22,
    "hip_L": 23, "hip_R": 24,
    "knee_L": 25, "knee_R": 26,
    "ankle_L": 27, "ankle_R": 28,
    "heel_L": 29, "heel_R": 30,
    "foot_index_L": 31, "foot_index_R": 32,
}

NEEDED = [
    "shoulder_L","shoulder_R","hip_L","hip_R",
    "knee_L","knee_R","ankle_L","ankle_R",
    "foot_index_L","foot_index_R","elbow_L","elbow_R","wrist_L","wrist_R"
]

def _mp_to_named_landmarks(kps: np.ndarray) -> Dict[str, Any]:
    kps = np.asarray(kps)
    L: Dict[str, Any] = {k: (np.nan, np.nan) for k in NEEDED}
    C: Dict[str, float] = {k: 0.0 for k in NEEDED}
    if kps.ndim == 2 and kps.shape[0] >= 33 and kps.shape[1] >= 2:
        has_conf = kps.shape[1] >= 3
        for name, idx in MP_IDX.items():
            x, y = float(kps[idx, 0]), float(kps[idx, 1])
            c = float(kps[idx, 2]) if has_conf else 1.0
            L[name] = (x, y); C[name] = c
    L["__conf"] = C
    return L

class FeatureService:
    def squat(self, pose_out: Any, dets: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # لو جاني dict جاهز من PoseRunner، استعمله مباشرة
        if isinstance(pose_out, dict) and all(k in pose_out for k in ["hip_L","hip_R"]):
            L = pose_out
        else:
            L = _mp_to_named_landmarks(pose_out)

        feats = build_features(landmarks=L, detections=dets, exercise_type="squat")
        # نظّف الأرقام
        safe = {}
        for k, v in feats.items():
            try:
                fv = float(v)
                if not np.isfinite(fv):
                    fv = 0.0
                safe[k] = fv
            except Exception:
                pass
        return safe
