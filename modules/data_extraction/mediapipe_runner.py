import cv2
import mediapipe as mp

from typing import Dict, Tuple, Optional

Point = Tuple[float, float]

_INDEX_TO_NAME = {
    11: "shoulder_L", 12: "shoulder_R",
    23: "hip_L",      24: "hip_R",
    25: "knee_L",     26: "knee_R",
    27: "ankle_L",    28: "ankle_R",
    15: "wrist_L",    16: "wrist_R",
    13: "elbow_L",    14: "elbow_R",
    29: "heel_L",     30: "heel_R",
    31: "foot_index_L", 32: "foot_index_R",
}

class PoseRunner:
    def __init__(self, static_image_mode: bool = False, model_complexity: int = 1,
                 enable_segmentation: bool = False, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        if self._pose:
            self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def process_bgr(self, frame_bgr) -> Optional[Dict[str, float]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._pose.process(frame_rgb)
        if not res or not res.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        out: Dict[str, float] = {}
        lm = res.pose_landmarks.landmark

        for idx, name in _INDEX_TO_NAME.items():
            p = lm[idx]
            out[name] = (float(p.x), float(p.y))  
            out[f"{name}_conf"] = float(p.visibility)

        if "shoulder_L" in out and "shoulder_R" in out:
            out["shoulder_mid"] = (
                (out["shoulder_L"][0] + out["shoulder_R"][0]) / 2.0,
                (out["shoulder_L"][1] + out["shoulder_R"][1]) / 2.0,
            )
        if "hip_L" in out and "hip_R" in out:
            out["hip_mid"] = (
                (out["hip_L"][0] + out["hip_R"][0]) / 2.0,
                (out["hip_L"][1] + out["hip_R"][1]) / 2.0,
            )

        return out