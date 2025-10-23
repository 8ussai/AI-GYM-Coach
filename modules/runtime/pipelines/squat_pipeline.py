# modules/runtime/pipelines/squat_pipeline.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from collections import deque

import numpy as np

from .base_pipeline import BasePipeline
from modules.common.workers import BaseModelWorker
from modules.common.overlays import draw_basic_hud
from modules.common.fsm import SquatFSM, SquatThresholds

# الخدمات (متوافقة مع كودك الحالي)
from modules.runtime.services.pose_service import PoseService   # uses PoseRunner.process_bgr
from modules.runtime.services.yolo_service import YoloService   # uses YoloRunner.infer_bgr
from modules.runtime.services.feature_service import FeatureService
from modules.runtime.services.model_service import GRUService


class _SquatWorker(BaseModelWorker):
    """
    Worker لتمرين السكوات:
      Pose(dict) -> YOLO(dict) -> build_features(...) -> FSM -> GRU -> HUD
    """
    def load_models(self) -> None:
        # خدمات الإدخال والنماذج
        self.pose = PoseService()

        # اضبط مسار الوزن تبعك حسب مكانه الفعلي
        yolo_weights = Path("yolo_training") / "runs" / "train" / "train" / "weights" / "best.pt"
        self.yolo = YoloService(
            weights=str(yolo_weights),
            conf=getattr(self.settings, "confidence", 0.5),
        )
        self.feat = FeatureService()

        # FSM
        self.fsm = SquatFSM(SquatThresholds())
        self._reps = 0

        # GRU
        model_dir = Path("outputs") / "models" / "squat"
        self.gru = GRUService(
            model_path=model_dir / "squat_model.h5",
            meta_path=model_dir / "squat_meta.json",  # فيها feature_list = 9 عناصر
        )
        self.seq_len = self.gru.seq_len
        self.feat_dim = self.gru.feat_dim
        self.buffer: deque[np.ndarray] = deque(maxlen=self.seq_len)

    def _vectorize(self, f: Dict[str, float]) -> np.ndarray:
        """
        يبني متجه الميزات بنفس ترتيب التدريب المحدد في squat_meta.json.
        متوقَّع feat_dim = 9 بالضبط.
        """
        names = [
            "sq_knee_angle_L",
            "sq_knee_angle_R",
            "sq_torso_incline",
            "sq_pelvis_drop",
            "sq_stance_ratio",
            "sq_elbow_angle_L",
            "sq_elbow_angle_R",
            "sq_bar_present",
            "pose_confidence",
        ]
        vec = np.array([float(f.get(k, 0.0)) for k in names], dtype="float32")

        # تنظيف أي قيم شاطحة
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

        # تحقق من الطول المتوقع (9)
        if vec.shape[0] != 9:
            raise ValueError(f"feat_dim mismatch: expected 9, got {vec.shape[0]}")
        return vec

    def infer_one(self, frame: np.ndarray) -> Dict[str, Any]:
        # 1) Pose + YOLO
        landmarks = self.pose.keypoints(frame)    # dict أو None
        dets = self.yolo.detect(frame)            # dict مثل {"yolo":[...]}

        # ما في لاندماركات؟ رجّع HUD فقط
        if not landmarks:
            metrics = {"reps": self._reps, "angle": None, "state": None, "quality": None}
            overlay = draw_basic_hud(frame, metrics | {"fps": 0})
            return {"overlay": overlay, "metrics": metrics, "events": []}

        # 2) ميزات باستخدام دالتك build_features(...)
        feats = self.feat.squat(landmarks, dets)
        if not feats:
            metrics = {"reps": self._reps, "angle": None, "state": None, "quality": None}
            overlay = draw_basic_hud(frame, metrics | {"fps": 0})
            return {"overlay": overlay, "metrics": metrics, "events": []}

        knee_mean = 0.5 * (feats.get("sq_knee_angle_L", 0.0) + feats.get("sq_knee_angle_R", 0.0))

        # 3) FSM عدّ التكرارات
        ev = self.fsm.update(knee_mean)
        events: List[Tuple[str, dict]] = []
        if ev:
            events.append(ev)
            if ev[0] == "RepEnd":
                self._reps = self.fsm.rep_count

        # 4) GRU على نافذة الميزات
        vec = self._vectorize(feats)
        self.gru.push(vec)
        gru_out = self.gru.predict() if self.gru.ready() else {"state": None, "quality": None}
        state = gru_out.get("state")
        quality = gru_out.get("quality")

        # 5) مخرجات
        metrics = {
            "reps": self._reps,
            "angle": knee_mean,
            "state": state,
            "quality": quality,
        }
        overlay = draw_basic_hud(frame, metrics | {"fps": 0})
        return {"overlay": overlay, "metrics": metrics, "events": events}


class SquatPipeline(BasePipeline):
    """
    Pipeline السكوات: يبني الـ Worker ويعيده للواجهة.
    """
    def build_worker(self) -> BaseModelWorker:
        return _SquatWorker(self.source, self.settings)
