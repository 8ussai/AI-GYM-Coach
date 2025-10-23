# modules/runtime/services/yolo_service.py
from __future__ import annotations
from typing import Any, Dict, Optional

from modules.data_extraction.yolo_runner import YoloRunner


class YoloService:
    """
    غلاف خفيف لـ YoloRunner عندك.
    - يستدعي infer_bgr(frame_bgr)
    - يرجّع dict مثل {"yolo": [ {cls, conf, bbox}, ... ]}
    """
    def __init__(self, weights: str, conf: float = 0.5, iou: Optional[float] = None, device: str = ""):
        # YoloRunner عندك يقبل نفس البراميتر تقريباً
        self._backend = YoloRunner(weights=weights, conf=conf, iou=iou, device=device)
        if not hasattr(self._backend, "infer_bgr"):
            raise AttributeError("YoloRunner must provide infer_bgr(frame_bgr)")

    def detect(self, frame_bgr) -> Dict[str, Any]:
        out = self._backend.infer_bgr(frame_bgr)
        return out or {"yolo": []}
