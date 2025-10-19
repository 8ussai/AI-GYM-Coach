#!/usr/bin/env python3
# YOLO runner using ultralytics. Returns list of detections with cls/conf/bbox.

from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics is required. Install via: pip install ultralytics") from e

from modules.common.config import YOLO_WEIGHTS, YOLO_CONF_THRESH, YOLO_IOU_THRESH

class YoloRunner:
    def __init__(self, weights: Optional[Path] = None, conf: float = None, iou: float = None, device: str = ""):
        self.weights = Path(weights) if weights else Path(YOLO_WEIGHTS)
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")
        self.conf = YOLO_CONF_THRESH if conf is None else conf
        self.iou = YOLO_IOU_THRESH if iou is None else iou
        self.device = device
        self.model = YOLO(str(self.weights))

    def infer_bgr(self, frame_bgr) -> Dict[str, Any]:
        """Run YOLO on a BGR frame and return dict with 'yolo': [{'cls':int,'conf':float,'bbox':(x,y,w,h)}]."""
        # ultralytics can take numpy arrays directly
        res = self.model.predict(
            source=frame_bgr, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )
        detections: List[Dict[str, Any]] = []
        if not res:
            return {"yolo": detections}

        r0 = res[0]
        if not hasattr(r0, "boxes") or r0.boxes is None:
            return {"yolo": detections}

        boxes = r0.boxes
        # xywh in pixels or normalized? xyxy by default; convert to xywh
        import numpy as np
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        clss  = boxes.cls.cpu().numpy()  if hasattr(boxes.cls, "cpu")  else boxes.cls

        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            detections.append({
                "cls": int(k),
                "conf": float(c),
                "bbox": (float(x1), float(y1), w, h),
            })
        return {"yolo": detections}
