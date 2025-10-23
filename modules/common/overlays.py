# modules/common/overlays.py
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def draw_basic_hud(frame: np.ndarray, metrics: Dict) -> np.ndarray:
    """
    يرسم شريط HUD بسيط: FPS, Reps, Angle, State, Quality.
    لا يعتمد على الواجهة، فقط OpenCV.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # خلفية أعلى الإطار
    cv2.rectangle(out, (10, 10), (w - 10, 50), (0, 0, 0), -1)

    line1 = f"FPS {metrics.get('fps', 0):.1f} | Reps {metrics.get('reps', 0)} | Angle {metrics.get('angle', '-')}"
    cv2.putText(out, line1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2, cv2.LINE_AA)

    y = 80
    state = metrics.get("state")
    if state:
        cv2.putText(out, f"State: {state}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)
        y += 40

    q = metrics.get("quality")
    if q is not None:
        try:
            q_txt = f"{float(q):.2f}"
        except Exception:
            q_txt = str(q)
        cv2.putText(out, f"Qual: {q_txt}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

    return out
