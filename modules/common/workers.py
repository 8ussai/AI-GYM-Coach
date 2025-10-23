# modules/common/workers.py
from __future__ import annotations

import time
from typing import Optional, Dict, Any, List, Tuple, Union

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal


class BaseModelWorker(QThread):
    """
    Worker أساسي لالتقاط الفيديو وتشغيل الاستدلال داخل QThread بدون تعطيل الـ UI.

    إشارات:
      - frame_ready(np.ndarray): إطار جاهز للعرض (بعد الرسم إن وُجد).
      - metrics_ready(dict): قاموس قياسات مثل {"fps":..,"reps":..,"angle":..,"state":..}.
      - event(str, dict): أحداث مهمة مثل ("RepStart", {...}) أو ("RepEnd", {...}).
      - error(str): رسالة خطأ قابلة للعرض.

    لتخصيص السلوك:
      - override load_models() لتحميل النماذج والخدمات.
      - override preprocess(frame) لأي تجهيز قبل infer_one.
      - override infer_one(frame) ليُرجع {"overlay","metrics","events"}.
    """
    frame_ready = Signal(np.ndarray)
    metrics_ready = Signal(dict)
    event = Signal(str, dict)
    error = Signal(str)

    def __init__(self, source: Union[int, str], settings_obj):
        super().__init__()
        self.source = source
        self.settings = settings_obj
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    # ---------- نقاط التوسعة ----------
    def load_models(self) -> None:
        """حمّل النماذج/الخدمات هنا."""
        pass

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """تجهيز بسيط للإطار قبل الاستدلال."""
        return frame

    def infer_one(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        نفّذ الاستدلال لفريم واحد.

        يُتوقع إرجاع:
          {"overlay": np.ndarray | None,
           "metrics": dict,
           "events": List[Tuple[str, dict]]}
        """
        return {"overlay": frame, "metrics": {}, "events": []}

    def on_end_of_stream(self) -> None:
        """تُستدعى عند انتهاء الفيديو/الكاميرا."""
        pass

    # ---------- إدارة المصدر ----------
    def _open_source(self) -> bool:
        if isinstance(self.source, int):
            # CAP_DSHOW يتجنّب تأخير فتح الكاميرا على ويندوز
            try:
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            except Exception:
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap or not self._cap.isOpened():
            self.error.emit(f"Failed to open source: {self.source}")
            return False
        return True

    def _close_source(self) -> None:
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def stop(self) -> None:
        self._running = False

    # ---------- الحلقة الرئيسية ----------
    def run(self) -> None:
        self._running = True
        if not self._open_source():
            return

        # تحميل النماذج
        try:
            self.load_models()
        except Exception as e:
            self.error.emit(f"Model load failed: {e}")
            self._close_source()
            return

        t0 = time.time()
        frames = 0
        target_fps = max(1, int(getattr(self.settings, "target_fps", 30)))
        interval = 1.0 / float(target_fps)

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.on_end_of_stream()
                self.error.emit("Stream ended")
                break

            if getattr(self.settings, "flip_camera", False):
                frame = cv2.flip(frame, 1)

            try:
                frame_p = self.preprocess(frame)
                out = self.infer_one(frame_p)
            except Exception as e:
                self.error.emit(f"Infer error: {e}")
                break

            overlay = out.get("overlay")
            self.frame_ready.emit(overlay if overlay is not None else frame)

            frames += 1
            elapsed = max(1e-6, time.time() - t0)
            fps = frames / elapsed

            metrics = {"fps": fps}
            metrics.update(out.get("metrics", {}))
            self.metrics_ready.emit(metrics)

            for name, payload in out.get("events", []):
                self.event.emit(name, payload)

            time.sleep(max(0.0, interval))

        self._close_source()
