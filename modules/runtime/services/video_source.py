# modules/runtime/services/video_source.py
from __future__ import annotations
from typing import Optional, Tuple, Union
import sys
import cv2
import numpy as np


class VideoSource:
    """
    غلاف بسيط لـ cv2.VideoCapture عشان تقدر تبدّله أو تعمل له Mock بالاختبارات.
    مش ضروري إذا الـ Worker بيفتح الكاميرا بنفسه، بس مفيد للمرونة.
    """
    def __init__(self, source: Union[int, str]):
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        if isinstance(self.source, int):
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            elif sys.platform == "darwin":
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = cv2.VideoCapture(self.source)
        return bool(self.cap and self.cap.isOpened())

    def read(self) -> Tuple[bool, np.ndarray | None]:
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            try:
                self.cap.release()
            finally:
                self.cap = None
