# modules/runtime/services/pose_service.py
from __future__ import annotations
from typing import Optional, Dict, Any

from modules.data_extraction.mediapipe_runner import PoseRunner


class PoseService:
    """
    غلاف مباشر لـ PoseRunner عندك.
    - يأخذ إطار BGR من OpenCV
    - يرجع dict لاندماركات أو None
    """
    def __init__(self, **runner_kwargs):
        self._runner = PoseRunner(**runner_kwargs)
        if not hasattr(self._runner, "process_bgr"):
            raise AttributeError("PoseRunner must provide process_bgr(frame_bgr) returning landmarks dict or None.")

    def keypoints(self, frame_bgr) -> Optional[Dict[str, Any]]:
        # mediapipe_runner.PoseRunner.process_bgr يتوقع BGR ويرجع dict أو None
        return self._runner.process_bgr(frame_bgr)

    def close(self):
        if hasattr(self._runner, "close"):
            self._runner.close()
