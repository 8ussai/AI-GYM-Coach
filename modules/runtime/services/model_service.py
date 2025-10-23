# modules/runtime/services/model_service.py
from __future__ import annotations
from typing import Dict, Any, Deque
from pathlib import Path
from collections import deque
import numpy as np

from modules.model_training.gru_infer import GRUInfer


class GRUService:
    """
    إدارة نافذة الميزات والاستدلال عبر GRUInfer.
    يقرأ seq_len/feat_dim من الميتا أو من شكل الموديل نفسه.
    """
    def __init__(self, model_path: Path, meta_path: Path | None = None):
        self._infer = GRUInfer(model_path=model_path, meta_path=meta_path)
        self.seq_len = self._infer.seq_len
        self.feat_dim = self._infer.feat_dim
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

    def push(self, vec: np.ndarray) -> None:
        if vec.shape[-1] != self.feat_dim:
            raise ValueError(f"feat_dim mismatch: expected {self.feat_dim}, got {vec.shape[-1]}")
        self._buffer.append(vec)
        # مرر إلى GRUInfer أيضاً لو حابب تستخدم بافره الداخلي
        self._infer.push(vec)

    def ready(self) -> bool:
        return len(self._buffer) == self.seq_len

    def predict(self) -> Dict[str, Any]:
        """
        يرجع dict مثل: {"state": "down", "probs": [...], "quality": 0.93}
        إذا مش جاهز، يرجع قيم None.
        """
        return self._infer.infer() if self.ready() else {"state": None, "probs": None, "quality": None}

    def reset(self) -> None:
        self._buffer.clear()
        self._infer.reset()
