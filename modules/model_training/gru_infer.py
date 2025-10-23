# modules/model_training/gru_infer.py
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Any, Optional, Sequence

import numpy as np
import tensorflow as tf


DEFAULT_CLASSES: Sequence[str] = ("idle", "down", "up")


def _enable_tf_memory_growth() -> None:
    """
    تشغيل memory growth على كروت NVIDIA لتجنب حجز كل الذاكرة من البداية.
    آمن على الأجهزة بدون GPU.
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        # ما نكسر الدنيا إذا فشل
        pass


def _load_meta(meta_path: Path) -> dict:
    """
    يقرأ ملف meta JSON اختياري:
      {
        "seq_len": 30,
        "feat_dim": 6,
        "classes": ["idle","down","up"],
        "feat_mean": [..],      # اختياري
        "feat_std":  [..]       # اختياري
      }
    """
    try:
        if meta_path and meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _infer_input_shape(model: tf.keras.Model) -> tuple[int, int]:
    """
    يستنتج (T, F) من شكل الإدخال: (None, T, F).
    لو None يرجّع قيم افتراضية معقولة.
    """
    shp = model.inputs[0].shape  # TensorShape([None, T, F])
    seq_len = int(shp[1]) if shp[1] is not None else 30
    feat_dim = int(shp[2]) if shp[2] is not None else 6
    return seq_len, feat_dim


class GRUInfer:
    """
    واجهة استدلال بسيطة لنموذج GRU مدرّب على تسلسلات ميزات.
    - تتراكم الميزات فريماً بفريم داخل نافذة طولها seq_len
    - عند الاكتمال، يتم الاستدلال ويرجع:
        {"state": <label or None>, "probs": [..] or None, "quality": float or None}

    الاستخدام:
        infer = GRUInfer("outputs/models/squat/squat_model.h5", "outputs/models/squat/squat_meta.json")
        for vec in stream:   # vec: np.ndarray (feat_dim,)
            infer.push(vec)
            if infer.can_infer():
                out = infer.infer()
    """

    def __init__(
        self,
        model_path: str | Path,
        meta_path: str | Path | None = None,
        class_names: Optional[Sequence[str]] = None,
        use_memory_growth: bool = True,
    ):
        _enable_tf_memory_growth() if use_memory_growth else None

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # حمّل الموديل
        self.model: tf.keras.Model = tf.keras.models.load_model(self.model_path.as_posix())

        # إعداد الشكل والأصناف
        seq_len_i, feat_dim_i = _infer_input_shape(self.model)
        meta = _load_meta(Path(meta_path) if meta_path else Path())

        self.seq_len: int = int(meta.get("seq_len", seq_len_i))
        self.feat_dim: int = int(meta.get("feat_dim", feat_dim_i))
        self.class_names: list[str] = list(class_names) if class_names else list(meta.get("classes", DEFAULT_CLASSES))

        # scaler اختياري للتطبيع (إن وُجد في الميتا)
        mu = meta.get("feat_mean")
        std = meta.get("feat_std")
        self._mu: Optional[np.ndarray] = np.array(mu, dtype="float32") if isinstance(mu, (list, tuple)) else None
        self._std: Optional[np.ndarray] = np.array(std, dtype="float32") if isinstance(std, (list, tuple)) else None
        if self._mu is not None and self._std is not None and (
            self._mu.shape != (self.feat_dim,) or self._std.shape != (self.feat_dim,)
        ):
            # تجاهل scaler لو الأبعاد لا تطابق
            self._mu, self._std = None, None

        # المخزن المؤقت
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)

    # --------- إدارة البافر ---------
    def reset(self) -> None:
        """تفريغ النافذة."""
        self.buffer.clear()

    def push(self, feats_vec: np.ndarray) -> None:
        """
        إضافة متجه ميزات لفريم واحد.
        يتوقع شكل (feat_dim,).
        يطبّق تنظيف NaN/Inf وتطبيع إذا متاح.
        """
        v = np.asarray(feats_vec, dtype="float32")
        if v.ndim != 1 or v.shape[0] != self.feat_dim:
            raise ValueError(f"Expected vector shape ({self.feat_dim},), got {v.shape}")

        # تنظيف
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        # تطبيع اختياري
        if self._mu is not None and self._std is not None:
            std = np.where(self._std == 0.0, 1.0, self._std)
            v = (v - self._mu) / std

        self.buffer.append(v)

    def ready(self) -> bool:
        """النافذة جاهزة للاستدلال؟"""
        return len(self.buffer) == self.seq_len

    can_infer = ready  # اسم بديل لمن يحب الأسماء الطويلة

    # --------- الاستدلال ---------
    def infer(self) -> Dict[str, Any]:
        """
        ينفّذ الاستدلال ويرجع dict:
          - state: اسم الحالة أو None
          - probs: قائمة احتمالات الطرْح (C,) أو None
          - quality: أعلى احتمال (float) أو None
        """
        if not self.ready():
            return {"state": None, "probs": None, "quality": None}

        x = np.stack(self.buffer, axis=0)      # (T, F)
        x = np.expand_dims(x, axis=0)          # (1, T, F)

        probs = self.model.predict(x, verbose=0)[0]  # (C,)
        # تأكد من أنها احتمالات
        probs = np.asarray(probs, dtype="float32")
        if probs.ndim != 1:
            probs = probs.reshape(-1)

        idx = int(np.argmax(probs)) if probs.size > 0 else -1
        state = self.class_names[idx] if 0 <= idx < len(self.class_names) else None
        quality = float(np.max(probs)) if probs.size > 0 else None

        return {"state": state, "probs": probs.tolist(), "quality": quality}
