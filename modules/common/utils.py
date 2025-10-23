# modules/common/utils.py
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Sequence, Any

import numpy as np


def deque_to_array(dq: Deque, dtype=float) -> np.ndarray:
    """تحويل deque إلى ndarray بسرعة."""
    if not dq:
        return np.empty((0,), dtype=dtype)
    return np.asarray(list(dq), dtype=dtype)


def smooth_signal(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    تنعيم بسيط (moving average) بطول نافذة k.
    يحافظ على الطول ويستخدم padding من الحواف.
    """
    if x.size == 0 or k <= 1:
        return x
    k = int(k)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    ker = np.ones(k, dtype=float) / float(k)
    return np.convolve(xp, ker, mode="valid")


def safe_float(v: Any, default: float = 0.0) -> float:
    """تحويل آمن إلى float مع قيمة افتراضية."""
    try:
        return float(v)
    except Exception:
        return float(default)


def diff1(x: Sequence[float]) -> np.ndarray:
    """مشتقة أولى بسيطة على متتالية أعداد."""
    if not x:
        return np.empty((0,), dtype=float)
    arr = np.asarray(x, dtype=float)
    if arr.size < 2:
        return np.zeros_like(arr)
    return np.diff(arr, n=1, prepend=arr[0])
