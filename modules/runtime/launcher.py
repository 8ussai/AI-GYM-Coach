# modules/runtime/launcher.py
from __future__ import annotations
from typing import Optional, Union

from .registry import get_pipeline_factory
from modules.common.workers import BaseModelWorker


def get_worker(exercise: str, source: Union[int, str], settings) -> Optional[BaseModelWorker]:
    """
    نقطة الدخول من الواجهة:
      ترجع Worker مناسب للتمرين المحدد، أو None إذا غير مسجل.
    """
    factory = get_pipeline_factory(exercise)
    if not factory:
        return None
    pipeline = factory(source=source, settings=settings)
    return pipeline.build_worker()
