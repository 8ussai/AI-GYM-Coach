# modules/runtime/registry.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any

from .pipelines.squat_pipeline import SquatPipeline

# تسجيل التمارين -> كلاس Pipeline
REGISTRY: Dict[str, Callable[..., Any]] = {
    "squats": SquatPipeline,
    # "dumbbell": DumbbellPipeline,
    # "barbell":  BarbellPipeline,
}


def get_pipeline_factory(exercise: str) -> Optional[Callable[..., Any]]:
    """
    يرجع Constructor للبايبلاين حسب اسم التمرين.
    """
    return REGISTRY.get(exercise)
