# modules/runtime/services/fsm_service.py
from __future__ import annotations
from typing import Tuple, Dict, Any
from modules.common.fsm import SquatFSM, SquatThresholds


class SquatFSMService:
    """
    غلاف بسيط حول FSM السكوات للاستخدام داخل البايبلاين.
    """
    def __init__(self, thresholds: SquatThresholds | None = None):
        self.fsm = SquatFSM(thresholds or SquatThresholds())

    def update(self, knee_angle_mean: float):
        """
        يرجع إيفنت أو None:
          ("RepStart", {...}) أو ("RepEnd", {...})
        """
        return self.fsm.update(knee_angle_mean)

    @property
    def reps(self) -> int:
        return self.fsm.rep_count

    def reset(self) -> None:
        self.fsm.reset()
