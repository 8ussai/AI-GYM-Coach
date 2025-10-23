# modules/common/fsm.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SquatThresholds:
    """
    عتبات مبسطة لحالة السكوات. عدّلها لتطابق بياناتك.
    """
    start_deg: float = 150.0   # فوقها وضع راحة
    bottom_deg: float = 110.0  # النزول تحتها يعتبر عمق كافٍ
    finish_deg: float = 150.0  # العودة فوقها لإنهاء الريبة
    min_time_s: float = 0.4
    max_time_s: float = 6.0


class SquatFSM:
    """
    FSM بسيط لعدّ التكرارات بناءً على متوسط زاوية الركبتين.
    حالات: idle -> down -> idle
    """
    def __init__(self, th: SquatThresholds):
        self.th = th
        self.state = "idle"
        self.rep_count = 0

    def reset(self):
        self.state = "idle"
        self.rep_count = 0

    def update(self, knee_angle_mean: float):
        """
        يدخل متوسط زاوية الركبة، ويرجع حدث اختياري:
          ("RepStart", {...}) أو ("RepEnd", {...}) أو None
        """
        event = None

        if self.state == "idle":
            if knee_angle_mean < self.th.bottom_deg:
                self.state = "down"
                event = ("RepStart", {"angle": knee_angle_mean})
        elif self.state == "down":
            if knee_angle_mean > self.th.finish_deg:
                self.state = "idle"
                self.rep_count += 1
                event = ("RepEnd", {"angle": knee_angle_mean, "rep": self.rep_count})

        return event
