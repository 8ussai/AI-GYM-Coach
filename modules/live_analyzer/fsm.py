#!/usr/bin/env python3
# modules/live_analyzer/fsm.py

from dataclasses import dataclass

SIG_COL = "sq_knee_angle_mean"

@dataclass
class SquatThresholds:
    rest_knee_deg: float = 155.0
    start_knee_deg: float = 145.0
    bottom_knee_deg: float = 110.0
    max_torso_incline: float = 20.0
    max_pelvis_drop: float = 0.20
    min_stance_ratio: float = 0.8
    max_stance_ratio: float = 1.6
    min_rep_time_s: float = 0.5
    max_rep_time_s: float = 6.0
    cooldown_s: float = 0.15
    rise_from_bottom_deg: float = 15.0

class SquatFSM:
    def __init__(self, th: SquatThresholds):
        self.th = th
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.start_t = None
        self.min_val = None
        self.last_end_t = -1e9

    def step(self, t_s: float, knee_mean: float):
        if knee_mean is None:
            return None

        if self.state == "IDLE":
            if knee_mean <= self.th.start_knee_deg:
                self.state = "INREP"
                self.start_t = t_s
                self.min_val = knee_mean
                return ("START", self.start_t)

        elif self.state == "INREP":
            if self.min_val is None or knee_mean < self.min_val:
                self.min_val = knee_mean
            bottom = self.min_val if self.min_val is not None else knee_mean
            rise = knee_mean - bottom
            hit_rest = knee_mean >= self.th.rest_knee_deg
            hit_rise = rise >= self.th.rise_from_bottom_deg
            if hit_rest or hit_rise:
                end_t = t_s
                dur = end_t - (self.start_t or end_t)
                bottom_angle = bottom
                self.state = "COOLDOWN"
                self.last_end_t = end_t
                s, e = self.start_t, end_t
                self.start_t, self.min_val = None, None
                if self.th.min_rep_time_s <= dur <= self.th.max_rep_time_s:
                    return ("END", (s, e, bottom_angle))

        elif self.state == "COOLDOWN":
            if (t_s - self.last_end_t) >= self.th.cooldown_s:
                self.state = "IDLE"

        return None
