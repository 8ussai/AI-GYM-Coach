"""
Real-time rep tracking using Finite State Machine (FSM).

Adapted from modules.data_processing.derive_reps for live webcam inference.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import time


@dataclass
class SquatThresholds:
    rest_knee_deg: float = 165.0
    start_knee_deg: float = 155.0
    bottom_knee_deg: float = 120.0
    max_torso_incline: float = 20.0
    max_pelvis_drop: float = 0.20
    min_stance_ratio: float = 0.8
    max_stance_ratio: float = 1.6
    min_rep_time_s: float = 0.5
    max_rep_time_s: float = 6.0
    cooldown_s: float = 0.15


@dataclass
class RepData:
    rep_id: int
    start_time: float
    end_time: float
    bottom_angle: float
    max_torso_incline: float
    max_pelvis_drop: float
    mean_stance_ratio: float
    label: str
    reason: str
    features_sequence: list


class RealtimeRepTracker:
    def __init__(self, thresholds: Optional[SquatThresholds] = None):
        self.th = thresholds or SquatThresholds()
        self.reset()

    def reset(self):
        self.state = "IDLE"
        self.rep_counter = 0
        self.start_time = None
        self.min_angle = None
        self.last_end_time = -1e9
        
        self.current_rep_features = []
        self.current_rep_torso = []
        self.current_rep_pelvis = []
        self.current_rep_stance = []

    def update(self, features: dict, timestamp: float) -> Optional[RepData]:
        knee_L = features.get("sq_knee_angle_L")
        knee_R = features.get("sq_knee_angle_R")
        
        if knee_L is None or knee_R is None or np.isnan(knee_L) or np.isnan(knee_R):
            return None
        
        knee_angle = (knee_L + knee_R) / 2.0
        torso = features.get("sq_torso_incline", 0.0)
        pelvis = features.get("sq_pelvis_drop", 0.0)
        stance = features.get("sq_stance_ratio", 1.0)

        if self.state == "IDLE":
            if knee_angle <= self.th.start_knee_deg:
                self.state = "INREP"
                self.start_time = timestamp
                self.min_angle = knee_angle
                self.current_rep_features = [features.copy()]
                self.current_rep_torso = [torso]
                self.current_rep_pelvis = [pelvis]
                self.current_rep_stance = [stance]

        elif self.state == "INREP":
            self.current_rep_features.append(features.copy())
            self.current_rep_torso.append(torso)
            self.current_rep_pelvis.append(pelvis)
            self.current_rep_stance.append(stance)
            
            if knee_angle < self.min_angle:
                self.min_angle = knee_angle

            if knee_angle >= self.th.rest_knee_deg:
                duration = timestamp - self.start_time
                
                if self.th.min_rep_time_s <= duration <= self.th.max_rep_time_s:
                    self.rep_counter += 1
                    
                    label, reason = self._label_rep(
                        bottom_angle=self.min_angle,
                        torso_incline=max(self.current_rep_torso) if self.current_rep_torso else 0,
                        pelvis_drop=max(self.current_rep_pelvis) if self.current_rep_pelvis else 0,
                        stance_ratio=np.mean(self.current_rep_stance) if self.current_rep_stance else 1.0
                    )
                    
                    rep_data = RepData(
                        rep_id=self.rep_counter,
                        start_time=self.start_time,
                        end_time=timestamp,
                        bottom_angle=self.min_angle,
                        max_torso_incline=max(self.current_rep_torso) if self.current_rep_torso else 0,
                        max_pelvis_drop=max(self.current_rep_pelvis) if self.current_rep_pelvis else 0,
                        mean_stance_ratio=np.mean(self.current_rep_stance) if self.current_rep_stance else 1.0,
                        label=label,
                        reason=reason,
                        features_sequence=self.current_rep_features.copy()
                    )
                    
                    self.last_end_time = timestamp
                    self.state = "COOLDOWN"
                    self.current_rep_features = []
                    self.current_rep_torso = []
                    self.current_rep_pelvis = []
                    self.current_rep_stance = []
                    
                    return rep_data
                else:
                    self.state = "COOLDOWN"
                    self.current_rep_features = []

        elif self.state == "COOLDOWN":
            if timestamp - self.last_end_time >= self.th.cooldown_s:
                self.state = "IDLE"

        return None

    def _label_rep(self, bottom_angle: float, torso_incline: float, 
                   pelvis_drop: float, stance_ratio: float) -> Tuple[str, str]:
        reasons = []
        
        if bottom_angle > self.th.bottom_knee_deg:
            reasons.append("low_depth")
        
        if torso_incline > self.th.max_torso_incline:
            reasons.append("back_rounding")
        
        if pelvis_drop > self.th.max_pelvis_drop:
            reasons.append("asymmetry")
        
        if stance_ratio < self.th.min_stance_ratio or stance_ratio > self.th.max_stance_ratio:
            reasons.append("stance_width")
        
        if not reasons:
            return "Correct", ""
        
        for r in ["low_depth", "back_rounding", "asymmetry", "stance_width"]:
            if r in reasons:
                return "Incorrect", r
        
        return "Incorrect", reasons[0]

    def get_current_state(self):
        return {
            "state": self.state,
            "rep_count": self.rep_counter,
            "in_progress": self.state == "INREP"
        }
