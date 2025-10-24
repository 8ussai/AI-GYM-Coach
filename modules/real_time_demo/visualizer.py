"""
Visualization utilities for real-time squat form analysis.

Draws MediaPipe skeleton, form metrics overlays, and feedback on frames.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


class Visualizer:
    def __init__(self):
        self.skeleton_color = (0, 255, 0)
        self.joint_color = (255, 0, 0)
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)
        self.correct_color = (0, 255, 0)
        self.incorrect_color = (0, 0, 255)
        self.warning_color = (0, 165, 255)
        
        self.connections = [
            ("shoulder_L", "shoulder_R"),
            ("shoulder_L", "hip_L"),
            ("shoulder_R", "hip_R"),
            ("hip_L", "hip_R"),
            ("shoulder_L", "elbow_L"),
            ("elbow_L", "wrist_L"),
            ("shoulder_R", "elbow_R"),
            ("elbow_R", "wrist_R"),
            ("hip_L", "knee_L"),
            ("knee_L", "ankle_L"),
            ("hip_R", "knee_R"),
            ("knee_R", "ankle_R"),
        ]

    def draw_skeleton(self, frame: np.ndarray, landmarks: Dict) -> np.ndarray:
        h, w = frame.shape[:2]
        frame_copy = frame.copy()
        
        for conn in self.connections:
            pt1_name, pt2_name = conn
            if pt1_name in landmarks and pt2_name in landmarks:
                pt1 = landmarks[pt1_name]
                pt2 = landmarks[pt2_name]
                
                if isinstance(pt1, tuple) and isinstance(pt2, tuple):
                    x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                    x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                    cv2.line(frame_copy, (x1, y1), (x2, y2), self.skeleton_color, 2)
        
        for joint_name, pos in landmarks.items():
            if isinstance(pos, tuple) and not joint_name.endswith("_conf") and not joint_name.endswith("_mid"):
                x, y = int(pos[0] * w), int(pos[1] * h)
                cv2.circle(frame_copy, (x, y), 5, self.joint_color, -1)
        
        return frame_copy

    def draw_metrics(self, frame: np.ndarray, features: Dict, state: str, rep_count: int) -> np.ndarray:
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        panel_width = 300
        panel_height = h
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:, :] = (40, 40, 40)
        
        y_offset = 30
        line_height = 35
        
        cv2.putText(panel, "SQUAT FORM ANALYSIS", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.text_color, 2)
        y_offset += line_height
        
        cv2.line(panel, (10, y_offset), (panel_width - 10, y_offset), (100, 100, 100), 1)
        y_offset += 20
        
        cv2.putText(panel, f"Reps: {rep_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        y_offset += line_height
        
        state_color = self.correct_color if state == "INREP" else self.text_color
        cv2.putText(panel, f"State: {state}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 1)
        y_offset += line_height + 10
        
        cv2.putText(panel, "METRICS:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        knee_L = features.get("sq_knee_angle_L", 0)
        knee_R = features.get("sq_knee_angle_R", 0)
        if not np.isnan(knee_L):
            cv2.putText(panel, f"Knee L: {knee_L:.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        y_offset += line_height
        
        if not np.isnan(knee_R):
            cv2.putText(panel, f"Knee R: {knee_R:.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        y_offset += line_height
        
        torso = features.get("sq_torso_incline", 0)
        if not np.isnan(torso):
            torso_color = self.incorrect_color if torso > 20 else self.correct_color
            cv2.putText(panel, f"Torso: {torso:.1f}deg", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, torso_color, 1)
        y_offset += line_height
        
        pelvis = features.get("sq_pelvis_drop", 0)
        if not np.isnan(pelvis):
            pelvis_color = self.incorrect_color if pelvis > 0.2 else self.correct_color
            cv2.putText(panel, f"Pelvis: {pelvis:.3f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, pelvis_color, 1)
        y_offset += line_height
        
        stance = features.get("sq_stance_ratio", 0)
        if not np.isnan(stance):
            stance_color = self.incorrect_color if (stance < 0.8 or stance > 1.6) else self.correct_color
            cv2.putText(panel, f"Stance: {stance:.2f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, stance_color, 1)
        y_offset += line_height
        
        bar = features.get("sq_bar_present", 0)
        bar_text = "Yes" if bar == 1 else "No"
        cv2.putText(panel, f"Bar: {bar_text}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)
        
        combined = np.hstack([frame_copy, panel])
        return combined

    def draw_rep_feedback(self, frame: np.ndarray, rep_label: str, reason: str, 
                         prediction: Optional[str] = None, confidence: Optional[float] = None) -> np.ndarray:
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        color = self.correct_color if rep_label == "Correct" else self.incorrect_color
        
        feedback_y = 50
        cv2.putText(frame_copy, f"Rep: {rep_label}", (20, feedback_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        if reason:
            reason_text = reason.replace("_", " ").title()
            cv2.putText(frame_copy, f"Issue: {reason_text}", (20, feedback_y + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.warning_color, 2)
        
        if prediction is not None and confidence is not None:
            pred_color = self.correct_color if prediction == "Correct" else self.incorrect_color
            cv2.putText(frame_copy, f"AI: {prediction} ({confidence*100:.1f}%)", 
                       (20, feedback_y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
        
        return frame_copy

    def create_text_overlay(self, text: str, color: Tuple[int, int, int] = None) -> np.ndarray:
        if color is None:
            color = self.text_color
        
        img = np.zeros((100, 400, 3), dtype=np.uint8)
        cv2.putText(img, text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return img
