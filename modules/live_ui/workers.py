#!/usr/bin/env python3
# modules/live_ui/workers.py
"""
Background workers for video capture and pose analysis.
"""

import time
from typing import Optional, Dict, Any, Union
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from modules.live_analyzer.inference import GRUInference
from modules.live_analyzer.tts import speak
from modules.data_extraction.mediapipe_runner import PoseRunner
from modules.data_extraction.yolo_runner import YoloRunner
from modules.common.feature_builder import build_features


class VideoWorker(QThread):
    """Worker thread for video capture (camera or file)."""
    
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(dict)
    error = Signal(str)
    
    def __init__(self, source: Union[int, str], target_fps: float = 30.0, flip: bool = False):
        super().__init__()
        self.source = source
        self.target_fps = target_fps
        self.flip = flip
        self._running = False
        self.cap = None
        
    def stop(self):
        """Stop the video capture."""
        self._running = False
    
    def run(self):
        """Main capture loop."""
        self._running = True
        
        # Open video source
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(str(self.source))
        
        if not self.cap.isOpened():
            self.error.emit(f"فشل فتح المصدر: {self.source}")
            return
        
        # Set resolution if camera
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_time = 1.0 / self.target_fps
        last_emit_time = 0
        frame_count = 0
        fps_start = time.time()
        
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                # End of video or camera disconnected
                if isinstance(self.source, str):
                    # Video file ended, could loop or stop
                    break
                else:
                    self.error.emit("فقدان الاتصال بالكاميرا")
                    break
            
            # Flip if needed
            if self.flip:
                frame = cv2.flip(frame, 1)
            
            # Throttle to target FPS
            now = time.time()
            if now - last_emit_time >= frame_time:
                self.frame_ready.emit(frame)
                last_emit_time = now
                frame_count += 1
            
            # Calculate FPS every second
            elapsed = now - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                self.stats_ready.emit({"fps": fps})
                frame_count = 0
                fps_start = now
        
        if self.cap:
            self.cap.release()
        self._running = False


class AnalyzerWorker(QThread):
    """Background worker for pose analysis and GRU inference."""
    
    overlay_ready = Signal(np.ndarray)
    rep_event = Signal(dict)
    status = Signal(str)
    error = Signal(str)
    
    def __init__(self, exercise: str, speak_out: bool = True, 
                 yolo_weights=None, confidence_threshold: float = 0.5):
        super().__init__()
        
        self.exercise = exercise
        self.speak_out = speak_out
        self._running = False
        
        # Frame queue
        self.frame_queue = []
        self.max_queue_size = 5
        
        # Initialize inference engine
        try:
            self.inference_engine = GRUInference(
                exercise=exercise,
                confidence_threshold=confidence_threshold
            )
            self.status.emit("تم تحميل الموديل بنجاح")
        except Exception as e:
            self.error.emit(f"فشل تحميل الموديل: {str(e)}")
            raise
        
        # Initialize pose runner
        try:
            self.pose_runner = PoseRunner(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            self.error.emit(f"فشل تهيئة MediaPipe: {str(e)}")
            raise
        
        # Initialize YOLO (optional)
        self.yolo_runner = None
        if yolo_weights:
            try:
                self.yolo_runner = YoloRunner(weights=yolo_weights)
                self.status.emit("تم تحميل YOLO")
            except Exception as e:
                print(f"[WARN] YOLO not loaded: {e}")
        
        # State tracking
        self.rep_count = 0
        self.current_features = {}
        self.fsm_state = "IDLE"
        self.rep_start_time = 0
        self.last_rep_time = 0
        self.min_rep_cooldown = 0.5  # seconds
        
        # Squat FSM thresholds
        self.REST_KNEE = 165.0
        self.START_KNEE = 155.0
        self.BOTTOM_KNEE = 120.0
        
    def push_frame(self, frame: np.ndarray):
        """Push a new frame for processing (called from GUI thread)."""
        if len(self.frame_queue) < self.max_queue_size:
            self.frame_queue.append(frame.copy())
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
    
    def run(self):
        """Main worker loop."""
        self._running = True
        self.status.emit("جاهز للتحليل...")
        
        while self._running:
            if not self.frame_queue:
                time.sleep(0.005)  # 5ms
                continue
            
            frame = self.frame_queue.pop(0)
            
            try:
                overlay = self.process_frame(frame)
                if overlay is not None:
                    self.overlay_ready.emit(overlay)
            except Exception as e:
                self.error.emit(f"خطأ: {str(e)}")
                print(f"[ERROR] Frame processing failed: {e}")
        
        # Cleanup
        self.pose_runner.close()
        self.status.emit("تم الإيقاف")
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame."""
        
        # Extract pose landmarks
        landmarks = self.pose_runner.process_bgr(frame)
        if landmarks is None:
            # No pose detected
            return self.draw_no_pose(frame)
        
        # YOLO detection (optional)
        detections = None
        if self.yolo_runner:
            try:
                detections = self.yolo_runner.infer_bgr(frame)
            except Exception as e:
                print(f"[WARN] YOLO inference failed: {e}")
        
        # Build features
        try:
            features = build_features(landmarks, detections, self.exercise)
            self.current_features = features.copy()
        except Exception as e:
            print(f"[ERROR] Feature building failed: {e}")
            return frame
        
        # Push to inference buffer
        self.inference_engine.push_frame_features(features)
        
        # Update FSM for rep counting
        self.fsm_update(features)
        
        # Draw overlay
        overlay = self.draw_overlay(frame, landmarks, features)
        
        return overlay
    
    def fsm_update(self, features: Dict[str, Any]):
        """Finite state machine for rep counting."""
        
        # Get average knee angle
        knee_L = features.get("sq_knee_angle_L", 180.0)
        knee_R = features.get("sq_knee_angle_R", 180.0)
        
        # Handle None or NaN values
        if knee_L is None or np.isnan(knee_L):
            knee_L = 180.0
        if knee_R is None or np.isnan(knee_R):
            knee_R = 180.0
        
        knee_avg = (knee_L + knee_R) / 2.0
        
        current_time = time.time()
        
        if self.fsm_state == "IDLE":
            # Check if starting to descend
            if knee_avg <= self.START_KNEE:
                self.fsm_state = "INREP"
                self.rep_start_time = current_time
                self.status.emit("🔽 نزول...")
        
        elif self.fsm_state == "INREP":
            # Check if returning to standing
            if knee_avg >= self.REST_KNEE:
                # Check cooldown to avoid double counting
                if current_time - self.last_rep_time >= self.min_rep_cooldown:
                    self.fsm_state = "IDLE"
                    self.last_rep_time = current_time
                    self.handle_rep_completed()
                else:
                    # Too soon, ignore
                    self.fsm_state = "IDLE"
    
    def handle_rep_completed(self):
        """Called when a rep is detected."""
        
        # Get prediction from accumulated buffer
        result = self.inference_engine.predict_from_buffer()
        
        if result is None:
            # Not enough frames in buffer yet
            self.status.emit("⚠️ بيانات غير كافية")
            return
        
        label = result["label"]
        prob = result["probability"]
        
        # Generate feedback
        feedback = self.inference_engine.get_form_feedback(
            self.current_features,
            label
        )
        
        # Increment counter
        self.rep_count += 1
        
        # Emit event to GUI
        self.rep_event.emit({
            "rep": self.rep_count,
            "pred": label,
            "prob": prob,
            "reason": feedback
        })
        
        # Speak feedback if enabled
        if self.speak_out and feedback:
            try:
                speak(feedback)
            except Exception as e:
                print(f"[WARN] TTS failed: {e}")
        
        # Update status
        emoji = "✅" if label == "Correct" else "❌"
        self.status.emit(f"{emoji} العدة {self.rep_count}: {label}")
        
        # Clear buffer for next rep
        self.inference_engine.clear_buffer()
    
    def draw_overlay(self, frame: np.ndarray, landmarks: Dict, 
                    features: Dict[str, Any]) -> np.ndarray:
        """Draw pose overlay and info on frame."""
        
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        # Draw skeleton connections
        connections = [
            ("shoulder_L", "shoulder_R"),
            ("shoulder_L", "elbow_L"),
            ("elbow_L", "wrist_L"),
            ("shoulder_R", "elbow_R"),
            ("elbow_R", "wrist_R"),
            ("shoulder_L", "hip_L"),
            ("shoulder_R", "hip_R"),
            ("hip_L", "hip_R"),
            ("hip_L", "knee_L"),
            ("knee_L", "ankle_L"),
            ("hip_R", "knee_R"),
            ("knee_R", "ankle_R"),
        ]
        
        # Draw lines
        for pt1, pt2 in connections:
            if pt1 in landmarks and pt2 in landmarks:
                x1, y1 = int(landmarks[pt1][0] * w), int(landmarks[pt1][1] * h)
                x2, y2 = int(landmarks[pt2][0] * w), int(landmarks[pt2][1] * h)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Draw keypoints
        for name, pos in landmarks.items():
            if name.endswith("_conf") or name.endswith("_mid"):
                continue
            x, y = pos
            px, py = int(x * w), int(y * h)
            cv2.circle(overlay, (px, py), 6, (0, 255, 255), -1)
            cv2.circle(overlay, (px, py), 8, (255, 0, 0), 2)
        
        # Draw info overlay
        knee_L = features.get("sq_knee_angle_L", 0)
        knee_R = features.get("sq_knee_angle_R", 0)
        knee_avg = (knee_L + knee_R) / 2.0
        
        # Info background
        cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(overlay, (5, 5), (300, 100), (255, 255, 255), 2)
        
        # Text info
        cv2.putText(overlay, f"Reps: {self.rep_count}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(overlay, f"Knee: {knee_avg:.0f} deg", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        state_text = "Standing" if self.fsm_state == "IDLE" else "Squatting"
        state_color = (100, 200, 255) if self.fsm_state == "IDLE" else (255, 150, 0)
        cv2.putText(overlay, state_text, (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        return overlay
    
    def draw_no_pose(self, frame: np.ndarray) -> np.ndarray:
        """Draw warning when no pose is detected."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        
        text = "No pose detected"
        text_ar = "لا يوجد شخص في الكادر"
        
        cv2.putText(overlay, text, (w//2 - 150, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(overlay, text_ar, (w//2 - 150, h//2 + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return overlay