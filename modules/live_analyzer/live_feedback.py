from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------- Optional MediaPipe ----------------------------
try:
    import mediapipe as mp  # type: ignore
    _HAVE_MP = True
except Exception:  # pragma: no cover
    mp = None
    _HAVE_MP = False


# ------------------------------- Utilities ---------------------------------

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute angle ABC (with B at the vertex) in degrees.
    Vectors BA and BC.
    """
    v1 = a - b
    v2 = c - b
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom <= 1e-6:
        return 180.0
    cosang = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b."""
    return a + (b - a) * t


class EMA:
    """Exponential moving average for scalars."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self.val: Optional[float] = None

    def update(self, x: float) -> float:
        if self.val is None:
            self.val = x
        else:
            self.val = _lerp(self.val, x, self.alpha)
        return self.val

    def reset(self):
        """Reset the EMA state."""
        self.val = None


# --------------------------------- TTS gate --------------------------------
class _TTSGate:
    """Rate-limited TTS wrapper to prevent spam."""
    
    def __init__(self, speak_fn: Optional[Callable[[str], None]], cooldown_s: float = 1.2):
        self.speak_fn = speak_fn
        self.cooldown = cooldown_s
        self.last_text = ""
        self.last_t = 0.0
        self.enabled = True

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def say(self, text: str):
        """Speak text if cooldown has passed and enabled."""
        now = time.perf_counter()
        if not self.enabled or not self.speak_fn:
            return
        
        # Skip if same text within cooldown
        if text == self.last_text and now - self.last_t < self.cooldown:
            return
        
        # Skip if any speech within cooldown
        if now - self.last_t < self.cooldown:
            return
            
        self.last_text = text
        self.last_t = now
        try:
            self.speak_fn(text)
        except Exception as e:
            print(f"[TTS Error] {e}")


# ------------------------------- Pose wrapper ------------------------------
class _Pose:
    """Tiny wrapper around MediaPipe Pose with a consistent output API.

    Returns: (ok: bool, landmarks: dict[idx] -> (x,y), visibility: dict[idx]->v)
    Coordinates are pixel coordinates in the given frame.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.ok = _HAVE_MP
        self.width = width
        self.height = height
        self._pose = None
        self._mp_pose = None
        
        if _HAVE_MP:
            try:
                self._mp_pose = mp.solutions.pose
                # Enable smooth landmarks for nicer live output
                self._pose = self._mp_pose.Pose(
                    model_complexity=1, 
                    enable_segmentation=False,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"[MediaPipe Error] Failed to initialize: {e}")
                self.ok = False

    def detect(self, frame_bgr: np.ndarray) -> Tuple[bool, Dict[int, Tuple[float, float]], Dict[int, float]]:
        """Detect pose landmarks in the frame."""
        if not self.ok or self._pose is None:
            return False, {}, {}
        
        try:
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Make image read-only to improve performance
            frame_rgb.flags.writeable = False
            res = self._pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            if not res or not res.pose_landmarks:
                return False, {}, {}
            
            lm = res.pose_landmarks.landmark
            coords: Dict[int, Tuple[float, float]] = {}
            vis: Dict[int, float] = {}
            
            for i, p in enumerate(lm):
                coords[i] = (p.x * w, p.y * h)
                vis[i] = p.visibility
                
            return True, coords, vis
            
        except Exception as e:
            print(f"[Pose Detection Error] {e}")
            return False, {}, {}

    def close(self):
        """Clean up resources."""
        if self._pose:
            try:
                self._pose.close()
            except Exception:
                pass


# ------------------------------- Squat FSM ---------------------------------
@dataclass
class SquatThresholds:
    """Thresholds for squat detection state machine."""
    rest_knee_deg: float = 170.0   # above -> idle (standing up)
    start_knee_deg: float = 160.0  # crossing downward -> start
    bottom_knee_deg: float = 100.0 # must go below to count depth
    min_rep_time_s: float = 0.4
    max_rep_time_s: float = 6.0
    cooldown_s: float = 0.15


class SquatFSM:
    """Finite state machine for squat rep counting."""
    
    def __init__(self, thr: SquatThresholds):
        self.t = thr
        self.state = "idle"  # idle -> down -> up
        self.rep_timer = 0.0
        self.last_state_change = time.perf_counter()
        self.reps = 0
        self.reached_bottom = False

    def reset(self):
        """Reset FSM to initial state."""
        self.state = "idle"
        self.rep_timer = 0.0
        self.last_state_change = time.perf_counter()
        self.reps = 0
        self.reached_bottom = False

    def update(self, knee_deg_mean: float) -> Tuple[List[str], Optional[str]]:
        """Advance FSM given the mean knee angle.
        
        Returns: (messages, cue_for_tts)
        """
        msgs: List[str] = []
        cue: Optional[str] = None
        now = time.perf_counter()
        dt = now - self.last_state_change

        if self.state == "idle":
            if knee_deg_mean < self.t.start_knee_deg:
                self.state = "down"
                self.rep_timer = 0.0
                self.reached_bottom = False
                self.last_state_change = now
                msgs.append("ابدأ النزول")
                cue = "ابدأ النزول"
                
        elif self.state == "down":
            self.rep_timer += dt
            
            if knee_deg_mean <= self.t.bottom_knee_deg:
                self.reached_bottom = True
                self.state = "up"
                self.last_state_change = now
                msgs.append("اطلع لفوق")
                cue = "اطلع لفوق"
            elif knee_deg_mean > self.t.rest_knee_deg:
                # Aborted rep - didn't reach depth
                self.state = "idle"
                self.last_state_change = now
                msgs.append("أعد المحاولة")
                cue = "أعد المحاولة"
            elif self.rep_timer > self.t.max_rep_time_s:
                # Taking too long
                self.state = "idle"
                self.last_state_change = now
                msgs.append("وقت طويل جداً")
                cue = "أعد المحاولة"
                
        elif self.state == "up":
            self.rep_timer += dt
            
            if knee_deg_mean >= self.t.rest_knee_deg:
                # Rep potentially done
                if self.t.min_rep_time_s <= self.rep_timer <= self.t.max_rep_time_s and self.reached_bottom:
                    self.reps += 1
                    msgs.append(f"عدّة مكتملة: {self.reps}")
                    cue = "عدّة مكتملة"
                else:
                    msgs.append("العدّة غير صالحة")
                    cue = "العدّة غير صالحة"
                    
                self.state = "idle"
                self.last_state_change = now
                self.reached_bottom = False
            elif knee_deg_mean < self.t.start_knee_deg:
                # Dipped down again - keep tracking
                pass
            elif self.rep_timer > self.t.max_rep_time_s:
                # Taking too long
                self.state = "idle"
                self.last_state_change = now
                msgs.append("وقت طويل جداً")
                
        return msgs, cue


# ---------------------------- Drawing utilities ----------------------------
# MediaPipe pose landmark indices for body connections
_LIMBS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
    (11, 12),            # shoulders
    (23, 24),            # hips
    (11, 23),            # left torso
    (12, 24)             # right torso
]


def _draw_landmarks(img: np.ndarray, coords: Dict[int, Tuple[float, float]], vis: Dict[int, float]):
    """Draw pose landmarks and skeleton on image."""
    # Draw connections first (so they appear behind joints)
    for a, b in _LIMBS:
        if a in coords and b in coords and vis.get(a, 0) > 0.3 and vis.get(b, 0) > 0.3:
            pa = (int(coords[a][0]), int(coords[a][1]))
            pb = (int(coords[b][0]), int(coords[b][1]))
            cv2.line(img, pa, pb, (0, 200, 255), 2)
    
    # Draw joints on top
    for i, (x, y) in coords.items():
        if vis.get(i, 0) < 0.3:
            continue
        cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.circle(img, (int(x), int(y)), 5, (255, 255, 255), 1)


# --------------------------------- Engine ----------------------------------
class LiveFeedbackEngine:
    """Main analyzer that processes frames and returns an annotated frame plus
    textual feedback. Currently supports squat exercise tracking.
    """

    def __init__(self, exercise: str = "squat", tts_callback: Optional[Callable[[str], None]] = None):
        self.exercise = exercise
        self.pose = _Pose()
        self.knee_ema = EMA(alpha=0.3)
        self.thr = SquatThresholds()
        self.fsm = SquatFSM(self.thr)
        self.tts = _TTSGate(tts_callback, cooldown_s=1.2)
        self._frame_count = 0

    # ----------------------------- Control API ----------------------------
    def set_exercise(self, exercise: str):
        """Change the exercise type."""
        self.exercise = exercise
        # Could add different FSMs for different exercises here

    def enable_tts(self, enabled: bool):
        """Enable or disable text-to-speech feedback."""
        self.tts.set_enabled(enabled)

    def reset_state(self):
        """Reset all tracking state."""
        self.knee_ema.reset()
        self.fsm.reset()
        self._frame_count = 0

    # ------------------------------- Helpers ------------------------------
    def _extract_squat_angles(
        self, 
        coords: Dict[int, Tuple[float, float]], 
        vis: Dict[int, float]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract left and right knee angles from pose landmarks.
        
        Returns: (left_knee_deg, right_knee_deg) or (None, None) if not visible
        """
        # Required landmarks: hips(23,24), knees(25,26), ankles(27,28)
        required = [23, 24, 25, 26, 27, 28]
        
        if any(i not in coords or vis.get(i, 0.0) < 0.3 for i in required):
            return None, None
        
        try:
            # Left knee: hip(23) - knee(25) - ankle(27)
            l_knee = _angle_deg(
                np.array(coords[23]), 
                np.array(coords[25]), 
                np.array(coords[27])
            )
            
            # Right knee: hip(24) - knee(26) - ankle(28)
            r_knee = _angle_deg(
                np.array(coords[24]), 
                np.array(coords[26]), 
                np.array(coords[28])
            )
            
            return l_knee, r_knee
            
        except Exception as e:
            print(f"[Angle Extraction Error] {e}")
            return None, None

    def _draw_angle_bar(
        self, 
        vis_frame: np.ndarray, 
        knee_s: float, 
        bar_x: int = 20
    ):
        """Draw vertical angle bar with thresholds."""
        h, w = vis_frame.shape[:2]
        
        # Draw threshold lines
        thresholds = [
            ("rest", self.thr.rest_knee_deg, (0, 200, 0)),
            ("start", self.thr.start_knee_deg, (0, 200, 200)),
            ("bottom", self.thr.bottom_knee_deg, (0, 0, 255)),
        ]
        
        for label, th_val, color in thresholds:
            y = int(_lerp(100, h - 40, (180 - th_val) / 100.0))
            cv2.line(vis_frame, (bar_x, y), (bar_x + 120, y), color, 2)
            cv2.putText(
                vis_frame, 
                f"{label}:{int(th_val)}", 
                (bar_x + 130, y + 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )
        
        # Draw current angle indicator
        y_cur = int(_lerp(100, h - 40, (180 - knee_s) / 100.0))
        cv2.circle(vis_frame, (bar_x + 60, y_cur), 6, (255, 255, 255), -1)
        cv2.circle(vis_frame, (bar_x + 60, y_cur), 7, (0, 0, 0), 1)
        
        # Draw angle value
        cv2.putText(
            vis_frame, 
            f"knee:{knee_s:.1f}", 
            (bar_x, 80), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )

    def _draw_state_info(self, vis_frame: np.ndarray):
        """Draw rep counter and state info."""
        h, w = vis_frame.shape[:2]
        
        # State background
        cv2.rectangle(vis_frame, (w - 280, 10), (w - 10, 90), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (w - 280, 10), (w - 10, 90), (200, 200, 200), 2)
        
        # State text
        cv2.putText(
            vis_frame, 
            f"state: {self.fsm.state}", 
            (w - 260, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (200, 255, 200), 
            2
        )
        
        # Rep counter
        cv2.putText(
            vis_frame, 
            f"reps: {self.fsm.reps}", 
            (w - 260, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (200, 255, 200), 
            2
        )

    def _get_quality_tip(self, knee_s: float) -> Optional[str]:
        """Get form tip based on current angle and state."""
        if knee_s > self.thr.rest_knee_deg and self.fsm.state == "idle":
            return "شد الجسم واستعد"
        elif knee_s > self.thr.start_knee_deg and self.fsm.state == "down":
            return "انزل أكثر"
        elif knee_s < self.thr.bottom_knee_deg and self.fsm.state == "up":
            return "اطلع للأعلى تدريجيًا"
        return None

    # ------------------------------- Process ------------------------------
    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a single frame and return annotated frame + feedback messages.
        
        Args:
            frame_bgr: Input frame in BGR format (OpenCV standard)
            
        Returns:
            (annotated_frame, messages): Tuple of visualized frame and text messages
        """
        self._frame_count += 1
        messages: List[str] = []
        
        # Validate input
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
            return frame_bgr if frame_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8), ["Invalid frame"]
        
        vis_frame = frame_bgr.copy()
        
        # Detect pose
        ok, coords, vis = self.pose.detect(frame_bgr)
        
        if not ok:
            cv2.putText(
                vis_frame, 
                "No pose detected", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 0, 255), 
                2
            )
            return vis_frame, messages

        # Draw skeleton
        _draw_landmarks(vis_frame, coords, vis)
        
        # Extract knee angles
        lk, rk = self._extract_squat_angles(coords, vis)
        
        if lk is None or rk is None:
            cv2.putText(
                vis_frame, 
                "Pose not stable", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (0, 165, 255), 
                2
            )
            return vis_frame, messages

        # Calculate smoothed knee angle
        knee_mean = (lk + rk) * 0.5
        knee_s = self.knee_ema.update(knee_mean)

        # Draw angle visualization
        self._draw_angle_bar(vis_frame, knee_s)

        # Update FSM and get feedback
        fsm_msgs, cue = self.fsm.update(knee_s)
        messages.extend(fsm_msgs)
        
        if cue:
            self.tts.say(cue)

        # Draw state info
        self._draw_state_info(vis_frame)

        # Get and display quality tips
        tip = self._get_quality_tip(knee_s)
        if tip:
            messages.append(tip)
            # Only speak tips occasionally to avoid spam
            if self._frame_count % 60 == 0:  # Every ~2 seconds at 30fps
                self.tts.say(tip)

        return vis_frame, messages

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.pose.close()
        except Exception:
            pass