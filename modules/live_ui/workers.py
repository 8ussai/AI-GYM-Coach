#!/usr/bin/env python3
# modules/live_ui/workers.py

import time
from typing import Optional, Union, List
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QObject

from modules.live_analyzer.feature_extractor import FeatureExtractor, YOLOConfig
from modules.live_analyzer.fsm import SquatFSM, SquatThresholds, SIG_COL
from modules.live_analyzer.inference import LiveInference
from modules.live_analyzer.tts import speak
from modules.common.paths import get_models_dir

class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, source: Union[int, str], target_fps: float = 30.0, flip: bool = True):
        super().__init__()
        self.source = source
        self.target_fps = float(target_fps)
        self.flip = bool(flip)
        self._running = False
        self._cap = None

    def stop(self):
        self._running = False

    def open_source(self) -> bool:
        if isinstance(self.source, int):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap or not self._cap.isOpened():
            self.error.emit(f"Failed to open source: {self.source}")
            return False
        return True

    def run(self):
        self._running = True
        if not self.open_source():
            return
        interval = 1.0 / max(1.0, self.target_fps)
        frames = 0
        t0 = time.time()

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.error.emit("Stream ended")
                break
            if self.flip:
                frame = cv2.flip(frame, 1)

            self.frame_ready.emit(frame)
            frames += 1
            dt = time.time() - t0
            fps = frames / dt if dt > 0 else 0.0
            self.stats_ready.emit({"fps": fps})

            # frame pacing
            time.sleep(max(0.0, interval))

        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = None


class AnalyzerWorker(QThread):
    # GUI signals
    overlay_ready = Signal(np.ndarray)   # frame with optional overlay
    rep_event = Signal(dict)             # {"rep": n, "pred": "Correct/Incorrect", "prob": 0.xx, "reason": "..."}
    status = Signal(str)
    error = Signal(str)

    def __init__(self, exercise: str = "squat", speak_out: bool = False,
                 yolo_weights: Optional[str] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.exercise = exercise
        self.speak_out = bool(speak_out)
        self.yolo_weights = yolo_weights
        self._running = False
        self._frame_queue: List[tuple] = []  # (t_s, frame_bgr)
        self._last_t0 = None

        # analysis state
        self.fe = None
        self.fsm = SquatFSM(SquatThresholds())
        self.infer = None
        self.frames_buf: List[dict] = []
        self.rep_counter = 0

    def push_frame(self, frame_bgr: np.ndarray):
        if not self._running:
            return
        t_now = time.time()
        if self._last_t0 is None:
            self._last_t0 = t_now
        t_s = t_now - self._last_t0
        self._frame_queue.append((t_s, frame_bgr))

    def stop(self):
        self._running = False

    def _init_models(self):
        # Feature extractor
        ycfg = YOLOConfig(weights_path=self.yolo_weights) if self.yolo_weights else None
        self.fe = FeatureExtractor(yolo_cfg=ycfg)
        # Inference
        self.infer = LiveInference(get_models_dir(self.exercise), self.exercise)

    def _close_models(self):
        if self.fe:
            try:
                self.fe.close()
            except Exception:
                pass
        self.fe = None

    def run(self):
        try:
            self._running = True
            self.status.emit("Initializing...")
            self._init_models()
            self.status.emit("Running")
        except Exception as e:
            self.error.emit(f"Init failed: {e}")
            return

        while self._running:
            if not self._frame_queue:
                time.sleep(0.005)
                continue

            t_s, frame = self._frame_queue.pop(0)
            # extract features
            feats, lm = self.fe.compute_features(frame)

            # compute knee mean for FSM
            knee_mean = float(np.nanmean([feats["sq_knee_angle_L"], feats["sq_knee_angle_R"]]))
            evt = self.fsm.step(t_s, knee_mean)

            # keep minimal per-frame row in buffer
            row = {"t_s": t_s}
            row.update(feats)
            row[SIG_COL] = knee_mean
            self.frames_buf.append(row)

            # overlay (optional minimal draw)
            overlay = frame.copy()
            # draw simple text HUD
            cv2.putText(overlay, f"t={t_s:0.2f}s   knee={knee_mean:0.1f}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,230,200), 2, cv2.LINE_AA)
            self.overlay_ready.emit(overlay)

            if evt is None:
                continue

            if evt[0] == "START":
                # could beep or show state
                pass

            elif evt[0] == "END":
                start_t, end_t, bottom_angle = evt[1]
                X = self.infer.build_sequence(self.frames_buf, start_t, end_t)
                if X is None:
                    continue
                prob = self.infer.predict_prob(X)
                pred = "Correct" if prob >= 0.5 else "Incorrect"

                # quick rule reason for UI
                reason = ""
                th = self.fsm.th
                # compute indicative aggregates:
                seg = [r for r in self.frames_buf if start_t <= r["t_s"] <= end_t]
                if seg:
                    torso_max = float(np.nanmax([r["sq_torso_incline"] for r in seg]))
                    pelvis_max = float(np.nanmax([r["sq_pelvis_drop"] for r in seg]))
                    stance_mean = float(np.nanmean([r["sq_stance_ratio"] for r in seg]))
                    bottom = float(np.nanmin([r[SIG_COL] for r in seg]))
                    if not (bottom <= th.bottom_knee_deg):
                        reason = "low_depth"
                    elif torso_max > th.max_torso_incline:
                        reason = "back_rounding"
                    elif pelvis_max > th.max_pelvis_drop:
                        reason = "asymmetry"
                    elif stance_mean < th.min_stance_ratio or stance_mean > th.max_stance_ratio:
                        reason = "stance_width"

                self.rep_counter += 1
                msg = {"rep": self.rep_counter, "pred": pred, "prob": float(prob), "reason": reason}
                self.rep_event.emit(msg)
                speak(f"Rep {self.rep_counter}: {pred}", enable=self.speak_out)

                # memory hygiene: keep only last 2s of buffer after end
                keep_after = end_t - 2.0
                self.frames_buf = [r for r in self.frames_buf if r["t_s"] >= keep_after]

        self._close_models()
        self.status.emit("Stopped")
