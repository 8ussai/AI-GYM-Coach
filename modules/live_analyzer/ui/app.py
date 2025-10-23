#!/usr/bin/env python3
"""
AI‑GYM‑COACH – Desktop UI (PySide6) - FIXED VERSION

Key fixes:
1. Fixed indentation in __init__
2. Better error handling for engine initialization
3. Fixed Qt attribute setting (moved before QApplication creation)
4. Added proper numpy array handling
5. Better frame conversion with memory safety

Run:
    python -m modules.live_analyzer.ui.app
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

# --- Optional: import analyzer + TTS. If missing, fall back to no‑op stubs.
try:
    from modules.live_analyzer.live_feedback import LiveFeedbackEngine  # type: ignore
except Exception:  # pragma: no cover
    class LiveFeedbackEngine:  # minimal stub
        def __init__(self, exercise: str = "squat", tts_callback=None):
            self.exercise = exercise
            self._tts = tts_callback
        def set_exercise(self, exercise: str):
            self.exercise = exercise
        def enable_tts(self, enabled: bool):
            pass
        def reset_state(self):
            pass
        def process(self, frame_bgr):
            # No processing. Just return the frame and no messages.
            return frame_bgr, []

try:
    from modules.live_analyzer.tts_utils import TTSManager  # type: ignore
except Exception:  # pragma: no cover
    class TTSManager:  # minimal stub
        def __init__(self, voice: Optional[str] = None):
            pass
        def speak(self, text: str):
            print(f"[TTS] {text}")
        def stop(self):
            pass


# ---------------------------- Capture Thread -----------------------------
class CaptureWorker(QtCore.QThread):
    """Camera / Video reader worker.

    Signals:
        frameReady(np.ndarray): emits BGR frame arrays
        streamEnded(): emitted when video finishes or camera stops
        fps(float): periodically emit measured FPS
    """

    frameReady = QtCore.Signal(object)
    streamEnded = QtCore.Signal()
    fps = QtCore.Signal(float)

    def __init__(self, source: int | str = 0, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self._source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._target_fps: Optional[float] = None  # None = as fast as possible

    def set_source(self, source: int | str):
        self._source = source

    def set_target_fps(self, fps: Optional[float]):
        self._target_fps = fps

    def stop(self):
        self._running = False

    def run(self):
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            self.streamEnded.emit()
            return
        self._running = True

        last = time.perf_counter()
        frames = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break

            self.frameReady.emit(frame)
            frames += 1

            # regulate FPS if a target is set
            if self._target_fps and self._target_fps > 0:
                delay = max(0.0, (1.0 / self._target_fps))
                self.msleep(int(delay * 1000))

            now = time.perf_counter()
            if now - last >= 1.0:
                self.fps.emit(frames / (now - last))
                frames = 0
                last = now

        self._cap.release()
        self.streamEnded.emit()


# ------------------------------- Main UI --------------------------------
@dataclass
class UIState:
    exercise: str = "squat"
    tts_enabled: bool = True
    source_label: str = "Camera 0"


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI‑GYM‑COACH – Live Analyzer")
        self.resize(1100, 720)

        # Core state
        self.state = UIState()
        self.tts = TTSManager()
        
        # Initialize engine with proper error handling
        try:
            self.engine = LiveFeedbackEngine(exercise=self.state.exercise, tts_callback=self.tts.speak)
        except TypeError:
            # Fallback for versions that don't accept tts_callback
            self.engine = LiveFeedbackEngine(exercise=self.state.exercise)
        
        # Enable TTS if possible
        try:
            self.engine.enable_tts(True)
        except Exception:
            pass

        # Capture worker
        self.worker: Optional[CaptureWorker] = None

        # Build UI
        self._build_ui()
        self._connect_signals()

    # ------------------------- UI Construction -------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Video view
        self.videoLabel = QtWidgets.QLabel()
        self.videoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.videoLabel.setMinimumSize(640, 360)
        self.videoLabel.setStyleSheet("background: #111; color: #aaa; border: 1px solid #333;")
        self.videoLabel.setText("Preview will appear here")

        # Controls
        self.btnOpenVideo = QtWidgets.QPushButton("Open Video…")
        self.btnStartCam = QtWidgets.QPushButton("Start Camera")
        self.btnStop = QtWidgets.QPushButton("Stop")
        self.btnStop.setEnabled(False)

        self.comboExercise = QtWidgets.QComboBox()
        self.comboExercise.addItems(["squat", "chest", "back", "shoulder", "biceps"])

        self.chkTTS = QtWidgets.QCheckBox("Voice feedback")
        self.chkTTS.setChecked(True)

        self.spinCamIndex = QtWidgets.QSpinBox()
        self.spinCamIndex.setRange(0, 9)
        self.spinCamIndex.setValue(0)
        self.spinCamIndex.setPrefix("Cam ")

        self.spinTargetFPS = QtWidgets.QDoubleSpinBox()
        self.spinTargetFPS.setDecimals(1)
        self.spinTargetFPS.setRange(0.0, 120.0)
        self.spinTargetFPS.setValue(0.0)
        self.spinTargetFPS.setSuffix(" fps (0=unlimited)")

        self.lblStatus = QtWidgets.QLabel("Idle")
        self.lblFPS = QtWidgets.QLabel("— fps")

        # Layout
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btnOpenVideo)
        controls.addWidget(self.btnStartCam)
        controls.addWidget(self.btnStop)
        controls.addSpacing(16)
        controls.addWidget(QtWidgets.QLabel("Exercise:"))
        controls.addWidget(self.comboExercise)
        controls.addSpacing(12)
        controls.addWidget(self.chkTTS)
        controls.addSpacing(12)
        controls.addWidget(self.spinCamIndex)
        controls.addSpacing(12)
        controls.addWidget(self.spinTargetFPS)
        controls.addStretch(1)
        controls.addWidget(self.lblFPS)

        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.videoLabel, stretch=1)
        layout.addLayout(controls)
        layout.addWidget(self.lblStatus)

    # --------------------------- Signal wiring --------------------------
    def _connect_signals(self):
        self.btnOpenVideo.clicked.connect(self._on_open_video)
        self.btnStartCam.clicked.connect(self._on_start_camera)
        self.btnStop.clicked.connect(self._on_stop)
        self.comboExercise.currentTextChanged.connect(self._on_exercise_changed)
        self.chkTTS.toggled.connect(self._on_tts_toggled)

    # ----------------------------- Handlers -----------------------------
    def _on_open_video(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select video", str(Path.cwd()),
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)"
        )
        if not file:
            return
        self.state.source_label = Path(file).name
        self._start_stream(file)

    def _on_start_camera(self):
        cam_idx = int(self.spinCamIndex.value())
        self.state.source_label = f"Camera {cam_idx}"
        self._start_stream(cam_idx)

    def _on_stop(self):
        self._stop_stream()

    def _on_exercise_changed(self, value: str):
        self.state.exercise = value
        self.engine.set_exercise(value)
        self.engine.reset_state()
        self._status(f"Exercise: {value}")

    def _on_tts_toggled(self, checked: bool):
        self.state.tts_enabled = checked
        self.engine.enable_tts(checked)
        if not checked:
            try:
                self.tts.stop()
            except Exception:
                pass

    # ----------------------------- Streaming ----------------------------
    def _start_stream(self, source: int | str):
        self._stop_stream()
        self.worker = CaptureWorker(source)
        target_fps = float(self.spinTargetFPS.value())
        self.worker.set_target_fps(None if target_fps <= 0.0 else target_fps)

        self.worker.frameReady.connect(self._on_frame)
        self.worker.streamEnded.connect(self._on_stream_end)
        self.worker.fps.connect(self._on_fps)

        self.btnOpenVideo.setEnabled(False)
        self.btnStartCam.setEnabled(False)
        self.btnStop.setEnabled(True)
        self._status(f"Streaming: {self.state.source_label}")

        self.worker.start()

    def _stop_stream(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1000)
        self.worker = None
        self.btnOpenVideo.setEnabled(True)
        self.btnStartCam.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.lblFPS.setText("— fps")
        self._status("Idle")

    # ----------------------------- Frame path ---------------------------
    @QtCore.Slot(object)
    def _on_frame(self, frame_bgr):
        try:
            # Ensure frame is valid
            if frame_bgr is None or not isinstance(frame_bgr, np.ndarray):
                return
                
            vis_bgr, messages = self.engine.process(frame_bgr)
            
            # Ensure processed frame is valid
            if vis_bgr is None or not isinstance(vis_bgr, np.ndarray):
                vis_bgr = frame_bgr
                
        except Exception as e:
            # If analyzer crashes, show raw frame and surface error in status.
            vis_bgr, messages = frame_bgr, [f"Analyzer error: {e}"]

        # Show messages as transient status
        if messages:
            self._status(" | ".join(messages))

        # Convert BGR -> RGB -> QImage with proper memory handling
        try:
            rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            
            # Create QImage with copy of data to avoid memory issues
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
            
            pixmap = QtGui.QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                self.videoLabel.width(), 
                self.videoLabel.height(), 
                QtCore.Qt.KeepAspectRatio, 
                QtCore.Qt.SmoothTransformation
            )
            self.videoLabel.setPixmap(scaled_pixmap)
        except Exception as e:
            self._status(f"Display error: {e}")

    @QtCore.Slot()
    def _on_stream_end(self):
        self._status("Stream ended")
        self._stop_stream()

    @QtCore.Slot(float)
    def _on_fps(self, value: float):
        self.lblFPS.setText(f"{value:.1f} fps")

    def _status(self, text: str):
        self.lblStatus.setText(text)

    # ------------------------------ Cleanup -----------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            self._stop_stream()
            self.tts.stop()
        except Exception:
            pass
        super().closeEvent(event)


# --------------------------------- Main ---------------------------------
def main():
    # High‑DPI friendly rendering - MUST be set BEFORE QApplication creation
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)
    
    app = QtWidgets.QApplication(sys.argv)
    
    QtCore.QCoreApplication.setOrganizationName("AI‑GYM‑COACH")
    QtCore.QCoreApplication.setApplicationName("Live Analyzer")

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()