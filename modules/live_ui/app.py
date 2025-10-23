#!/usr/bin/env python3
# modules/live_ui/app.py
"""
PySide6 GUI: Home → Squat screen.
- Choose Camera or Video
- Start/Stop live analysis (FSM + GRU inference + optional TTS)
- Ready to extend to other exercises by wiring new screens with same workers.
"""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QFormLayout, QComboBox, QSpinBox, QCheckBox, QSizePolicy,
    QStackedWidget, QStatusBar
)

from modules.live_ui.workers import VideoWorker, AnalyzerWorker

def cvimg_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(960, 540)
        self.setStyleSheet(
            "background: #111926; color: #90a0b0; border: 2px solid #334155; border-radius: 12px; font-size: 16px; font-weight: bold;"
        )
        self.setText("Select camera or video, then press Start")

    def set_frame(self, frame: np.ndarray):
        pix = cvimg_to_qpixmap(frame)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        if self.pixmap():
            self.setPixmap(self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(ev)

class SquatScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.worker_cam: Optional[VideoWorker] = None
        self.worker_an: Optional[AnalyzerWorker] = None
        self.current_source: Optional[Union[int, str]] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(12,12,12,12)
        root.setSpacing(12)

        # Left controls
        left = QGroupBox("Controls")
        left.setMaximumWidth(320)
        l = QVBoxLayout(left)

        form = QFormLayout()
        self.cam_combo = QComboBox()
        self._populate_cameras()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)

        self.flip_check = QCheckBox("Flip camera")
        self.flip_check.setChecked(True)

        self.tts_check = QCheckBox("Voice feedback")
        self.tts_check.setChecked(True)

        form.addRow("Camera:", self.cam_combo)
        form.addRow("Target FPS:", self.fps_spin)
        form.addRow("", self.flip_check)
        form.addRow("", self.tts_check)
        l.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_cam = QPushButton("Open Camera")
        self.btn_cam.clicked.connect(self._select_first_camera)
        self.btn_vid = QPushButton("Open Video")
        self.btn_vid.clicked.connect(self._open_video)
        btn_row.addWidget(self.btn_cam)
        btn_row.addWidget(self.btn_vid)
        l.addLayout(btn_row)

        self.btn_start = QPushButton("Start")
        self.btn_start.clicked.connect(self.handle_start)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.handle_stop)

        l.addWidget(self.btn_start)
        l.addWidget(self.btn_stop)
        l.addStretch()

        # Center video
        center_box = QVBoxLayout()
        self.video_label = VideoLabel()
        center_box.addWidget(self.video_label, 1)

        # Right metrics
        right = QGroupBox("Metrics")
        right.setMaximumWidth(300)
        r = QVBoxLayout(right)

        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_last = QLabel("Last: -")
        self.lbl_status = QLabel("Status: Waiting")
        for lab in [self.lbl_fps, self.lbl_last, self.lbl_status]:
            lab.setStyleSheet("color: #cbd5e1; font-size: 14px;")
        r.addWidget(self.lbl_fps)
        r.addWidget(self.lbl_last)
        r.addWidget(self.lbl_status)
        r.addStretch()

        root.addWidget(left, 0)
        root.addLayout(center_box, 1)
        root.addWidget(right, 0)

        # periodic UI update
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._tick)
        self._ui_timer.start(500)

    def _populate_cameras(self, max_index: int = 4):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append(i)
        if found:
            for i in found:
                self.cam_combo.addItem(f"Camera {i}", i)
        else:
            self.cam_combo.addItem("No camera", None)
        self.cam_combo.blockSignals(False)

    @Slot()
    def _select_first_camera(self):
        if self.cam_combo.count() > 0 and self.cam_combo.itemData(0) is not None:
            self.current_source = int(self.cam_combo.itemData(0))
        else:
            self.current_source = 0

    @Slot()
    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", str(Path.home()), "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.current_source = path

    @Slot()
    def handle_start(self):
        if self.worker_cam and self.worker_cam.isRunning():
            return
        if self.current_source is None:
            self._select_first_camera()  # best effort
        if self.current_source is None:
            self.lbl_status.setText("Status: No camera")
            return

        # start analyzer
        self.worker_an = AnalyzerWorker(exercise="squat",
                                        speak_out=self.tts_check.isChecked(),
                                        yolo_weights=None)  # set path if you want YOLO bar class
        self.worker_an.overlay_ready.connect(self._on_overlay)
        self.worker_an.rep_event.connect(self._on_rep)
        self.worker_an.status.connect(lambda s: self.lbl_status.setText(f"Status: {s}"))
        self.worker_an.error.connect(lambda e: self.lbl_status.setText(f"Error: {e}"))
        self.worker_an.start()

        # start video worker
        self.worker_cam = VideoWorker(self.current_source,
                                      target_fps=float(self.fps_spin.value()),
                                      flip=self.flip_check.isChecked())
        self.worker_cam.frame_ready.connect(self._on_frame)
        self.worker_cam.stats_ready.connect(lambda d: self.lbl_fps.setText(f"FPS: {d.get('fps',0.0):.1f}"))
        self.worker_cam.error.connect(lambda e: self.lbl_status.setText(f"Error: {e}"))
        self.worker_cam.start()
        self.lbl_status.setText("Status: Running")

    @Slot()
    def handle_stop(self):
        if self.worker_cam and self.worker_cam.isRunning():
            self.worker_cam.stop()
            self.worker_cam.wait(1000)
        if self.worker_an and self.worker_an.isRunning():
            self.worker_an.stop()
            self.worker_an.wait(1000)
        self.lbl_status.setText("Status: Stopped")

    @Slot(np.ndarray)
    def _on_frame(self, frame: np.ndarray):
        self.video_label.set_frame(frame)
        if self.worker_an and self.worker_an.isRunning():
            self.worker_an.push_frame(frame)

    @Slot(np.ndarray)
    def _on_overlay(self, frame: np.ndarray):
        # if you prefer overlay instead raw: display overlay
        self.video_label.set_frame(frame)

    @Slot(dict)
    def _on_rep(self, info: dict):
        rep = info.get("rep")
        pred = info.get("pred")
        prob = info.get("prob")
        reason = info.get("reason", "")
        txt = f"Rep {rep}: {pred} (p={prob:.2f})"
        if reason:
            txt += f" – {reason}"
        self.lbl_last.setText(f"Last: {txt}")

    def cleanup(self):
        self.handle_stop()

    def _tick(self):
        pass  # reserved for UI updates

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gym Coach - Live Analyzer")
        self.resize(1400, 800)
        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)
        self.status = QStatusBar(self)
        self.setStatusBar(self.status)

        # Home
        home = QWidget()
        v = QVBoxLayout(home)
        title = QLabel("Choose Exercise")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #e2e8f0;")
        v.addWidget(title)

        row = QHBoxLayout()
        btn_squat = QPushButton("Squat")
        btn_squat.setMinimumSize(220, 120)
        btn_squat.setStyleSheet("font-size: 20px; font-weight: bold;")
        btn_squat.clicked.connect(self.open_squat)
        row.addStretch()
        row.addWidget(btn_squat)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()

        self.home = home
        self.squat_screen = SquatScreen()

        self.stacked.addWidget(self.home)
        self.stacked.addWidget(self.squat_screen)
        self.status.showMessage("Ready")

    @Slot()
    def open_squat(self):
        self.stacked.setCurrentWidget(self.squat_screen)
        self.status.showMessage("Squat")

    def closeEvent(self, ev):
        try:
            self.squat_screen.cleanup()
        except Exception:
            pass
        super().closeEvent(ev)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
