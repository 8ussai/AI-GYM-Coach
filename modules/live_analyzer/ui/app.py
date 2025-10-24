#!/usr/bin/env python3
"""
Gym Coach App - Multi-Exercise Trainer (Live RTL/LTR Localization — Integrated)

This app.py wires your UI with the live analyzer and TTS we agreed on:
  • Uses VideoWorker for camera/file input
  • Pipes frames through LiveFeedbackEngine (MediaPipe-based fallback-safe)
  • Updates on-screen metrics (FPS/Reps/Angle/Status)
  • Optional TTS via TTSManager; tolerant if engine signature differs

Run from project root:
    python -m modules.live_analyzer.ui.app

Folder layout expected:
modules/
  live_analyzer/
    ui/app.py                    ← this file
    live_feedback.py             ← provided earlier
    tts_utils.py                 ← provided earlier
    __init__.py
  __init__.py

Notes
- Reads AppSettings from this file for flip/show_landmarks/target_fps.
- Status and metrics are refreshed live; engine messages appear in status.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Union
from enum import Enum

import numpy as np
import cv2

from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer, QObject
from PySide6.QtGui import QAction, QIcon, QKeySequence, QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QToolBar, QStatusBar, QFormLayout, QGroupBox, QSpinBox,
    QDoubleSpinBox, QComboBox, QFrame, QStackedWidget,
    QCheckBox, QSizePolicy
)

# ----------------- Analyzer + TTS imports with graceful fallback -----------------
try:
    from modules.live_analyzer.live_feedback import LiveFeedbackEngine  # type: ignore
except Exception:  # pragma: no cover
    class LiveFeedbackEngine:  # minimal stub
        def __init__(self, exercise: str = "squat", *args, **kwargs):
            self.exercise = exercise
            self.fsm = type("_FSM", (), {"reps": 0, "state": "idle"})()
            self.knee_ema = type("_EMA", (), {"val": None})()
        def set_exercise(self, e: str):
            self.exercise = e
        def enable_tts(self, _e: bool):
            pass
        def reset_state(self):
            self.fsm.reps = 0
            self.knee_ema.val = None
        def process(self, frame_bgr):
            return frame_bgr, []

try:
    from modules.live_analyzer.tts_utils import TTSManager  # type: ignore
except Exception:  # pragma: no cover
    class TTSManager:
        def __init__(self, *args, **kwargs):
            pass
        def speak(self, text: str):
            print(f"[TTS] {text}")
        def stop(self):
            pass

# ---------- Language & Events ----------

class Language(Enum):
    ENGLISH = "en"
    ARABIC = "ar"

class LangBus(QObject):
    changed = Signal()

lang_bus = LangBus()

class Translations:
    TEXTS = {
        Language.ENGLISH: {
            # App
            "app_title": "Gym Coach - Exercise Tracker",
            "home": "Home",
            "settings": "Settings",
            "language_label": "Language:",
            "english": "English",
            "arabic": "العربية",
            # Home
            "choose_exercise": "Choose Your Exercise",
            "squats": "Squats",
            "squats_desc": "Lower body strength training",
            "dumbbell": "Dumbbell Exercise",
            "dumbbell_desc": "Upper body with dumbbells",
            "barbell": "Barbell Exercise",
            "barbell_desc": "Compound movements with barbell",
            # Screens
            "controls": "Controls",
            "metrics": "Metrics",
            "camera": "Camera",
            "open_camera": "Open Camera",
            "open_video": "Open Video",
            "back_home": "Back to Home",
            "start": "Start",
            "start_workout": "Start Workout",
            "stop_workout": "Stop Workout",
            # Settings
            "camera_settings": "Camera Settings",
            "flip_camera": "Flip Camera Horizontally",
            "show_landmarks": "Show Pose Landmarks",
            "confidence": "Detection Confidence",
            "target_fps": "Target FPS",
            # Metrics
            "reps": "Reps",
            "angle": "Angle",
            "status": "Status",
            "speed": "Speed",
            "frame_sec": "frames/sec",
            "degree": "degrees",
            # Status text
            "waiting": "Waiting",
            "running": "Running",
            "stopped": "Stopped",
            "no_camera": "No camera found",
            # Hints
            "select_source_hint": "Select camera or video to begin",
            # Toolbar
            "title_brand": "🏋️ Gym Coach",
            "ready": "Ready",
        },
        Language.ARABIC: {
            # App
            "app_title": "مُدرب جيم - تتبّع التمارين",
            "home": "الرئيسية",
            "settings": "الإعدادات",
            "language_label": "اللغة:",
            "english": "English",
            "arabic": "العربية",
            # Home
            "choose_exercise": "اختر تمرينك",
            "squats": "سكوات",
            "squats_desc": "تقوية الجزء السفلي من الجسم",
            "dumbbell": "تمرين دمبلز",
            "dumbbell_desc": "تمارين الجزء العلوي بالدمبلز",
            "barbell": "تمرين بار",
            "barbell_desc": "تمارين مركبة بالبار",
            # Screens
            "controls": "التحكم",
            "metrics": "القياسات",
            "camera": "كاميرا",
            "open_camera": "فتح الكاميرا",
            "open_video": "فتح فيديو",
            "back_home": "العودة للرئيسية",
            "start": "ابدأ",
            "start_workout": "ابدأ التمرين",
            "stop_workout": "أوقف التمرين",
            # Settings
            "camera_settings": "إعدادات الكاميرا",
            "flip_camera": "قلب الكاميرا أفقيًا",
            "show_landmarks": "إظهار نقاط الجسم",
            "confidence": "دقة الكشف",
            "target_fps": "سرعة العرض",
            # Metrics
            "reps": "التكرارات",
            "angle": "الزاوية",
            "status": "الحالة",
            "speed": "السرعة",
            "frame_sec": "إطار/ثانية",
            "degree": "درجة",
            # Status text
            "waiting": "في الانتظار",
            "running": "جاري التشغيل",
            "stopped": "متوقف",
            "no_camera": "لا توجد كاميرا",
            # Hints
            "select_source_hint": "اختر كاميرا أو فيديو للبدء",
            # Toolbar
            "title_brand": "🏋️ مدرب الجيم",
            "ready": "جاهز",
        }
    }

    current_lang = Language.ENGLISH

    @classmethod
    def get(cls, key: str) -> str:
        return cls.TEXTS[cls.current_lang].get(key, key)

    @classmethod
    def set_language(cls, lang: Language):
        cls.current_lang = lang
        lang_bus.changed.emit()  # notify listeners

# ---------- Settings Manager ----------

class AppSettings:
    flip_camera = True
    show_landmarks = True  # If False, raw frame is shown, still runs analyzer for metrics
    confidence = 0.5
    target_fps = 30

# ---------- Image Conversion ----------

def cvimg_to_qpixmap(frame: np.ndarray) -> QPixmap:
    if frame is None:
        return QPixmap()
    if len(frame.shape) == 2:
        h, w = frame.shape
        qimg = QImage(frame.data, w, h, w, QImage.Format_Grayscale8)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

# ---------- Video Worker ----------

class VideoWorker(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, source: Union[int, str] = 0, parent=None):
        super().__init__(parent)
        self.source = source
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_t = None

    def open_source(self) -> bool:
        if isinstance(self.source, int):
            if sys.platform.startswith("win"):
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            elif sys.platform == "darwin":
                self._cap = cv2.VideoCapture(self.source, cv2.CAP_AVFOUNDATION)
            else:
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)
        if not self._cap or not self._cap.isOpened():
            self.error.emit(f"Failed to open source: {self.source}")
            return False
        return True

    def close_source(self):
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        self._last_t = time.time()

        if not self.open_source():
            return

        interval = 1.0 / max(1e-3, AppSettings.target_fps)
        frames = 0
        t0 = time.time()

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                self.error.emit("Stream ended")
                break

            if AppSettings.flip_camera and isinstance(self.source, int):
                frame = cv2.flip(frame, 1)

            self.frame_ready.emit(frame)
            frames += 1

            dt_total = time.time() - t0
            fps = frames / dt_total if dt_total > 0 else 0.0
            self.stats_ready.emit({"fps": fps})

            t_elapsed = time.time() - self._last_t
            sleep_time = interval - t_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_t = time.time()

        self.close_source()

# ---------- Video Display ----------

class VideoLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setText(Translations.get("select_source_hint"))
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(720, 480)
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.setFont(font)
        self.setStyleSheet(
            """
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1d29, stop:1 #2d3142);
            color: #8892b0; border: 3px solid #3d4663; border-radius: 15px;
            """
        )
        lang_bus.changed.connect(self.retranslate)

    @Slot()
    def retranslate(self):
        if not self.pixmap():
            self.setText(Translations.get("select_source_hint"))

    def set_frame(self, frame: np.ndarray):
        pix = cvimg_to_qpixmap(frame)
        self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        if self.pixmap():
            self.setPixmap(self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

# ---------- Exercise Card ----------

class ExerciseCard(QFrame):
    clicked = Signal(str)

    def __init__(self, title_key: str, desc_key: str, icon: str, exercise_type: str):
        super().__init__()
        self.exercise_type = exercise_type
        self.title_key = title_key
        self.desc_key = desc_key
        self.setMinimumSize(280, 200)
        self.setCursor(Qt.PointingHandCursor)

        self.layout_ = QVBoxLayout(self)
        self.layout_.setSpacing(15)
        self.layout_.setContentsMargins(25, 25, 25, 25)

        self.icon_label = QLabel(icon)
        self.icon_label.setStyleSheet("font-size: 64px;")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.layout_.addWidget(self.icon_label)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #64ffda;")
        self.layout_.addWidget(self.title_label)

        self.desc_label = QLabel()
        self.desc_label.setAlignment(Qt.AlignCenter)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("font-size: 14px; color: #8892b0;")
        self.layout_.addWidget(self.desc_label)

        self.layout_.addStretch()

        self.btn = QPushButton()
        self.btn.setMinimumHeight(45)
        self.btn.setStyleSheet(
            """
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #27ae60);
                          color: white; font-size: 16px; font-weight: bold; border-radius: 10px; border: none; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #229954); }
            """
        )
        self.btn.clicked.connect(lambda: self.clicked.emit(self.exercise_type))
        self.layout_.addWidget(self.btn)

        self.setStyleSheet(
            """
            QFrame { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2d3142, stop:1 #1a1d29);
                    border: 2px solid #3d4663; border-radius: 15px; }
            QFrame:hover { border: 2px solid #64ffda; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3d4663, stop:1 #2d3142); }
            """
        )

        lang_bus.changed.connect(self.retranslate)
        self.retranslate()

    @Slot()
    def retranslate(self):
        self.title_label.setText(Translations.get(self.title_key))
        self.desc_label.setText(Translations.get(self.desc_key))
        self.btn.setText(Translations.get("start"))

# ---------- Home Screen ----------

class HomeScreen(QWidget):
    exercise_selected = Signal(str)
    settings_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.layout_ = QVBoxLayout(self)
        self.layout_.setSpacing(30)
        self.layout_.setContentsMargins(50, 50, 50, 50)

        self.title = QLabel()
        self.title.setStyleSheet("font-size: 36px; font-weight: bold; color: #ccd6f6;")
        self.title.setAlignment(Qt.AlignCenter)
        self.layout_.addWidget(self.title)

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(30)

        self.squat_card = ExerciseCard("squats", "squats_desc", "🦵", "squats")
        self.squat_card.clicked.connect(self.exercise_selected.emit)
        cards_layout.addWidget(self.squat_card)

        self.dumbbell_card = ExerciseCard("dumbbell", "dumbbell_desc", "🏋️", "dumbbell")
        self.dumbbell_card.clicked.connect(self.exercise_selected.emit)
        cards_layout.addWidget(self.dumbbell_card)

        self.barbell_card = ExerciseCard("barbell", "barbell_desc", "💪", "barbell")
        self.barbell_card.clicked.connect(self.exercise_selected.emit)
        cards_layout.addWidget(self.barbell_card)

        self.layout_.addLayout(cards_layout)
        self.layout_.addStretch()

        self.settings_btn = QPushButton()
        self.settings_btn.setMinimumHeight(50)
        self.settings_btn.setMaximumWidth(250)
        self.settings_btn.setStyleSheet(
            """
            QPushButton { background: #3d4663; border: 2px solid #4d5673; color: #ccd6f6; padding: 10px 20px; border-radius: 12px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #4d5673; border: 2px solid #64ffda; }
            """
        )
        self.settings_btn.clicked.connect(self.settings_clicked.emit)

        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(self.settings_btn)
        btn_container.addStretch()
        self.layout_.addLayout(btn_container)

        lang_bus.changed.connect(self.retranslate)
        self.retranslate()

    @Slot()
    def retranslate(self):
        self.title.setText(Translations.get("choose_exercise"))
        self.settings_btn.setText(f"⚙️ {Translations.get('settings')}")

# ---------- Settings Screen ----------

class SettingsScreen(QWidget):
    back_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.layout_ = QVBoxLayout(self)
        self.layout_.setSpacing(20)
        self.layout_.setContentsMargins(50, 50, 50, 50)

        self.title = QLabel()
        self.title.setStyleSheet("font-size: 32px; font-weight: bold; color: #ccd6f6;")
        self.layout_.addWidget(self.title)

        self.settings_group = QGroupBox()
        self.settings_layout = QFormLayout(self.settings_group)
        self.settings_layout.setSpacing(20)

        # store explicit label widgets to avoid None from labelForField
        self.flip_label = QLabel()
        self.flip_check = QCheckBox()
        self.flip_check.setChecked(AppSettings.flip_camera)
        self.flip_check.stateChanged.connect(self.on_flip_changed)
        self.settings_layout.addRow(self.flip_label, self.flip_check)

        self.landmarks_label = QLabel()
        self.landmarks_check = QCheckBox()
        self.landmarks_check.setChecked(AppSettings.show_landmarks)
        self.landmarks_check.stateChanged.connect(self.on_landmarks_changed)
        self.settings_layout.addRow(self.landmarks_label, self.landmarks_check)

        self.conf_label = QLabel()
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(AppSettings.confidence)
        self.conf_spin.setMinimumHeight(35)
        self.conf_spin.valueChanged.connect(self.on_confidence_changed)
        self.settings_layout.addRow(self.conf_label, self.conf_spin)

        self.fps_label = QLabel()
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(AppSettings.target_fps)
        self.fps_spin.setMinimumHeight(35)
        self.fps_spin.valueChanged.connect(self.on_fps_changed)
        self.settings_layout.addRow(self.fps_label, self.fps_spin)

        self.layout_.addWidget(self.settings_group)
        self.layout_.addStretch()

        self.back_btn = QPushButton()
        self.back_btn.setMinimumHeight(50)
        self.back_btn.setMaximumWidth(250)
        self.back_btn.setStyleSheet(
            """
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2980b9); color: white; font-size: 16px; font-weight: bold; border-radius: 10px; border: none; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2980b9, stop:1 #21618c); }
            """
        )
        self.back_btn.clicked.connect(self.back_clicked.emit)

        btn_container = QHBoxLayout()
        btn_container.addStretch()
        btn_container.addWidget(self.back_btn)
        btn_container.addStretch()
        self.layout_.addLayout(btn_container)

        lang_bus.changed.connect(self.retranslate)
        self.retranslate()

    @Slot()
    def retranslate(self):
        self.title.setText(f"⚙️ {Translations.get('settings')}")
        self.settings_group.setTitle(Translations.get("camera_settings"))
        self.flip_label.setText(Translations.get("flip_camera"))
        self.landmarks_label.setText(Translations.get("show_landmarks"))
        self.conf_label.setText(Translations.get("confidence"))
        self.fps_label.setText(Translations.get("target_fps"))
        self.back_btn.setText(f"← {Translations.get('back_home')}")

    @Slot(int)
    def on_flip_changed(self, state):
        AppSettings.flip_camera = bool(state)

    @Slot(int)
    def on_landmarks_changed(self, state):
        AppSettings.show_landmarks = bool(state)

    @Slot(float)
    def on_confidence_changed(self, value):
        AppSettings.confidence = value

    @Slot(int)
    def on_fps_changed(self, value):
        AppSettings.target_fps = value

# ---------- Exercise Screen ----------

class ExerciseScreen(QWidget):
    back_clicked = Signal()

    def __init__(self, exercise_type: str):
        super().__init__()
        self.exercise_type = exercise_type
        self.worker: Optional[VideoWorker] = None
        self.current_source: Union[int, str, None] = None
        self.ui_fps = 0.0

        # Analyzer + TTS
        self.tts = TTSManager()
        try:
            self.engine = LiveFeedbackEngine(exercise="squat", tts_callback=self.tts.speak)
        except TypeError:
            self.engine = LiveFeedbackEngine(exercise="squat")
            try:
                self.engine.enable_tts(True)
            except Exception:
                pass

        main_h = QHBoxLayout(self)
        main_h.setSpacing(15)
        main_h.setContentsMargins(15, 15, 15, 15)

        self.left_controls = self._build_left_controls()
        main_h.addWidget(self.left_controls, 0)

        cbox = QVBoxLayout()
        cbox.setSpacing(10)
        self.video_label = VideoLabel()
        cbox.addWidget(self.video_label, 1)
        main_h.addLayout(cbox, 1)

        self.right_metrics = self._build_right_metrics()
        main_h.addWidget(self.right_metrics, 0)

        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_metrics)
        self._fps_timer.start(500)

        QTimer.singleShot(300, self._auto_start)
        lang_bus.changed.connect(self.retranslate)
        self.retranslate()

    def _build_left_controls(self) -> QWidget:
        self.controls_group = QGroupBox()
        self.controls_group.setMaximumWidth(300)
        lay = QVBoxLayout(self.controls_group)
        lay.setSpacing(12)

        self.title = QLabel(self.exercise_type.upper())
        self.title.setStyleSheet("font-size: 20px; font-weight: bold; color: #64ffda;")
        self.title.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.title)

        form = QFormLayout()
        self.cam_label = QLabel()
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumHeight(35)
        form.addRow(self.cam_label, self.cam_combo)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_cam = QPushButton()
        self.btn_cam.clicked.connect(self._select_first_camera)
        self.btn_vid = QPushButton()
        self.btn_vid.clicked.connect(self._open_video)
        btn_row.addWidget(self.btn_cam)
        btn_row.addWidget(self.btn_vid)
        lay.addLayout(btn_row)

        lay.addStretch()

        self.btn_start = QPushButton()
        self.btn_start.setMinimumHeight(48)
        self.btn_start.setStyleSheet(
            """
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #27ae60); color: white; font-size: 14px; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #27ae60, stop:1 #229954); }
            """
        )

        self.btn_stop = QPushButton()
        self.btn_stop.setMinimumHeight(48)
        self.btn_stop.setStyleSheet(
            """
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #e74c3c, stop:1 #c0392b); color: white; font-size: 14px; font-weight: bold; border-radius: 10px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c0392b, stop:1 #a93226); }
            """
        )

        self.btn_back = QPushButton()
        self.btn_back.setMinimumHeight(42)

        lay.addWidget(self.btn_start)
        lay.addWidget(self.btn_stop)
        lay.addWidget(self.btn_back)

        self.btn_start.clicked.connect(self.handle_start)
        self.btn_stop.clicked.connect(self.handle_stop)
        self.btn_back.clicked.connect(self.on_back_clicked)

        self._populate_cameras()

        return self.controls_group

    def _build_right_metrics(self) -> QWidget:
        self.metrics_group = QGroupBox()
        self.metrics_group.setMaximumWidth(300)
        lay = QVBoxLayout(self.metrics_group)
        lay.setSpacing(16)

        self.speed_card, self.lbl_fps = self._create_metric_card("speed", "frame_sec", init_value="0")
        lay.addWidget(self.speed_card)

        self.reps_card, self.lbl_reps = self._create_metric_card("reps", unit_text="reps", init_value="0")
        lay.addWidget(self.reps_card)

        self.angle_card, self.lbl_angle = self._create_metric_card("angle", "degree", init_value="-")
        lay.addWidget(self.angle_card)

        self.status_card, self.lbl_status = self._create_status_card()
        lay.addWidget(self.status_card)

        lay.addStretch()
        return self.metrics_group

    def _create_metric_card(self, title_key, unit_key=None, unit_text=None, init_value="0"):
        card = QFrame()
        card.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d3142, stop:1 #3d4663); border-radius: 12px; padding: 10px; }")
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(5)

        title_lbl = QLabel()
        title_lbl.setStyleSheet("color: #8892b0; font-size: 12px;")
        card_lay.addWidget(title_lbl)

        value_lbl = QLabel(init_value)
        value_lbl.setStyleSheet("color: #64ffda; font-size: 28px; font-weight: bold;")
        value_lbl.setAlignment(Qt.AlignCenter)
        card_lay.addWidget(value_lbl)

        unit_lbl = QLabel()
        unit_lbl.setStyleSheet("color: #8892b0; font-size: 10px;")
        unit_lbl.setAlignment(Qt.AlignCenter)
        card_lay.addWidget(unit_lbl)

        def update_labels():
            title_lbl.setText(Translations.get(title_key))
            if unit_text is not None:
                unit_lbl.setText(unit_text)
            else:
                unit_lbl.setText(Translations.get(unit_key) if unit_key else "")
        lang_bus.changed.connect(update_labels)
        update_labels()

        return card, value_lbl

    def _create_status_card(self):
        card = QFrame()
        card.setStyleSheet("QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2d3142, stop:1 #3d4663); border-radius: 12px; padding: 15px; }")
        card_lay = QVBoxLayout(card)
        card_lay.setSpacing(8)

        title = QLabel()
        title.setStyleSheet("color: #8892b0; font-size: 12px;")
        card_lay.addWidget(title)

        status = QLabel()
        status.setStyleSheet("color: #ffd700; font-size: 16px; font-weight: bold;")
        status.setAlignment(Qt.AlignCenter)
        card_lay.addWidget(status)

        def update_labels():
            title.setText(Translations.get("status"))
            status.setText(Translations.get("waiting"))
        lang_bus.changed.connect(update_labels)
        update_labels()

        return card, status

    def _populate_cameras(self, max_index: int = 4):
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        found = []
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
            ok, _ = cap.read()
            cap.release()
            if ok:
                found.append(i)
        if found:
            for i in found:
                self.cam_combo.addItem(f"{Translations.get('camera')} {i}", i)
        else:
            self.cam_combo.addItem(Translations.get("no_camera"), None)
        self.cam_combo.blockSignals(False)

    def _auto_start(self):
        if self.cam_combo.count() > 0:
            data = self.cam_combo.itemData(0)
            if data is not None and isinstance(data, int):
                self.current_source = int(data)

    @Slot()
    def handle_start(self):
        if self.worker and self.worker.isRunning():
            return
        if self.current_source is None:
            if self.cam_combo.count() > 0:
                data = self.cam_combo.itemData(0)
                if data is not None and isinstance(data, int):
                    self.current_source = int(data)
        if self.current_source is None:
            self.lbl_status.setText(Translations.get("no_camera"))
            self.lbl_status.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
            return
        self.worker = VideoWorker(self.current_source)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats_ready.connect(self.on_stats)
        self.worker.error.connect(self.on_error)
        self.worker.start()
        self.lbl_status.setText(Translations.get("running"))
        self.lbl_status.setStyleSheet("color: #2ecc71; font-size: 16px; font-weight: bold;")

    @Slot()
    def handle_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        self.lbl_status.setText(Translations.get("stopped"))
        self.lbl_status.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")

    @Slot(np.ndarray)
    def on_frame(self, frame: np.ndarray):
        # Analyze frame
        vis_frame, messages = self.engine.process(frame)

        # Choose which to display based on settings
        display_frame = vis_frame if AppSettings.show_landmarks else frame
        self.video_label.set_frame(display_frame)

        # Update metrics: FPS handled elsewhere, reps/angle from engine internals
        try:
            reps = int(getattr(self.engine.fsm, "reps", 0))
            angle_val = getattr(self.engine.knee_ema, "val", None)
            self.lbl_reps.setText(str(reps))
            self.lbl_angle.setText("-" if angle_val is None else f"{angle_val:.1f}")
        except Exception:
            pass

        # Surface messages in status card and optionally TTS is already handled by engine
        if messages:
            self.lbl_status.setText(" | ".join(messages[-2:]))

    @Slot(dict)
    def on_stats(self, stats: dict):
        self.ui_fps = float(stats.get("fps", 0.0))

    @Slot(str)
    def on_error(self, msg: str):
        self.lbl_status.setText(msg)
        self.lbl_status.setStyleSheet("color: #e67e22; font-size: 14px; font-weight: bold;")

    def _update_metrics(self):
        self.lbl_fps.setText(f"{self.ui_fps:.1f}")

    @Slot()
    def on_back_clicked(self):
        self.handle_stop()
        self.back_clicked.emit()

    def cleanup(self):
        self.handle_stop()
        try:
            self.tts.stop()
        except Exception:
            pass

    @Slot()
    def retranslate(self):
        self.controls_group.setTitle(Translations.get("controls"))
        self.metrics_group.setTitle(Translations.get("metrics"))
        self.cam_label.setText(Translations.get("camera"))
        self.btn_cam.setText(f"📷 {Translations.get('open_camera')}")
        self.btn_vid.setText(f"🎞️ {Translations.get('open_video')}")
        self.btn_start.setText(f"▶️ {Translations.get('start_workout')}")
        self.btn_stop.setText(f"⏹️ {Translations.get('stop_workout')}")
        self.btn_back.setText(f"← {Translations.get('back_home')}")
        # refresh camera items text
        self._populate_cameras()

    def _select_first_camera(self):
        if self.cam_combo.count() > 0 and self.cam_combo.itemData(0) is not None:
            self.current_source = int(self.cam_combo.itemData(0))
        else:
            self.current_source = 0

    def _open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, Translations.get("open_video"), str(Path.home()), "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if path:
            self.current_source = path

# ---------- Main Window ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400, 800)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.home_screen = HomeScreen()
        self.settings_screen = SettingsScreen()

        self.stacked_widget.addWidget(self.home_screen)
        self.stacked_widget.addWidget(self.settings_screen)

        self.home_screen.exercise_selected.connect(self.open_exercise)
        self.home_screen.settings_clicked.connect(self.open_settings)
        self.settings_screen.back_clicked.connect(self.go_home)

        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))

        self._apply_style()

        self.exercise_screens = {}

        lang_bus.changed.connect(self.retranslate_all)
        self.retranslate_all()

    def _build_toolbar(self):
        self.tb = QToolBar("Main")
        self.tb.setIconSize(QSize(20, 20))
        self.tb.setMovable(False)

        self.title_label = QLabel()
        self.title_label.setStyleSheet("color: #64ffda; font-size: 16px; font-weight: bold; padding: 0 15px;")
        self.tb.addWidget(self.title_label)

        self.tb.addSeparator()

        self.act_home = QAction(self)
        self.act_home.triggered.connect(self.go_home)
        self.tb.addAction(self.act_home)

        self.act_settings = QAction(self)
        self.act_settings.triggered.connect(self.open_settings)
        self.tb.addAction(self.act_settings)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.tb.addWidget(spacer)

        self.lang_label = QLabel()
        self.lang_label.setStyleSheet("color: #ccd6f6; padding: 0 10px;")
        self.tb.addWidget(self.lang_label)

        self.lang_combo = QComboBox()
        # add items later in retranslate_all
        self.lang_combo.setMinimumWidth(120)
        self.lang_combo.currentIndexChanged.connect(self.change_language)
        self.tb.addWidget(self.lang_combo)

        self.addToolBar(self.tb)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f111a, stop:1 #1a1d29); }
            QGroupBox { border: 2px solid #3d4663; border-radius: 12px; margin-top: 15px; padding-top: 20px; color: #ccd6f6; font-weight: bold; font-size: 14px; background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a1d29, stop:1 #2d3142); }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 5px 10px; background: #3d4663; border-radius: 6px; }
            QLabel { color: #ccd6f6; font-size: 13px; }
            QComboBox, QDoubleSpinBox, QSpinBox { background: #2d3142; color: #ccd6f6; border: 2px solid #3d4663; border-radius: 8px; padding: 6px 10px; font-size: 13px; }
            QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover { border: 2px solid #64ffda; }
            QCheckBox { color: #ccd6f6; spacing: 10px; }
            QCheckBox::indicator { width: 20px; height: 20px; border: 2px solid #3d4663; border-radius: 4px; background: #2d3142; }
            QCheckBox::indicator:checked { background: #2ecc71; border: 2px solid #27ae60; }
            QPushButton { background: #3d4663; border: 2px solid #4d5673; color: #ccd6f6; padding: 8px 15px; border-radius: 10px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background: #4d5673; border: 2px solid #64ffda; }
            QPushButton:pressed { background: #2d3142; }
            QToolBar { background: #1a1d29; border-bottom: 2px solid #3d4663; spacing: 8px; padding: 8px; }
            QToolBar QToolButton { background: #2d3142; border: 2px solid #3d4663; border-radius: 8px; padding: 6px 12px; color: #ccd6f6; font-size: 13px; }
            QToolBar QToolButton:hover { background: #3d4663; border: 2px solid #64ffda; }
            QStatusBar { background: #1a1d29; color: #8892b0; border-top: 2px solid #3d4663; font-size: 12px; }
            """
        )

    @Slot(str)
    def open_exercise(self, exercise_type: str):
        if exercise_type not in self.exercise_screens:
            screen = ExerciseScreen(exercise_type)
            screen.back_clicked.connect(self.go_home)
            self.exercise_screens[exercise_type] = screen
            self.stacked_widget.addWidget(screen)
        self.stacked_widget.setCurrentWidget(self.exercise_screens[exercise_type])
        self.statusBar().showMessage(f"{Translations.get('title_brand')}  |  {exercise_type.upper()}")

    @Slot()
    def open_settings(self):
        self.stacked_widget.setCurrentWidget(self.settings_screen)
        self.statusBar().showMessage(Translations.get("settings"))

    @Slot()
    def go_home(self):
        for screen in self.exercise_screens.values():
            screen.cleanup()
        self.stacked_widget.setCurrentWidget(self.home_screen)
        self.statusBar().showMessage(Translations.get("ready"))

    @Slot(int)
    def change_language(self, index):
        lang = self.lang_combo.itemData(index)
        Translations.set_language(lang)
        # switch app direction for Arabic
        app = QApplication.instance()
        if lang == Language.ARABIC:
            app.setLayoutDirection(Qt.RightToLeft)
        else:
            app.setLayoutDirection(Qt.LeftToRight)
        # refresh text
        self.retranslate_all()

    def _refresh_lang_combo(self):
        current_lang = Translations.current_lang
        self.lang_combo.blockSignals(True)
        self.lang_combo.clear()
        self.lang_combo.addItem(Translations.get("english"), Language.ENGLISH)
        self.lang_combo.addItem(Translations.get("arabic"), Language.ARABIC)
        self.lang_combo.setCurrentIndex(0 if current_lang == Language.ENGLISH else 1)
        self.lang_combo.blockSignals(False)

    @Slot()
    def retranslate_all(self):
        self.setWindowTitle(Translations.get("app_title"))
        self.title_label.setText(Translations.get("title_brand"))
        self.act_home.setText(f"🏠 {Translations.get('home')}")
        self.act_settings.setText(f"⚙️ {Translations.get('settings')}")
        self.lang_label.setText(f"🌐 {Translations.get('language_label')}")
        self._refresh_lang_combo()
        self.home_screen.retranslate()
        self.settings_screen.retranslate()
        for screen in self.exercise_screens.values():
            screen.retranslate()

    def closeEvent(self, event):
        for screen in self.exercise_screens.values():
            screen.cleanup()
        super().closeEvent(event)

# ---------- Application Entry ----------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # default direction LTR; language switch will flip it
    app.setLayoutDirection(Qt.LeftToRight)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
