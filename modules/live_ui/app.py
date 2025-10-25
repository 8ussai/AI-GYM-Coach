#!/usr/bin/env python3
# modules/live_ui/app.py
"""
PySide6 GUI: AI Gym Coach - Real-time Squat Analysis
- Live camera feed with pose detection
- GRU model inference for form validation
- Real-time audio feedback in Arabic
- Rep counting with form quality assessment
"""

import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PySide6.QtCore import Qt, QSize, QTimer, Slot, Signal
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QGroupBox, QFormLayout, QComboBox, QSpinBox, QCheckBox, QSizePolicy,
    QStackedWidget, QStatusBar, QProgressBar, QTextEdit
)

from modules.live_ui.workers import VideoWorker, AnalyzerWorker


def cvimg_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert OpenCV image to QPixmap for display"""
    if frame is None:
        return QPixmap()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class VideoLabel(QLabel):
    """Custom label for video display with styling"""
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(960, 540)
        self.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
            "stop:0 #1a1f2e, stop:1 #2d3748); "
            "color: #e2e8f0; "
            "border: 3px solid #4a5568; "
            "border-radius: 16px; "
            "font-size: 18px; "
            "font-weight: bold;"
        )
        self.setText("🎥 اختر الكاميرا أو الفيديو واضغط Start")

    def set_frame(self, frame: np.ndarray):
        """Update displayed frame"""
        pix = cvimg_to_qpixmap(frame)
        if not pix.isNull():
            self.setPixmap(pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        """Handle resize events"""
        if self.pixmap() and not self.pixmap().isNull():
            self.setPixmap(self.pixmap().scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(ev)


class MetricCard(QWidget):
    """Custom widget for displaying metrics"""
    def __init__(self, title: str, initial_value: str = "0"):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2d3748, stop:1 #1a202c);
                border: 2px solid #4a5568;
                border-radius: 12px;
            }
        """)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: 600;")
        title_label.setAlignment(Qt.AlignCenter)

        self.value_label = QLabel(initial_value)
        self.value_label.setStyleSheet("color: #e2e8f0; font-size: 28px; font-weight: bold;")
        self.value_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str):
        """Update the metric value"""
        self.value_label.setText(value)

    def set_color(self, color: str):
        """Change value color"""
        self.value_label.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")


class SquatScreen(QWidget):
    """Main squat analysis screen"""
    def __init__(self):
        super().__init__()
        self.worker_cam: Optional[VideoWorker] = None
        self.worker_an: Optional[AnalyzerWorker] = None
        self.current_source: Optional[Union[int, str]] = None

        # Statistics
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0

        self._setup_ui()

        # UI update timer
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._tick)
        self._ui_timer.start(100)

    def _setup_ui(self):
        """Setup the user interface"""
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        # === LEFT PANEL: Controls ===
        left = self._create_control_panel()

        # === CENTER: Video Display ===
        center_box = QVBoxLayout()
        self.video_label = VideoLabel()
        center_box.addWidget(self.video_label, 1)

        # === RIGHT PANEL: Metrics & Feedback ===
        right = self._create_metrics_panel()

        root.addWidget(left, 0)
        root.addLayout(center_box, 1)
        root.addWidget(right, 0)

    def _create_control_panel(self) -> QGroupBox:
        """Create the left control panel"""
        left = QGroupBox("⚙️ الإعدادات")
        left.setMaximumWidth(340)
        left.setStyleSheet("""
            QGroupBox {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 16px;
                font-size: 16px;
                font-weight: bold;
                color: #e2e8f0;
            }
        """)

        l = QVBoxLayout(left)
        l.setSpacing(12)

        # Camera selection
        form = QFormLayout()
        form.setSpacing(10)

        self.cam_combo = QComboBox()
        self.cam_combo.setStyleSheet("padding: 8px; font-size: 14px;")
        self._populate_cameras()

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(5, 60)
        self.fps_spin.setValue(30)
        self.fps_spin.setStyleSheet("padding: 8px; font-size: 14px;")

        self.flip_check = QCheckBox("قلب الكاميرا")
        self.flip_check.setChecked(True)
        self.flip_check.setStyleSheet("font-size: 14px; color: #cbd5e1;")

        self.tts_check = QCheckBox("التعليق الصوتي")
        self.tts_check.setChecked(True)
        self.tts_check.setStyleSheet("font-size: 14px; color: #cbd5e1;")

        # Add threshold control
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(50, 95)
        self.threshold_spin.setValue(35)
        self.threshold_spin.setSuffix("%")
        self.threshold_spin.setStyleSheet("padding: 8px; font-size: 14px;")

        form.addRow("الكاميرا:", self.cam_combo)
        form.addRow("FPS المستهدف:", self.fps_spin)
        form.addRow("حد القبول:", self.threshold_spin)
        form.addRow("", self.flip_check)
        form.addRow("", self.tts_check)
        l.addLayout(form)

        # Source buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_cam = QPushButton("📷 كاميرا")
        self.btn_cam.setStyleSheet(self._button_style())
        self.btn_cam.clicked.connect(self._select_first_camera)

        self.btn_vid = QPushButton("🎬 فيديو")
        self.btn_vid.setStyleSheet(self._button_style())
        self.btn_vid.clicked.connect(self._open_video)

        btn_row.addWidget(self.btn_cam)
        btn_row.addWidget(self.btn_vid)
        l.addLayout(btn_row)

        # Start/Stop buttons
        self.btn_start = QPushButton("▶️ ابدأ")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #059669, stop:1 #10b981);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #047857, stop:1 #059669);
            }
            QPushButton:disabled {
                background: #374151;
                color: #6b7280;
            }
        """)
        self.btn_start.clicked.connect(self.handle_start)

        self.btn_stop = QPushButton("⏹️ توقف")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #dc2626, stop:1 #ef4444);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #b91c1c, stop:1 #dc2626);
            }
            QPushButton:disabled {
                background: #374151;
                color: #6b7280;
            }
        """)
        self.btn_stop.clicked.connect(self.handle_stop)
        self.btn_stop.setEnabled(False)

        l.addWidget(self.btn_start)
        l.addWidget(self.btn_stop)
        l.addStretch()

        return left

    def _create_metrics_panel(self) -> QGroupBox:
        """Create the right metrics panel"""
        right = QGroupBox("📊 الإحصائيات")
        right.setMaximumWidth(340)
        right.setStyleSheet("""
            QGroupBox {
                background: #1e293b;
                border: 2px solid #334155;
                border-radius: 12px;
                padding: 16px;
                font-size: 16px;
                font-weight: bold;
                color: #e2e8f0;
            }
        """)

        r = QVBoxLayout(right)
        r.setSpacing(12)

        # Metric cards
        self.card_fps = MetricCard("FPS", "0.0")
        self.card_reps = MetricCard("العدات", "0")
        self.card_accuracy = MetricCard("الدقة", "0%")

        r.addWidget(self.card_fps)
        r.addWidget(self.card_reps)
        r.addWidget(self.card_accuracy)

        # Status and feedback section
        status_group = QWidget()
        status_group.setStyleSheet("""
            QWidget {
                background: #0f172a;
                border: 2px solid #1e293b;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        status_layout = QVBoxLayout(status_group)

        self.lbl_status = QLabel("⏸️ في الانتظار...")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: 600;")
        self.lbl_status.setWordWrap(True)

        self.lbl_feedback = QLabel("")
        self.lbl_feedback.setStyleSheet("color: #fbbf24; font-size: 13px; margin-top: 8px;")
        self.lbl_feedback.setWordWrap(True)

        status_layout.addWidget(self.lbl_status)
        status_layout.addWidget(self.lbl_feedback)

        r.addWidget(status_group)

        # Last rep info
        self.lbl_last = QLabel("آخر عدة: -")
        self.lbl_last.setStyleSheet("""
            color: #cbd5e1;
            font-size: 13px;
            padding: 10px;
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #1e293b;
        """)
        self.lbl_last.setWordWrap(True)
        r.addWidget(self.lbl_last)

        # Debug log
        debug_label = QLabel("🔍 Debug Log:")
        debug_label.setStyleSheet("color: #94a3b8; font-size: 12px; margin-top: 10px;")
        r.addWidget(debug_label)

        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setMaximumHeight(120)
        self.debug_log.setStyleSheet("""
            QTextEdit {
                background: #0f172a;
                color: #64748b;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
                font-family: 'Courier New', monospace;
            }
        """)
        r.addWidget(self.debug_log)

        r.addStretch()

        return right

    def _button_style(self) -> str:
        """Common button style"""
        return """
            QPushButton {
                background: #334155;
                color: #e2e8f0;
                border: 1px solid #475569;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #475569;
            }
            QPushButton:disabled {
                background: #1e293b;
                color: #64748b;
            }
        """

    def _populate_cameras(self, max_index: int = 4):
        """Find and populate available cameras"""
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
                self.cam_combo.addItem(f"كاميرا {i}", i)
        else:
            self.cam_combo.addItem("لا توجد كاميرا", None)

        self.cam_combo.blockSignals(False)

    def _add_debug_log(self, message: str):
        """Add message to debug log"""
        self.debug_log.append(message)
        scrollbar = self.debug_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot()
    def _select_first_camera(self):
        """Select first available camera"""
        if self.cam_combo.count() > 0 and self.cam_combo.itemData(0) is not None:
            self.current_source = int(self.cam_combo.itemData(0))
            self.lbl_status.setText(f"✅ تم اختيار كاميرا {self.current_source}")
            self._add_debug_log(f"[INFO] Selected camera {self.current_source}")
        else:
            self.current_source = 0
            self.lbl_status.setText("⚠️ محاولة استخدام الكاميرا الافتراضية")
            self._add_debug_log("[WARN] No camera found, using default")

    @Slot()
    def _open_video(self):
        """Open video file dialog"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "اختر ملف فيديو",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if path:
            self.current_source = path
            self.lbl_status.setText(f"✅ تم اختيار الفيديو")
            self._add_debug_log(f"[INFO] Selected video: {Path(path).name}")

    @Slot()
    def handle_start(self):
        """Start video capture and analysis"""
        if self.worker_cam and self.worker_cam.isRunning():
            return

        if self.current_source is None:
            self._select_first_camera()

        if self.current_source is None:
            self.lbl_status.setText("❌ لا توجد كاميرا متاحة")
            self._add_debug_log("[ERROR] No camera available")
            return

        # Reset statistics
        self.total_reps = 0
        self.good_reps = 0
        self.bad_reps = 0
        self._update_metrics()
        self.debug_log.clear()
        self._add_debug_log("[START] Initializing analyzer...")

        try:
            # ========= توافق مع تواقيع مختلفة لـ AnalyzerWorker =========
            threshold = self.threshold_spin.value() / 100.0

            def make_analyzer():
                base_kwargs = dict(
                    exercise="squat",
                    speak_out=self.tts_check.isChecked(),
                    yolo_weights=None
                )
                # 1) اسم حديث
                try:
                    return AnalyzerWorker(**base_kwargs, confidence_threshold=threshold)
                except TypeError:
                    pass
                # 2) اسم بديل شائع
                try:
                    return AnalyzerWorker(**base_kwargs, threshold=threshold)
                except TypeError:
                    pass
                # 3) إنشاء بدون باراميتر ثم محاولة استخدام setter إن وجد
                an = AnalyzerWorker(**base_kwargs)
                for setter_name in ("set_confidence_threshold", "set_threshold", "set_accept_prob"):
                    if hasattr(an, setter_name):
                        try:
                            getattr(an, setter_name)(threshold)
                            break
                        except Exception:
                            pass
                return an

            self.worker_an = make_analyzer()
            # ============================================================

            self.worker_an.overlay_ready.connect(self._on_overlay)
            self.worker_an.rep_event.connect(self._on_rep)
            self.worker_an.status.connect(self._on_status)
            self.worker_an.error.connect(self._on_error)
            self.worker_an.start()

            self._add_debug_log(f"[INFO] Analyzer started (threshold≈{threshold:.2f})")

            # Start video worker
            self.worker_cam = VideoWorker(
                self.current_source,
                target_fps=float(self.fps_spin.value()),
                flip=self.flip_check.isChecked()
            )

            self.worker_cam.frame_ready.connect(self._on_frame)
            self.worker_cam.stats_ready.connect(self._on_stats)
            self.worker_cam.error.connect(self._on_error)
            self.worker_cam.start()

            self._add_debug_log("[INFO] Video worker started")

            self.lbl_status.setText("▶️ التحليل قيد التشغيل...")
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_cam.setEnabled(False)
            self.btn_vid.setEnabled(False)

        except Exception as e:
            self._add_debug_log(f"[ERROR] Failed to start: {str(e)}")
            self.lbl_status.setText(f"❌ خطأ في التشغيل: {str(e)}")

    @Slot()
    def handle_stop(self):
        """Stop video capture and analysis"""
        self._add_debug_log("[STOP] Stopping workers...")

        if self.worker_cam and self.worker_cam.isRunning():
            self.worker_cam.stop()
            self.worker_cam.wait(1000)
        self.worker_cam = None

        if self.worker_an and self.worker_an.isRunning():
            self.worker_an.stop()
            self.worker_an.wait(1000)
        self.worker_an = None

        self.lbl_status.setText("⏹️ تم الإيقاف")
        self.lbl_feedback.setText("")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_cam.setEnabled(True)
        self.btn_vid.setEnabled(True)

        self._add_debug_log("[STOP] All workers stopped")

    @Slot(np.ndarray)
    def _on_frame(self, frame: np.ndarray):
        """Handle new frame from video worker"""
        if self.worker_an and self.worker_an.isRunning():
            self.worker_an.push_frame(frame)

    @Slot(np.ndarray)
    def _on_overlay(self, frame: np.ndarray):
        """Display frame with pose overlay"""
        self.video_label.set_frame(frame)

    @Slot(dict)
    def _on_rep(self, info: dict):
        """Handle rep completion event"""
        rep = info.get("rep", 0)
        pred = info.get("pred", "unknown")
        prob = info.get("prob", 0.0)
        reason = info.get("reason", "")

        self._add_debug_log(f"[REP] #{rep} | pred={pred} | prob={prob:.2%} | reason={reason}")

        self.total_reps = rep

        pred_lower = str(pred).lower().strip()
        is_good = pred_lower in ["good", "جيد", "صحيح", "1", "correct", "pass"]

        if is_good:
            self.good_reps += 1
            status_emoji = "✅"
            color = "#10b981"
            feedback_msg = "👍 ممتاز! استمر"
        else:
            self.bad_reps += 1
            status_emoji = "❌"
            color = "#ef4444"
            feedback_msg = f"⚠️ يحتاج تحسين"

        txt = f"{status_emoji} العدة {rep}: {pred} ({prob:.1%})"
        if reason:
            txt += f"\n💡 {reason}"
            feedback_msg = f"📢 {reason}"

        self.lbl_last.setText(txt)
        self.lbl_last.setStyleSheet(f"""
            color: {color};
            font-size: 13px;
            font-weight: bold;
            padding: 10px;
            background: #0f172a;
            border-radius: 8px;
            border: 2px solid {color};
        """)

        self.lbl_feedback.setText(feedback_msg)
        self.lbl_feedback.setStyleSheet(f"color: {color}; font-size: 13px; margin-top: 8px;")

        self._update_metrics()

        if self.tts_check.isChecked():
            self._add_debug_log(f"[TTS] Speaking: {reason if reason else feedback_msg}")

    @Slot(dict)
    def _on_stats(self, stats: dict):
        """Update FPS display"""
        fps = stats.get('fps', 0.0)
        self.card_fps.set_value(f"{fps:.1f}")

    @Slot(str)
    def _on_status(self, status: str):
        """Update status message"""
        self.lbl_status.setText(f"ℹ️ {status}")
        self._add_debug_log(f"[STATUS] {status}")

    @Slot(str)
    def _on_error(self, error: str):
        """Handle error message"""
        self.lbl_status.setText(f"❌ خطأ: {error}")
        self.lbl_status.setStyleSheet("color: #ef4444; font-size: 14px; font-weight: 600;")
        self._add_debug_log(f"[ERROR] {error}")

    def _update_metrics(self):
        """Update metrics display"""
        self.card_reps.set_value(str(self.total_reps))

        if self.total_reps > 0:
            accuracy = (self.good_reps / self.total_reps) * 100
            self.card_accuracy.set_value(f"{accuracy:.0f}%")

            if accuracy >= 80:
                self.card_accuracy.set_color("#10b981")
            elif accuracy >= 60:
                self.card_accuracy.set_color("#fbbf24")
            else:
                self.card_accuracy.set_color("#ef4444")
        else:
            self.card_accuracy.set_value("0%")
            self.card_accuracy.set_color("#94a3b8")

    def cleanup(self):
        """Cleanup resources"""
        self.handle_stop()

    def _tick(self):
        """Periodic UI update"""
        pass


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Gym Coach - مدرب الصالة الذكي")
        self.resize(1600, 900)

        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #1e293b);
            }
            QLabel {
                color: #e2e8f0;
            }
        """)

        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)

        self.status = QStatusBar(self)
        self.status.setStyleSheet("""
            QStatusBar {
                background: #1e293b;
                color: #94a3b8;
                border-top: 1px solid #334155;
                padding: 5px;
            }
        """)
        self.setStatusBar(self.status)

        self.home = self._create_home_screen()
        self.squat_screen = SquatScreen()

        self.stacked.addWidget(self.home)
        self.stacked.addWidget(self.squat_screen)
        self.status.showMessage("🏠 جاهز للبدء")

    def _create_home_screen(self) -> QWidget:
        home = QWidget()
        home.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f172a, stop:1 #1e293b);
            }
        """)

        v = QVBoxLayout(home)
        v.setSpacing(30)

        title = QLabel("🏋️ اختر التمرين")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #e2e8f0;
            margin: 40px;
        """)
        v.addWidget(title)

        subtitle = QLabel("مدرب AI لتحليل وتصحيح حركاتك الرياضية")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 20px;
            color: #94a3b8;
            margin-bottom: 20px;
        """)
        v.addWidget(subtitle)

        row = QHBoxLayout()
        btn_squat = QPushButton("🦵 السكوات\nSquat")
        btn_squat.setMinimumSize(280, 180)
        btn_squat.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                color: white;
                border: 3px solid #60a5fa;
                border-radius: 20px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2563eb, stop:1 #1d4ed8);
                border-color: #93c5fd;
            }
        """)
        btn_squat.clicked.connect(self.open_squat)

        row.addStretch()
        row.addWidget(btn_squat)
        row.addStretch()

        v.addLayout(row)
        v.addStretch()

        return home

    @Slot()
    def open_squat(self):
        self.stacked.setCurrentWidget(self.squat_screen)
        self.status.showMessage("🦵 وضع السكوات النشط")
        self.squat_screen.lbl_status.setText("⏸️ في الانتظار...")
        self.squat_screen.lbl_feedback.clear()
        self.squat_screen.lbl_last.setText("آخر عدة: -")
        self.squat_screen.btn_start.setEnabled(True)
        self.squat_screen.btn_stop.setEnabled(False)
        self.squat_screen.btn_cam.setEnabled(True)
        self.squat_screen.btn_vid.setEnabled(True)

    def closeEvent(self, event):
        try:
            self.squat_screen.cleanup()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
