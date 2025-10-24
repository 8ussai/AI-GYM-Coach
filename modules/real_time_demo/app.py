#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Squat Form Analysis Demo using Streamlit, from Webcam or Uploaded Video.

Usage:
    streamlit run modules/real_time_demo/app.py --server.port 5000
"""

import sys
from pathlib import Path
import tempfile
import os
import time

import streamlit as st
import cv2
import numpy as np

# Make repo root importable
workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from modules.data_extraction.mediapipe_runner import PoseRunner
from modules.common.feature_builder import build_features
from modules.common.config import DEFAULT_EXERCISE
from modules.real_time_demo.rep_tracker import RealtimeRepTracker, SquatThresholds
from modules.real_time_demo.realtime_inference import RealtimeInference
from modules.real_time_demo.visualizer import Visualizer

try:
    from modules.data_extraction.yolo_runner import YoloRunner
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    YOLO_ERROR = str(e)


st.set_page_config(
    page_title="Squat Form Analysis - Real-time Demo",
    page_icon="🏋️",
    layout="wide"
)

@st.cache_resource
def load_models():
    pose = PoseRunner(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    yolo = None
    if YOLO_AVAILABLE:
        try:
            yolo = YoloRunner(device="cuda:0")
        except Exception as e:
            st.warning(f"YOLO initialization failed: {e}. Continuing without bar detection.")

    inference = RealtimeInference(exercise=DEFAULT_EXERCISE)
    return pose, yolo, inference


def open_capture(source_type: str, camera_index: int = 0, uploaded_path: str | None = None):
    """Create a cv2.VideoCapture from webcam or a saved uploaded file."""
    if source_type == "Webcam":
        cap = cv2.VideoCapture(camera_index)
        return cap
    else:
        if not uploaded_path or not os.path.exists(uploaded_path):
            return None
        cap = cv2.VideoCapture(uploaded_path)
        return cap


def main():
    st.title("🏋️ Real-Time Squat Form Analysis")

    if not YOLO_AVAILABLE:
        st.warning("⚠️ YOLO غير متوفر. راح نكمّل بدون كشف البار.")

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("الإعدادات")

        exercise = st.selectbox("التمرين", ["squat"], index=0)

        # مصدر الفيديو: كاميرا أو رفع ملف
        source_type = st.radio("مصدر الإدخال", ["Webcam", "Video file"], index=0, horizontal=True)

        uploaded_video = None
        temp_video_path = None
        camera_index = 0

        if source_type == "Webcam":
            camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        else:
            uploaded_video = st.file_uploader(
                "ارفع فيديو التمرين",
                type=["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=False
            )
            if uploaded_video is not None:
                # احفظ الملف مؤقتاً حتى يقدر OpenCV يفتحه
                suffix = "." + uploaded_video.name.split(".")[-1].lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded_video.read())
                tmp.flush()
                tmp.close()
                temp_video_path = tmp.name
                st.info(f"تم تجهيز الفيديو: {uploaded_video.name}")

        with st.expander("Thresholds متقدمة"):
            rest_knee = st.slider("Rest Knee Angle", 150, 180, 165)
            start_knee = st.slider("Start Knee Angle", 140, 170, 155)
            bottom_knee = st.slider("Bottom Knee Angle", 80, 140, 120)
            max_torso = st.slider("Max Torso Incline", 10, 40, 20)

        playback_speed =  st.slider("سرعة العرض (FPS افتراضي ~ 30 فيديو/ ~ كاميرا)", 1, 60, 30)

        start_demo = st.button("Start", type="primary")
        stop_demo = st.button("Stop")

    with col_left:
        video_placeholder = st.empty()
        status_placeholder = st.empty()

    st.markdown("---")

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    if start_demo:
        # حماية حالة التشغيل
        st.session_state.demo_running = True

        pose, yolo, inference = load_models()
        if pose is None or inference is None:
            st.error("فشل تحميل الموديلات الأساسية (Pose/Inference). راجع الإعدادات.")
            return

        thresholds = SquatThresholds(
            rest_knee_deg=rest_knee,
            start_knee_deg=start_knee,
            bottom_knee_deg=bottom_knee,
            max_torso_incline=max_torso
        )
        rep_tracker = RealtimeRepTracker(thresholds)
        visualizer = Visualizer()

        cap = None
        if source_type == "Webcam":
            cap = open_capture("Webcam", camera_index=camera_index)
            if not cap or not cap.isOpened():
                st.error(f"تعذّر فتح الكاميرا {camera_index}")
                return
            status_placeholder.info("جاري الالتقاط من الكاميرا... اضغط Stop للإيقاف.")
        else:
            if temp_video_path is None:
                st.error("ارفع فيديو أولاً، بعدين اضغط Start.")
                return
            cap = open_capture("Video file", uploaded_path=temp_video_path)
            if not cap or not cap.isOpened():
                st.error("تعذّر فتح ملف الفيديو.")
                return
            status_placeholder.info("تشغيل الفيديو المرفوع... سيتوقف تلقائياً عند النهاية.")

        total_reps = 0
        correct_reps = 0
        incorrect_reps = 0
        recent_feedback = None
        feedback_timer = 0.0

        # استخدم FPS لضبط التأخير بشكل تقريبي
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / float(playback_speed if playback_speed > 0 else 30)

        # حلقة التشغيل
        while st.session_state.get("demo_running", False):
            ret, frame = cap.read()
            if not ret:
                # انتهى الفيديو أو مشكلة قراءة
                if source_type == "Video file":
                    status_placeholder.success(f"انتهى الفيديو. Total reps: {total_reps}")
                else:
                    st.error("فشل التقاط إطار من الكاميرا.")
                break

            timestamp = time.time()

            landmarks = pose.process_bgr(frame)

            if landmarks is not None:
                frame = visualizer.draw_skeleton(frame, landmarks)

                detections = None
                if yolo is not None:
                    try:
                        detections = yolo.infer_bgr(frame)
                    except Exception:
                        detections = None

                features = build_features(landmarks, detections, exercise_type=exercise)
                completed_rep = rep_tracker.update(features, timestamp)

                if completed_rep is not None:
                    total_reps += 1
                    result = inference.classify_rep(completed_rep.features_sequence)
                    if result["prediction"] == "Correct":
                        correct_reps += 1
                    else:
                        incorrect_reps += 1

                    recent_feedback = {
                        "label": completed_rep.label,
                        "reason": completed_rep.reason,
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                    }
                    feedback_timer = timestamp + 3.0

                state_info = rep_tracker.get_current_state()
                frame = visualizer.draw_metrics(frame, features, state_info["state"], total_reps)

                if recent_feedback and timestamp < feedback_timer:
                    frame = visualizer.draw_rep_feedback(
                        frame,
                        recent_feedback["label"],
                        recent_feedback["reason"],
                        recent_feedback["prediction"],
                        recent_feedback["confidence"],
                    )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            stats_col1.metric("Total Reps", total_reps)
            stats_col2.metric("Correct", correct_reps)
            stats_col3.metric("Incorrect", incorrect_reps)
            accuracy = (correct_reps / total_reps * 100.0) if total_reps > 0 else 0.0
            stats_col4.metric("Accuracy", f"{accuracy:.1f}%")

            if stop_demo or not st.session_state.get("demo_running", False):
                break

            time.sleep(delay)

        cap.release()
        st.session_state.demo_running = False

        # نظّف الملف المؤقت لو تم رفع فيديو
        if source_type == "Video file" and temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

        status_placeholder.success(f"تم الإيقاف. Total reps: {total_reps}")


if __name__ == "__main__":
    main()
