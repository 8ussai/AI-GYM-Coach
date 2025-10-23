#!/usr/bin/env python3
# modules/live_analyzer/feature_extractor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import cv2

# MediaPipe
import mediapipe as mp
mp_pose = mp.solutions.pose

# YOLO (ultralytics) optional for bar_on_shoulder
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# استخدم نفس الدوال الموجودة عندك في geometry.py
# calculate_angle, angle_from_vertical, distance
from modules.common.geometry import calculate_angle, angle_from_vertical, distance  # :contentReference[oaicite:1]{index=1}


@dataclass
class YOLOConfig:
    weights_path: Optional[str] = None   # e.g. "yolo_training/runs/detect/exp/weights/best.pt"
    class_bar_on_shoulder: int = 1
    conf: float = 0.4
    iou: float = 0.45


class FeatureExtractor:
    """
    Live extractor that returns the exact same feature keys used in training:
      - sq_knee_angle_L, sq_knee_angle_R
      - sq_elbow_angle_L, sq_elbow_angle_R
      - sq_torso_incline
      - sq_pelvis_drop   (normalized)
      - sq_stance_ratio  (normalized)
      - pose_confidence
      - sq_bar_present   (0/1 via YOLO, optional)
    """
    def __init__(self, yolo_cfg: Optional[YOLOConfig] = None):
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                 enable_segmentation=False,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)
        self.yolo = None
        self.cfg = yolo_cfg
        if YOLO is not None and yolo_cfg and yolo_cfg.weights_path:
            try:
                self.yolo = YOLO(yolo_cfg.weights_path)
            except Exception as e:
                print(f"[WARN] YOLO load failed: {e}. Bar detection disabled.")

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass

    def _detect_bar(self, frame_bgr) -> int:
        if self.yolo is None:
            return 0
        try:
            res = self.yolo.predict(frame_bgr, verbose=False, conf=self.cfg.conf, iou=self.cfg.iou)[0]
            if res is not None and res.boxes is not None and len(res.boxes) > 0:
                cls = res.boxes.cls.detach().cpu().numpy().astype(int)
                return int((cls == self.cfg.class_bar_on_shoulder).any())
        except Exception as e:
            print(f"[WARN] YOLO inference failed: {e}")
        return 0

    def compute_features(self, frame_bgr) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """
        Returns:
          feats: dict with the exact keys used in training
          lm_xy: optional landmarks [33,2] for overlay
        """
        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        # إعداد مفاتيح ثابتة لو ما في لاندماركس
        empty_feats = {
            "sq_knee_angle_L": np.nan, "sq_knee_angle_R": np.nan,
            "sq_elbow_angle_L": np.nan, "sq_elbow_angle_R": np.nan,
            "sq_torso_incline": np.nan, "sq_pelvis_drop": np.nan,
            "sq_stance_ratio": np.nan, "pose_confidence": 0.0,
            "sq_bar_present": 0,
        }

        if not res.pose_landmarks:
            return empty_feats, None

        lm = res.pose_landmarks.landmark
        # نقاط 2D (x,y) مطلوبة لدوال geometry
        lm_xy = np.array([[lm[i].x, lm[i].y] for i in range(33)], dtype=float)

        # إندكسات MediaPipe اللي بنحتاجها
        L_SHO, R_SHO = 11, 12
        L_ELB, R_ELB = 13, 14
        L_WR,  R_WR  = 15, 16
        L_HIP, R_HIP = 23, 24
        L_KNEE,R_KNEE= 25, 26
        L_ANK, R_ANK = 27, 28

        # زوايا الركبة (ABC عند الركبة)
        knee_L = calculate_angle(tuple(lm_xy[L_HIP]), tuple(lm_xy[L_KNEE]), tuple(lm_xy[L_ANK]))
        knee_R = calculate_angle(tuple(lm_xy[R_HIP]), tuple(lm_xy[R_KNEE]), tuple(lm_xy[R_ANK]))

        # زوايا الكوع (ABC عند الكوع)
        elbow_L = calculate_angle(tuple(lm_xy[L_SHO]), tuple(lm_xy[L_ELB]), tuple(lm_xy[L_WR]))
        elbow_R = calculate_angle(tuple(lm_xy[R_SHO]), tuple(lm_xy[R_ELB]), tuple(lm_xy[R_WR]))

        # ميل الجذع: زاوية الخط بين منتصف الكتفين ومنتصف الوركين بالنسبة للمحور الرأسي
        shoulder_mid = ((lm_xy[L_SHO] + lm_xy[R_SHO]) * 0.5).tolist()
        hip_mid      = ((lm_xy[L_HIP] + lm_xy[R_HIP]) * 0.5).tolist()
        torso_incline = angle_from_vertical(tuple(shoulder_mid), tuple(hip_mid))

        # عرض الكتفين
        sho_width = distance(tuple(lm_xy[L_SHO]), tuple(lm_xy[R_SHO])) + 1e-6

        # pelvis_drop (فرق رأسي بين الوركين) مُطبّع على عرض الكتفين
        pelvis_drop = abs(lm_xy[L_HIP][1] - lm_xy[R_HIP][1]) / sho_width

        # stance_ratio (المسافة بين الكاحلين) / عرض الكتفين
        stance = distance(tuple(lm_xy[L_ANK]), tuple(lm_xy[R_ANK])) / sho_width

        # ثقة الوضعية: متوسط visibility لبعض المفاصل
        try:
            vis_idx = [L_SHO,R_SHO,L_ELB,R_ELB,L_WR,R_WR,L_HIP,R_HIP,L_KNEE,R_KNEE,L_ANK,R_ANK]
            pose_conf = float(np.clip(np.mean([lm[i].visibility for i in vis_idx]), 0.0, 1.0))
        except Exception:
            pose_conf = 0.0

        # YOLO bar presence (اختياري)
        bar_present = self._detect_bar(frame_bgr)

        feats = {
            "sq_knee_angle_L": float(knee_L) if knee_L is not None else np.nan,
            "sq_knee_angle_R": float(knee_R) if knee_R is not None else np.nan,
            "sq_elbow_angle_L": float(elbow_L) if elbow_L is not None else np.nan,
            "sq_elbow_angle_R": float(elbow_R) if elbow_R is not None else np.nan,
            "sq_torso_incline": float(torso_incline) if torso_incline is not None else np.nan,
            "sq_pelvis_drop": float(pelvis_drop),
            "sq_stance_ratio": float(stance),
            "pose_confidence": pose_conf,
            "sq_bar_present": int(bar_present),
        }
        return feats, lm_xy
