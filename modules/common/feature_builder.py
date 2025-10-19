from typing import Dict, Any, Optional, Iterable
import numpy as np
from modules.common.config import YOLO_BAR_CLASS_ID

from .geometry import calculate_angle, distance, angle_from_vertical

# إعدادات افتراضية حسب اتفقنا:
# - التطبيع بالمسافة بين الكتفين
# - إحداثيات MediaPipe normalized [0..1]
# - استخدام YOLO للكشف عن class_id = 1 (البار على الكتف) في السكوات

def _norm_scale(landmarks: Dict[str, Any]) -> float:
    """مقياس التطبيع: عرض الكتفين."""
    try:
        return max(distance(landmarks["shoulder_L"], landmarks["shoulder_R"]), 1e-6)
    except Exception:
        return 1.0

def _mean_conf(keys: Iterable[str], lm: Dict[str, Any]) -> float:
    """متوسط الثقة لو توفّر حقول *_conf، وإلا 1.0."""
    vals = []
    for k in keys:
        conf_key = f"{k}_conf"
        if conf_key in lm and lm[conf_key] is not None:
            vals.append(float(lm[conf_key]))
    return float(np.mean(vals)) if vals else 1.0

def _has_class(detections: Dict[str, Any], class_id: int, min_conf: float = 0.3) -> int:
    """
    يتحقق من وجود كائن لصف محدد في مخرجات YOLO.
    نتوقع detections["yolo"] كقائمة قواميس: {"cls": int, "conf": float, "bbox": (x,y,w,h)}.
    """
    if detections is None: 
        return 0
    items = detections.get("yolo", [])
    for obj in items:
        try:
            if int(obj.get("cls", -1)) == class_id and float(obj.get("conf", 0)) >= min_conf:
                return 1
        except Exception:
            continue
    return 0

def build_features(landmarks: Dict[str, Any], detections: Optional[Dict[str, Any]], exercise_type: str) -> Dict[str, Any]:
    """
    يحوّل landmarks + detections إلى ميزات جاهزة للكتابة في CSV لكل فريم.
    - landmarks: dict مفاتيحه أسماء نقاط MediaPipe مثل "shoulder_L"
                وقيمها tuples (x, y) normalized.
                يمكن توفر مفاتيح *_conf لثقة كل نقطة.
    - detections: dict اختيارية. نتوقع detections["yolo"] = قائمة كائنات من YOLO.
    - exercise_type: "squat" ... الخ.
    """
    feats: Dict[str, Any] = {}

    if exercise_type == "squat":
        # نقاط مهمة
        L = landmarks  # اختصار

        # مقياس التطبيع
        scale = _norm_scale(L)

        # زوايا الركبة
        feats["sq_knee_angle_L"] = calculate_angle(L["hip_L"], L["knee_L"], L["ankle_L"])
        feats["sq_knee_angle_R"] = calculate_angle(L["hip_R"], L["knee_R"], L["ankle_R"])

        # زوايا الورك (shoulder-hip-knee)
        feats["sq_hip_angle_L"] = calculate_angle(L["shoulder_L"], L["hip_L"], L["knee_L"])
        feats["sq_hip_angle_R"] = calculate_angle(L["shoulder_R"], L["hip_R"], L["knee_R"])

        # زوايا الكاحل (knee-ankle-foot_index)
        feats["sq_ankle_angle_L"] = calculate_angle(L["knee_L"], L["ankle_L"], L["foot_index_L"])
        feats["sq_ankle_angle_R"] = calculate_angle(L["knee_R"], L["ankle_R"], L["foot_index_R"])

        # ميل الجذع: الزاوية بين (hip_mid→shoulder_mid) والمحور الرأسي
        shoulder_mid = ((L["shoulder_L"][0]+L["shoulder_R"][0])/2, (L["shoulder_L"][1]+L["shoulder_R"][1])/2)
        hip_mid      = ((L["hip_L"][0]+L["hip_R"][0])/2,         (L["hip_L"][1]+L["hip_R"][1])/2)
        feats["sq_torso_incline"] = angle_from_vertical(shoulder_mid, hip_mid)

        # هبوط الحوض (فرق رأسي بين الوركين) normalized
        pelvis_drop_px = abs(L["hip_L"][1] - L["hip_R"][1])
        feats["sq_pelvis_drop"] = pelvis_drop_px / scale

        # عرض الوقفة normalized: مسافة الكاحلين ÷ عرض الكتفين
        stance = distance(L["ankle_L"], L["ankle_R"])
        feats["sq_stance_ratio"] = stance / scale

        # زوايا الكوعين المطلوبة منك
        feats["sq_elbow_angle_L"] = calculate_angle(L["shoulder_L"], L["elbow_L"], L["wrist_L"])
        feats["sq_elbow_angle_R"] = calculate_angle(L["shoulder_R"], L["elbow_R"], L["wrist_R"])

        # وجود البار على الكتف (YOLO class_id = 1)
        feats["sq_bar_present"] = _has_class(detections or {}, class_id=YOLO_BAR_CLASS_ID, min_conf=0.3)

        # ثقة الوضعية
        used_keys = [
            "shoulder_L","shoulder_R","hip_L","hip_R",
            "knee_L","knee_R","ankle_L","ankle_R",
            "foot_index_L","foot_index_R","elbow_L","elbow_R","wrist_L","wrist_R"
        ]
        feats["pose_confidence"] = _mean_conf(used_keys, L)

    else:
        # تمارين أخرى لاحقًا بنفس النمط
        pass

    return feats
