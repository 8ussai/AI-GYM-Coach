from pathlib import Path

# الجذر النسبي
BASE_DIR = Path(__file__).resolve().parents[2]

# إعداد YOLO
YOLO_DIR = BASE_DIR / "yolo_training"
YOLO_WEIGHTS = r"C:\Users\mrqus\Desktop\YOLO Objects Detectors\barbell_detector\dataset\runs\barbull-seg\weights\best.pt"          # أو أي وزن جاهز عندك
YOLO_BAR_CLASS_ID = 0  # default is 1 for barbell_on_shoulder


# ثقة الكشف الافتراضية (YOLO)
YOLO_CONF_THRESH = 0.3
YOLO_IOU_THRESH = 0.5

# إعدادات عامة
FPS_TARGET = 15
SMOOTHING_METHOD = "ema"
SMOOTHING_ALPHA = 0.2

# تمرين افتراضي لتجارب سريعة
DEFAULT_EXERCISE = "squat"

# المسارات الافتراضية للأوتبوت (تستخدمها باقي السكربتات)
OUTPUTS_BASE = BASE_DIR / "outputs"
RAW_DIR      = OUTPUTS_BASE / "raw"
CLEANED_DIR  = OUTPUTS_BASE / "cleaned"
DATASETS_DIR = OUTPUTS_BASE / "datasets"
MODELS_DIR   = OUTPUTS_BASE / "models"
