from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
YOLO_DIR = BASE_DIR / "yolo_training"
YOLO_WEIGHTS = YOLO_DIR / "runs" / "train" / "train" / "weights" / "best.pt"        
YOLO_DUMBELL_CLASS_ID = 0  
YOLO_BARBELL_CLASS_ID = 1  
YOLO_CONF_THRESH = 0.3
YOLO_IOU_THRESH = 0.5
FPS_TARGET = 15
SMOOTHING_METHOD = "ema"
SMOOTHING_ALPHA = 0.2
OUTPUTS_BASE = BASE_DIR / "outputs"
RAW_DIR      = OUTPUTS_BASE / "raw"
CLEANED_DIR  = OUTPUTS_BASE / "cleaned"
DATASETS_DIR = OUTPUTS_BASE / "datasets"
MODELS_DIR   = OUTPUTS_BASE / "models"