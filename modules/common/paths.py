from pathlib import Path

# جذر المشروع (modules/common/* → BASE_DIR)
BASE_DIR = Path(__file__).resolve().parents[2]

# مسارات عامة
VIDEOS_DIR   = BASE_DIR / "videos"
OUTPUTS_DIR  = BASE_DIR / "outputs"
RAW_DIR      = OUTPUTS_DIR / "raw"
CLEANED_DIR  = OUTPUTS_DIR / "cleaned"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
MODELS_DIR   = OUTPUTS_DIR / "models"

# دوال مساعدة مبنية على اسم التمرين (exercise)
def get_raw_csv(exercise: str):
    return RAW_DIR / f"{exercise}_frames.csv"

def get_cleaned_frames(exercise: str):
    return CLEANED_DIR / exercise / f"{exercise}_cleaned_frames.csv"

def get_cleaned_reps(exercise: str):
    return CLEANED_DIR / exercise / f"{exercise}_reps.csv"

def get_model_path(exercise: str):
    return MODELS_DIR / f"{exercise}_model.h5"

def get_model_meta(exercise: str):
    return MODELS_DIR / f"{exercise}_meta.json"

def get_dataset_dir(exercise: str) -> Path:
    """Return path to dataset directory for this exercise (used in model_training)."""
    p = OUTPUTS_DIR / "datasets" / exercise
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_models_dir(exercise: str) -> Path:
    """Return path to models directory for this exercise (for weights/meta)."""
    p = OUTPUTS_DIR / "models" / exercise
    p.mkdir(parents=True, exist_ok=True)
    return p