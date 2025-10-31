import pandas as pd
import json

from pathlib import Path

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_csv(df: pd.DataFrame, path: Path, mode: str = "w", header: bool | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if header is None:
        header = (not path.exists()) or (mode == "w")
    df.to_csv(path, index=False, mode=mode, header=header)

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)