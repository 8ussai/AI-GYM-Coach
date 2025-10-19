import pandas as pd

def exponential_moving_average(series: pd.Series, alpha: float = 0.2) -> pd.Series:
    return series.ewm(alpha=alpha, adjust=False).mean()

def moving_average(series: pd.Series, window: int = 5) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def smooth_dataframe(df: pd.DataFrame, columns, method: str = "ema", **kwargs) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if method == "ema":
            df[col] = exponential_moving_average(df[col], **kwargs)
        elif method == "mean":
            df[col] = moving_average(df[col], **kwargs)
    return df
