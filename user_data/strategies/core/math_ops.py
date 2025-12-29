from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator, default: float = 0.0):
    try:
        denom = np.asarray(denominator)
        num = np.asarray(numerator)
        return np.where(denom == 0, default, num / denom)
    except Exception:
        try:
            return float(numerator) / float(denominator)
        except Exception:
            return default


def rolling_ema(series: pd.Series, span: int) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False).mean()


def clip01(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, 0.0, 1.0)

