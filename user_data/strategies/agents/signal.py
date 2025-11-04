from __future__ import annotations

from dataclasses import dataclass
from typing import List

import math
import pandas as pd
import pandas_ta as ta


@dataclass
class Candidate:
    """Candidate 的职责说明。"""
    direction: str
    kind: str
    sl_pct: float
    tp_pct: float
    raw_score: float
    rr_ratio: float
    win_prob: float
    expected_edge: float
    squad: str


def compute_indicators(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Populate indicator columns required by downstream agents."""
    df["ema_fast"] = ta.ema(df["close"], length=cfg.ema_fast)
    df["ema_slow"] = ta.ema(df["close"], length=cfg.ema_slow)
    df["rsi"] = ta.rsi(df["close"], length=cfg.rsi_len)

    atr = ta.atr(df["high"], df["low"], df["close"], length=cfg.atr_len)
    df["atr"] = atr
    df["atr_pct"] = df["atr"] / df["close"]

    adx_df = ta.adx(df["high"], df["low"], df["close"], length=cfg.adx_len)
    adx_col = f"ADX_{cfg.adx_len}"
    if isinstance(adx_df, pd.DataFrame) and adx_col in adx_df.columns:
        df["adx"] = adx_df[adx_col]
    else:
        df["adx"] = 20.0

    return df


def gen_candidates(row: pd.Series) -> List[Candidate]:
    """Transform the latest indicator row into trading candidates."""
    out: list[Candidate] = []
    close = float(row["close"])
    ema_fast = float(row["ema_fast"])
    ema_slow = float(row["ema_slow"])
    rsi = float(row["rsi"])
    adx = float(row["adx"])
    atr_pct = float(row["atr_pct"])

    if any(math.isnan(v) for v in (close, ema_fast, ema_slow, rsi, adx, atr_pct)):
        return out

    if rsi < 25 and close < ema_fast * 0.985:
        sl = atr_pct * 1.2
        tp = atr_pct * 2.4
        raw = max(0.0, (25 - rsi) / 25.0)
        rr = tp / max(sl, 1e-9)
        win = min(0.9, max(0.5, 0.52 + 0.4 * raw))
        edge = win * tp - (1 - win) * sl
        out.append(Candidate("long", "mean_rev_long", sl, tp, raw, rr, win, edge, "MRL"))

    if ema_fast > ema_slow and adx > 20 and close < ema_fast * 0.99:
        sl = atr_pct * 1.0
        tp = atr_pct * 2.0
        raw = 0.5 * max(0.0, ema_fast / max(ema_slow, 1e-9) - 1.0) + 0.5 * max(0.0, (adx - 20) / 20)
        rr = tp / max(sl, 1e-9)
        win = min(0.95, max(0.5, 0.55 + 0.4 * raw))
        edge = win * tp - (1 - win) * sl
        out.append(Candidate("long", "pullback_long", sl, tp, raw, rr, win, edge, "PBL"))

    if ema_fast < ema_slow and adx > 25 and close > ema_fast * 1.01:
        sl = atr_pct * 1.2
        tp = atr_pct * 2.4
        raw = 0.5 * max(0.0, (adx - 25) / 25) + 0.5 * max(0.0, 1.0 - ema_fast / max(ema_slow, 1e-9))
        rr = tp / max(sl, 1e-9)
        win = min(0.95, max(0.5, 0.50 + 0.4 * raw))
        edge = win * tp - (1 - win) * sl
        out.append(Candidate("short", "trend_short", sl, tp, raw, rr, win, edge, "TRS"))

    return out
