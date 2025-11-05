# -*- coding: utf-8 -*-
"""信号指标计算与候选构建模块。

该模块负责：
1. 按照 V29 配置计算 EMA/RSI/ATR/ADX 等指标（包含 V29.1 修订 #1：
   当 ADX_{cfg.adx_len} 列不存在时回退使用 ADX_20 结果）；
2. 基于单根 K 线数据生成多种交易候选（MRL/PBL/TRS 等 squad）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import math
import pandas as pd
import pandas_ta as ta


@dataclass
class Candidate:
    """描述 signal 阶段产出的单个建仓候选。

    Attributes:
        direction: 信号方向，"long" 或 "short"。
        kind: 具体策略类型（如 mean_rev_long / pullback_long / trend_short）。
        sl_pct: 建议的止损百分比。
        tp_pct: 建议的止盈百分比。
        raw_score: 原始强度评分（0~1）。
        rr_ratio: Reward/Risk 比。
        win_prob: 预估胜率。
        expected_edge: 期望收益（胜率 * TP - (1-胜率) * SL）。
        squad: 所属 squad 名称，供财政分配时过滤。
    """

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
    """根据配置计算并附加信号所需的技术指标。

    - EMA/RSI/ATR 均使用 pandas_ta 计算；
    - ADX 部分遵循 V29.1 修订 #1：优先读取列名为 "ADX_{cfg.adx_len}" 的结果，
      若第三方库返回的列名缺失则回退到 "ADX_20" 输出；
    - 额外计算 tr_pct，供后续止损/止盈估算使用。

    Args:
        df: 原始 K 线数据。
        cfg: V29Config，提供指标长度配置。

    Returns:
        pd.DataFrame: 附加指标列后的数据帧，原地修改并返回。
    """

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
    elif isinstance(adx_df, pd.DataFrame) and "ADX_20" in adx_df.columns:
        df["adx"] = adx_df["ADX_20"]
    else:
        df["adx"] = 20.0

    return df


def gen_candidates(row: pd.Series) -> List[Candidate]:
    """将最近一根 K 线的指标值转化为可交易候选列表。

    当前实现含三类候选：
        - mean_rev_long (MRL)：极端 RSI 反转；
        - pullback_long (PBL)：多头回踩；
        - trend_short (TRS)：顺势做空。
    每个候选都根据 ATR% 估算止损/止盈、计算 Reward/Risk，并给出期望收益。

    Args:
        row: 包含 close/ema_fast/ema_slow/rsi/atr_pct/adx 等字段的 Series。

    Returns:
        List[Candidate]: 可能为空的候选列表。
    """

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
