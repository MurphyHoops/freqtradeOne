from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import math

import numpy as np
import pandas as pd
import talib.abstract as ta

from ..portfolio.global_backend import GlobalRiskBackend


@dataclass(frozen=True)
class MarketSnapshot:
    bias: float
    volatility: float
    btc_bias: float
    eth_bias: float
    btc_vol: float
    eth_vol: float


class MarketSensor:
    """感知层：提取 BTC/ETH 的合成 Bias 与 Volatility，并将结果推送到全局 backend。"""

    def __init__(
        self,
        backend: GlobalRiskBackend,
        weights: Optional[Dict[str, float]] = None,
        entropy_factor: float = 0.4,
    ) -> None:
        self.backend = backend
        self.weights = weights or {"BTC": 0.6, "ETH": 0.4}
        self.entropy_factor = float(entropy_factor)

    def _safe_bias_vol(self, df: Optional[pd.DataFrame]) -> tuple[float, float]:
        if df is None or len(df) < 5:
            return (0.0, 1.0)

        close = df["close"]
        high = df["high"] if "high" in df.columns else close
        low = df["low"] if "low" in df.columns else close

        try:
            ma200 = ta.SMA(df, timeperiod=200)
            rsi = ta.RSI(df, timeperiod=14)
        except Exception:
            ma200 = None
            rsi = None

        latest_close = float(close.iloc[-1])
        trend = 0.0
        if ma200 is not None and len(ma200) >= 1 and ma200.iloc[-1] not in (0, None, np.nan):
            ma_val = float(ma200.iloc[-1])
            if ma_val > 0:
                trend = (latest_close - ma_val) / ma_val

        momentum = 0.0
        if rsi is not None and len(rsi) >= 1:
            try:
                rsi_val = float(rsi.iloc[-1])
                momentum = (rsi_val - 50.0) / 50.0
            except Exception:
                momentum = 0.0

        bias = 0.6 * trend + 0.4 * momentum

        try:
            atr = ta.ATR(high=high, low=low, close=close, timeperiod=14)
            atr_pct = float(atr.iloc[-1] / max(latest_close, 1e-9)) if atr is not None else 0.0
        except Exception:
            atr_pct = 0.0
        volatility = max(0.5, min(3.0, atr_pct * 100.0))  # normalize to a comfortable 0.5~3.0 band

        return (float(bias), float(volatility))

    def analyze(self, btc_df: Optional[pd.DataFrame], eth_df: Optional[pd.DataFrame]) -> MarketSnapshot:
        """计算并写入全局市场偏置/波动率。"""

        btc_bias, btc_vol = self._safe_bias_vol(btc_df)
        eth_bias, eth_vol = self._safe_bias_vol(eth_df)

        w_btc = float(self.weights.get("BTC", 0.6) or 0.0)
        w_eth = float(self.weights.get("ETH", 0.4) or 0.0)
        total_w = max(w_btc + w_eth, 1e-9)

        bias_raw = (w_btc * btc_bias + w_eth * eth_bias) / total_w
        vol_raw = (w_btc * btc_vol + w_eth * eth_vol) / total_w

        # Altseason correction
        if btc_vol < 1.0 and eth_bias > 0.7:
            vol_raw = eth_vol * 1.5
            bias_raw = max(bias_raw, 0.8)

        bias_final = max(-1.0, min(1.0, bias_raw))
        entropy_boost = 1.0 + max(0.0, self.entropy_factor) * abs(math.sin((1 - bias_final) * (math.pi / 2)))
        vol_raw *= entropy_boost
        vol_final = max(0.5, min(3.0, vol_raw))

        try:
            self.backend.set_market_metrics(bias_final, vol_final)
        except Exception:
            # backend failures are non-fatal for sensing
            pass

        return MarketSnapshot(
            bias=bias_final,
            volatility=vol_final,
            btc_bias=btc_bias,
            eth_bias=eth_bias,
            btc_vol=btc_vol,
            eth_vol=eth_vol,
        )
