from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingContext:
    """Payload passed into SizerAgent.compute."""

    pair: str
    sl_pct: float
    tp_pct: float
    direction: str
    min_stake: float = 0.0      # exchange / Freqtrade 提供的最小保证金
    max_stake: float = 0.0      # exchange / Freqtrade 提供的最大保证金（可选）
    leverage: float = 1.0       # 当前交易的杠杆，用于名义 <-> 保证金转换
    proposed_stake: Optional[float] = None
    plan_atr_pct: Optional[float] = None
    exit_profile: Optional[str] = None
    bucket: Optional[str] = None
    current_rate: float = 0.0
    score: float = 0.0
