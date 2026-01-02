from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class EquityProvider:
    def __init__(self, initial_equity: float) -> None:
        self.equity_current = float(initial_equity)

    def to_snapshot(self) -> Dict[str, float]:
        return {"equity_current": self.equity_current}

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        self.equity_current = float(payload.get("equity_current", self.equity_current))

    def get_equity(self) -> float:
        return self.equity_current

    def on_trade_closed_update(self, profit_abs: float) -> None:
        self.equity_current += float(profit_abs)


@dataclass
class ActiveTradeMeta:
    sl_pct: float
    tp_pct: float
    direction: str
    entry_bar_tick: int
    entry_price: float
    bucket: str
    icu_bars_left: Optional[int]
    exit_profile: Optional[str] = None
    recipe: Optional[str] = None
    plan_timeframe: Optional[str] = None
    plan_atr_pct: Optional[float] = None
    tier_name: Optional[str] = None


@dataclass
class PairState:
    closs: int = 0
    local_loss: float = 0.0
    cooldown_bars_left: int = 0
    last_dir: Optional[str] = None
    last_score: float = 0.0
    last_kind: Optional[str] = None
    last_squad: Optional[str] = None
    last_sl_pct: float = 0.0
    last_tp_pct: float = 0.0
    last_atr_pct: float = 0.0
    last_exit_profile: Optional[str] = None
    last_recipe: Optional[str] = None
    active_trades: Dict[str, ActiveTradeMeta] = field(default_factory=dict)


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
    current_time: Optional[Any] = None
