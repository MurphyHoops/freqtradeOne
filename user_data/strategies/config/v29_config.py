from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class V29Config:
    """V29 策略的核心参数集合，集中控制时间周期、资金分配与风险阈值。"""

    timeframe: str = "5m"  # 策略主时间周期（例如 5m、1h）
    startup_candle_count: int = 210  # 启动时需要的历史K线数量

    # Portfolio caps & stress adjustments
    portfolio_cap_pct_base: float = 0.20  # 组合 VaR 基准占比上限
    drawdown_threshold_pct: float = 0.15  # 进入压力模式的回撤阈值

    # Treasury controls
    treasury_fast_split_pct: float = 0.30  # 财政 fast 桶占比
    fast_topK_squads: int = 10  # fast 桶允许的信号编队数量
    slow_universe_pct: float = 0.90  # slow 桶覆盖的候选比例
    min_injection_nominal_fast: float = 30.0  # fast 桶最小名义注入（USDT）
    min_injection_nominal_slow: float = 7.0  # slow 桶最小名义注入（USDT）

    # Debt / decay
    tax_rate_on_wins: float = 0.20  # 盈利“税率”用于填补债务池
    pain_decay_per_bar: float = 0.999  # 债务/疼痛的衰减系数
    clear_debt_on_profitable_cycle: bool = True  # 盈利周期是否清空债务池
    cycle_len_bars: int = 288  # 盈利周期长度（5m*24h）

    # Early lock / breakeven guards
    breakeven_lock_frac_of_tp: float = 0.5  # 早锁盈触发比例（相对 TP）
    breakeven_lock_eps_atr_pct: float = 0.1  # 早锁盈 ATR% 偏移

    # Finalize cadence
    force_finalize_mult: float = 1.5  # finalize 超时倍数
    reservation_ttl_bars: int = 6  # 预约 TTL（bar）

    # Indicators
    ema_fast: int = 50  # 快速 EMA 长度
    ema_slow: int = 200  # 慢速 EMA 长度
    rsi_len: int = 14  # RSI 长度
    atr_len: int = 14  # ATR 长度
    adx_len: int = 14  # ADX 长度（支持动态列名）

    # Behaviour toggles
    suppress_baseline_when_stressed: bool = True  # 压力期是否抑制 baseline VaR

    # Runtime
    dry_run_wallet_fallback: float = 1000.0  # Dry-run 默认资金
    enforce_leverage: float = 1.0  # 使用的杠杆倍数


def apply_overrides(cfg: V29Config, strategy_params: Optional[Mapping[str, Any]]) -> V29Config:
    """根据 strategy_params 对默认配置进行覆盖。"""
    if not strategy_params:
        return cfg
    for key, value in strategy_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
