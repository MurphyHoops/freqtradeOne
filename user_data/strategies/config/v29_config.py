"""V29 策略配置模块。

该模块集中定义了 `V29Config` 数据类，用于描述 TaxBrainV29 在运行过程中可能调整的所有参数。
除了列出默认值外，还提供 `apply_overrides` 帮助方法，便于将 Freqtrade `strategy_params`
中的动态配置覆盖到数据类实例上。

示例:
    >>> from user_data.strategies.config.v29_config import V29Config, apply_overrides
    >>> cfg = V29Config()
    >>> overrides = {"timeframe": "15m", "treasury_fast_split_pct": 0.4}
    >>> cfg = apply_overrides(cfg, overrides)
    >>> cfg.timeframe
    '15m'
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class V29Config:
    """V29 策略核心参数集合。

    本数据类涵盖 TaxBrainV29 在运行时常见的全部调参项，按主题分为：
    - 时间设定：`timeframe`、`startup_candle_count`。
    - 组合风险：`portfolio_cap_pct_base`、`drawdown_threshold_pct` 等。
    - 财政调度：fast/slow 桶的占比、候选数量与最小注入额。
    - 债务衰减：盈利回扣、疼痛衰减以及盈利周期清债开关。
    - 早锁盈：触发比例及基于 ATR% 的上移距离。
    - Finalize 节奏：超时倍数与预约 TTL。
    - 指标参数：EMA/RSI/ATR/ADX 长度与压力期抑制 baseline。
    - 运行参数：dry-run 备用资金与强制杠杆倍数。

    使用示例::
        >>> from user_data.strategies.config.v29_config import V29Config, apply_overrides
        >>> cfg = V29Config()
        >>> cfg.timeframe
        '5m'
        >>> cfg = apply_overrides(cfg, {'timeframe': '15m', 'treasury_fast_split_pct': 0.4})
        >>> cfg.timeframe
        '15m'
    """

    timeframe: str = "5m"  # 策略主时间周期（例如 5m、1h）
    startup_candle_count: int = 210  # 启动时需要的历史K线数量
    informative_timeframes: tuple[str, ...] = ()  # ָ����Ҫ��������з���ļ�ʱ���롣

    # Portfolio caps & stress adjustments
    portfolio_cap_pct_base: float = 0.20  # 组合 VaR 基准占比上限
    drawdown_threshold_pct: float = 0.15  # 进入压力模式的回撤阈值，配置这个之后，债务到这里了，仓位减半

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

    # 离场
    sl_k: float = 1.6
    tp_k: float =2.0
    # Behaviour toggles
    suppress_baseline_when_stressed: bool = True  # 压力期是否抑制 baseline VaR

    # Runtime
    dry_run_wallet_fallback: float = 1000.0  # Dry-run 默认资金
    enforce_leverage: float = 1.0  # 使用的杠杆倍数


def apply_overrides(cfg: V29Config, strategy_params: Optional[Mapping[str, Any]]) -> V29Config:
    """根据提供的 `strategy_params` 字典覆盖配置实例。

    Args:
        cfg: 已创建的 `V29Config` 实例，会被就地修改并返回。
        strategy_params: 来自 Freqtrade `config['strategy_params']` 的字典，
            键需与 `V29Config` 字段同名。

    Returns:
        V29Config: 覆盖后的同一个实例，方便链式调用。

    Example:
        >>> cfg = V29Config()
        >>> apply_overrides(cfg, {'timeframe': '1h'})
        V29Config(...)
    """
    if not strategy_params:
        return cfg
    for key, value in strategy_params.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
