# -*- coding: utf-8 -*-
"""信号插件所使用的统一数据契约。

该模块定义了候选信号、触发条件以及完整信号规格的数据结构，
用于在指标计算、信号注册中心与候选生成器之间传递结构化信息。
所有字段均保持与 V29.1 版本兼容，确保拆分后模块仍能与原策略协同工作。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional

Direction = Literal["long", "short"]


@dataclass(frozen=True)
class Candidate:
    """标准化后的建仓候选描述。

    Attributes:
        direction: 信号方向，"long" 表示做多，"short" 表示做空。
        kind: 信号名称（注册时的唯一标识），供风控和分层策略引用。
        sl_pct: 建议的止损百分比，例如 0.02 表示 2% 风险。
        tp_pct: 建议的止盈百分比。
        raw_score: 信号原始评分，用于排序或调试。
        rr_ratio: Reward/Risk 比值，tp_pct / sl_pct。
        win_prob: 预估胜率，范围 0~1。
        expected_edge: 期望收益，胜率与盈亏比换算后得到。
        squad: 所属战术小队名称，财政分配与分层策略会按该字段过滤。
    """

    direction: Direction
    kind: str
    sl_pct: float
    tp_pct: float
    raw_score: float
    rr_ratio: float
    win_prob: float
    expected_edge: float
    squad: str
    exit_profile: Optional[str] = None
    recipe: Optional[str] = None


@dataclass(frozen=True)
class Condition:
    """信号触发条件的声明式表示。

    Attributes:
        factor: 参与比较的因子名称，由 :class:FactorBank 提供。
        op: 条件运算符，可取 "<"、"<="、">"、">="、"=="、"between"、"outside" 等。
        value: 比较阈值；当 op 为区间类操作时代表下界。
        value_hi: 区间上界，仅在 op 为 "between" 或 "outside" 时使用。
        fn: 可选的自定义函数，接收因子数值返回布尔值，适合复杂逻辑。
    """

    factor: str
    op: Optional[str] = None
    value: Optional[float] = None
    value_hi: Optional[float] = None
    fn: Optional[Callable[[float], bool]] = None


@dataclass(frozen=True)
class SignalSpec:
    """注册中心记录的完整信号定义。"""
    name: str
    direction: Direction
    squad: str
    conditions: List[Condition]
    sl_fn: Callable[[Dict[str, float], Any], float]
    tp_fn: Callable[[Dict[str, float], Any], float]
    raw_fn: Callable[[Dict[str, float], Any], float]
    win_prob_fn: Callable[[Dict[str, float], Any, float], float]
    min_rr: float = 0.0
    min_edge: float = 0.0
    required_factors: tuple[str, ...] = ()
    timeframe: Optional[str] = None
