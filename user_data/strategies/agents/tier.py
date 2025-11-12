# -*- coding: utf-8 -*-
"""分层策略与候选过滤模块。

根据交易对的连续亏损次数（CLOSS）选择对应 TierPolicy，并在信号阶段
过滤满足当前档位约束的候选，确保风险与恢复策略匹配。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from .signal.schemas import Candidate


@dataclass
class TierPolicy:
    """定义单个 Tier 档位下的策略与风险参数集合。"""

    name: str
    allowed_kinds: set[str]
    min_raw_score: float
    min_rr_ratio: float
    min_edge: float
    sizing_algo: Literal["BASELINE", "TARGET_RECOVERY"]
    k_mult_base_pct: float
    recovery_factor: float
    cooldown_bars: int
    cooldown_bars_after_win: int
    per_pair_risk_cap_pct: float
    max_stake_notional_pct: float
    icu_force_exit_bars: int


class TierManager:
    """负责根据 CLOSS 获取合适的 TierPolicy。"""

    def __init__(self) -> None:
        """初始化三档策略参数（Healthy / Recovery / ICU）。"""
        self._t0 = TierPolicy(
            name="T0_healthy",
            allowed_kinds={
                "newbars_breakout_long_5m",
                "newbars_breakdown_short_5m",
            },
            min_raw_score=0.20,
            min_rr_ratio=0.2,
            min_edge=0.002,
            sizing_algo="BASELINE",
            k_mult_base_pct=1,
            recovery_factor=1.0,
            cooldown_bars=5,
            cooldown_bars_after_win=2,
            per_pair_risk_cap_pct=0.03,
            max_stake_notional_pct=0.15,
            icu_force_exit_bars=0,
        )
        self._t12 = TierPolicy(
            name="T12_recovery",
            allowed_kinds={
                "pullback_long",
                "trend_short",
                "newbars_breakout_long_30m",
                "newbars_breakdown_short_30m",
            },
            min_raw_score=0.15,
            min_rr_ratio=1.4,
            min_edge=0.003,
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=1,
            recovery_factor=1.5,
            cooldown_bars=10,
            cooldown_bars_after_win=4,
            per_pair_risk_cap_pct=0.02,
            max_stake_notional_pct=0.12,
            icu_force_exit_bars=30,
        )
        self._t3p = TierPolicy(
            name="T3p_ICU",
            allowed_kinds={"trend_short", "mean_rev_long"},
            min_raw_score=0.20,
            min_rr_ratio=1.6,
            min_edge=0.004,
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=1,
            recovery_factor=2.0,
            cooldown_bars=20,
            cooldown_bars_after_win=6,
            per_pair_risk_cap_pct=0.01,
            max_stake_notional_pct=0.10,
            icu_force_exit_bars=20,
        )

    def get(self, closs: int) -> TierPolicy:
        """根据连续亏损次数返回对应档位策略。"""

        if closs <= 0:
            return self._t0
        if closs <= 2:
            return self._t12
        return self._t3p


class TierAgent:
    """依据 TierPolicy 对候选集合进行过滤与择优。"""

    @staticmethod
    def filter_best(policy: TierPolicy, candidates: Sequence[Candidate]) -> Optional[Candidate]:
        """选出满足档位约束且期望收益最大的候选。"""

        ok: list[Candidate] = []
        for cand in candidates:
            if cand.kind not in policy.allowed_kinds:
                continue
            if cand.raw_score < policy.min_raw_score:
                continue
            if cand.rr_ratio < policy.min_rr_ratio:
                continue
            if cand.expected_edge < policy.min_edge:
                continue
            ok.append(cand)
        if not ok:
            return None
        ok.sort(key=lambda c: (c.expected_edge, c.raw_score), reverse=True)
        return ok[0]
