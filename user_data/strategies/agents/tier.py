from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from .signal import Candidate


@dataclass
class TierPolicy:
    """TierPolicy 的职责说明。"""
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
    """TierManager 的职责说明。"""
    def __init__(self) -> None:
        """处理 __init__ 的主要逻辑。"""
        self._t0 = TierPolicy(
            name="T0_healthy",
            allowed_kinds={"mean_rev_long", "pullback_long", "trend_short"},
            min_raw_score=0.20,
            min_rr_ratio=1.2,
            min_edge=0.002,
            sizing_algo="BASELINE",
            k_mult_base_pct=0.005,
            recovery_factor=1.0,
            cooldown_bars=5,
            cooldown_bars_after_win=2,
            per_pair_risk_cap_pct=0.03,
            max_stake_notional_pct=0.15,
            icu_force_exit_bars=0,
        )
        self._t12 = TierPolicy(
            name="T12_recovery",
            allowed_kinds={"pullback_long", "trend_short"},
            min_raw_score=0.15,
            min_rr_ratio=1.4,
            min_edge=0.003,
            sizing_algo="TARGET_RECOVERY",
            k_mult_base_pct=0.004,
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
            k_mult_base_pct=0.003,
            recovery_factor=2.0,
            cooldown_bars=20,
            cooldown_bars_after_win=6,
            per_pair_risk_cap_pct=0.01,
            max_stake_notional_pct=0.10,
            icu_force_exit_bars=20,
        )

    def get(self, closs: int) -> TierPolicy:
        """处理 get 的主要逻辑。"""
        if closs <= 0:
            return self._t0
        if closs <= 2:
            return self._t12
        return self._t3p


class TierAgent:
    """Tier-aware candidate filtering."""

    @staticmethod
    def filter_best(policy: TierPolicy, candidates: Sequence[Candidate]) -> Optional[Candidate]:
        """处理 filter_best 的主要逻辑。"""
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
