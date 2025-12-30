# -*- coding: utf-8 -*-
"""Treasury agent that converts debt pool and market metrics into allocation caps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...config.v30_config import V30Config
from .global_backend import GlobalRiskBackend


@dataclass
class AllocationPlan:
    k_long: float = 0.0
    k_short: float = 0.0
    theta: float = 0.0
    final_r: float = 0.0
    available: float = 0.0
    bias: float = 0.0
    volatility: float = 1.0


def _dynamic_portfolio_cap_pct(cfg: V30Config, debt_pool: float, equity: float) -> float:
    """Adjust portfolio cap percentage based on debt pressure."""

    risk_cfg = getattr(cfg, "risk", cfg)
    base = risk_cfg.portfolio_cap_pct_base
    if equity <= 0:
        return base * 0.5
    if (debt_pool / equity) > risk_cfg.drawdown_threshold_pct:
        return base * 0.5
    return base


class TreasuryAgent:
    """Compute UEOT allocation ratios K_long / K_short."""

    def __init__(
        self,
        cfg: V30Config,
        tier_mgr=None,
        backend: Optional[GlobalRiskBackend] = None,
    ) -> None:
        self.cfg = cfg
        self.tier_mgr = tier_mgr
        self.backend = backend

    def _extract_backend_snapshot(self) -> tuple[float, float, float]:
        snap = self.backend.get_snapshot() if self.backend else None
        bias = float(getattr(snap, "market_bias", 0.0) or 0.0)
        vol = float(getattr(snap, "market_volatility", 1.0) or 1.0)
        debt = float(getattr(snap, "debt_pool", 0.0) or 0.0)
        return (bias, vol, debt)

    def plan(self, state_snapshot: Dict[str, Any], equity: float) -> AllocationPlan:
        """Compute allocation ratios for the current cycle."""

        backend_bias, backend_vol, backend_debt = self._extract_backend_snapshot()

        debt_pool = float(state_snapshot.get("debt_pool", 0.0))
        if backend_debt > 0:
            debt_pool = backend_debt

        total_open_risk = float(state_snapshot.get("total_open_risk", 0.0))
        reserved_portfolio = float(state_snapshot.get("reserved_portfolio_risk", 0.0))

        cap_pct = _dynamic_portfolio_cap_pct(self.cfg, debt_pool, equity)
        portfolio_cap = cap_pct * equity
        used_cap = total_open_risk + reserved_portfolio
        free_cap = max(0.0, portfolio_cap - used_cap)

        debt_cap_pct = getattr(getattr(self.cfg, "trading", None), "treasury", getattr(self.cfg, "treasury", None)).debt_pool_cap_pct
        available_from_debt = min(debt_pool, debt_cap_pct * equity)
        available = min(available_from_debt, free_cap)

        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        aggressiveness = float(getattr(risk_cfg, "aggressiveness", 0.2) or 0.0)
        entropy_factor = float(getattr(risk_cfg, "entropy_factor", 0.0) or 0.0)
        vol_factor = float(getattr(risk_cfg, "volatility_factor", 1.0) or 1.0)

        bias = max(-1.0, min(1.0, backend_bias))
        volatility = max(0.5, min(3.0, backend_vol * vol_factor))
        pairs_data = state_snapshot.get("pairs", {}) or {}
        max_concurrent = max(1, len(pairs_data) or 1)

        raw_r = available * aggressiveness * volatility / max_concurrent if max_concurrent > 0 else 0.0
        safety_r = equity * 0.05
        final_r = min(raw_r, safety_r)

        theta = (1 - bias) * (math.pi / 2)
        chaos = final_r * entropy_factor * abs(math.sin(theta))

        base_long = max(0.0, final_r * max(math.cos(theta), 0.0))
        base_short = max(0.0, final_r * max(math.cos(math.pi - theta), 0.0))

        k_long = base_long + (chaos * 0.5)
        k_short = base_short + (chaos * 0.5)

        return AllocationPlan(
            k_long=k_long,
            k_short=k_short,
            theta=theta,
            final_r=final_r,
            available=available,
            bias=bias,
            volatility=volatility,
        )

    def evaluate_signal_quality(self, pair: str, score: float, closs: int) -> dict:
        """Simplified gatekeeper based on unified min_score (fast/slow buckets removed)."""

        gcfg = getattr(getattr(self.cfg, "risk", None), "gatekeeping", None)
        if not gcfg or not getattr(gcfg, "enabled", True):
            return {"allowed": True, "reason": "gatekeeping_disabled", "thresholds": {"min_score": 0.0}}

        min_score = float(getattr(gcfg, "min_score", 0.0) or 0.0)
        debt = 0.0
        try:
            debt = float(getattr(self.backend.get_snapshot(), "debt_pool", 0.0)) if self.backend else 0.0
        except Exception:
            debt = 0.0

        thresholds: Dict[str, float] = {"min_score": min_score}

        if score >= min_score:
            return {
                "allowed": True,
                "reason": "score_ok",
                "thresholds": thresholds,
                "score": score,
                "closs": closs,
                "debt": debt,
            }
        return {
            "allowed": False,
            "reason": f"Score<{min_score:.2f}",
            "thresholds": thresholds,
            "score": score,
            "closs": closs,
            "debt": debt,
        }
