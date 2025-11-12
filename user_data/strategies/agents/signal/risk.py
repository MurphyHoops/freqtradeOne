# -*- coding: utf-8 -*-
"""Risk planning utilities that derive SL/TP from exit profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ...config.v29_config import ExitProfile, StrategyRecipe, V29Config
from .factor_spec import apply_timeframe_to_factor
from .factors import FactorBank


@dataclass(frozen=True)
class RiskPlan:
    """Replication of the computed planning values for a candidate."""

    sl_pct: float
    tp_pct: float
    exit_profile: str
    recipe: Optional[str] = None


class RiskEstimator:
    """Translate exit profiles into usable SL/TP planning values."""

    def __init__(self, cfg: V29Config) -> None:
        self._cfg = cfg
        self._profiles: Dict[str, ExitProfile] = dict(getattr(cfg, "exit_profiles", {}) or {})
        self._entry_to_recipe: Dict[str, StrategyRecipe] = {}
        for recipe in getattr(cfg, "strategy_recipes", ()) or ():
            if not isinstance(recipe, StrategyRecipe):
                continue
            for entry in recipe.entries:
                self._entry_to_recipe[entry] = recipe

    def plan(self, entry: str, timeframe: Optional[str], fb: FactorBank, bag) -> Optional[RiskPlan]:
        """Return RiskPlan for the given entry if a recipe+profile exists."""

        recipe = self._entry_to_recipe.get(entry)
        if not recipe:
            return None
        profile = self._profiles.get(recipe.exit_profile)
        if not profile:
            return None
        atr_pct = self._resolve_atr_pct(profile, timeframe, fb, bag)
        if atr_pct is None or atr_pct <= 0:
            return None

        sl = self._compute_sl(profile, atr_pct)
        tp = self._compute_tp(profile, atr_pct, sl)
        if sl <= 0 or tp <= 0:
            return None

        return RiskPlan(
            sl_pct=float(sl),
            tp_pct=float(tp),
            exit_profile=recipe.exit_profile,
            recipe=recipe.name,
        )

    def _resolve_atr_pct(
        self,
        profile: ExitProfile,
        spec_timeframe: Optional[str],
        fb: FactorBank,
        bag,
    ) -> Optional[float]:
        target_tf = profile.atr_timeframe or spec_timeframe
        value: Optional[float]
        try:
            if target_tf is None or target_tf == spec_timeframe:
                value = float(bag["ATR_PCT"])
            else:
                resolved = apply_timeframe_to_factor("ATR_PCT", target_tf)
                value = float(fb.get(resolved))
        except (KeyError, ValueError, TypeError):
            return None
        if math.isnan(value) or math.isinf(value) or value <= 0:
            return None
        return value

    @staticmethod
    def _compute_sl(profile: ExitProfile, atr_pct: float) -> float:
        k = profile.atr_mul_sl if profile.atr_mul_sl and profile.atr_mul_sl > 0 else 0.0
        floor = profile.floor_sl_pct or 0.0
        sl = atr_pct * k
        return max(floor, sl)

    @staticmethod
    def _compute_tp(profile: ExitProfile, atr_pct: float, sl: float) -> float:
        if profile.atr_mul_tp and profile.atr_mul_tp > 0:
            return atr_pct * profile.atr_mul_tp
        # Fallback: maintain at least 2:1 reward-risk
        return max(sl * 2.0, 0.0)


__all__ = ["RiskEstimator", "RiskPlan"]
