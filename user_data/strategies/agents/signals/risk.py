# -*- coding: utf-8 -*-
"""Risk planning utilities that derive SL/TP from exit profiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from ...config.v29_config import ExitProfile, StrategyRecipe, V29Config
from ..exits.profiles import compute_plan_from_atr
from .factors import FactorBank, apply_timeframe_to_factor


@dataclass(frozen=True)
class RiskPlan:
    """Replication of the computed planning values for a candidate."""

    sl_pct: float
    tp_pct: float
    exit_profile: str
    recipe: Optional[str] = None
    min_rr: Optional[float] = None
    min_edge: Optional[float] = None
    plan_timeframe: Optional[str] = None
    plan_atr_pct: Optional[float] = None


class RiskEstimator:
    """Translate exit profiles into usable SL/TP planning values."""

    def __init__(self, cfg: V29Config) -> None:
        self._cfg = cfg
        self._profiles: Dict[str, ExitProfile] = dict(
            getattr(getattr(cfg, "strategy", None), "exit_profiles", getattr(cfg, "exit_profiles", {})) or {}
        )
        self._entry_to_recipe: Dict[str, StrategyRecipe] = {}
        for recipe in getattr(cfg, "strategy_recipes", ()) or ():
            if not isinstance(recipe, StrategyRecipe):
                continue
            for entry in recipe.entries:
                self._entry_to_recipe[entry] = recipe
        default_name = getattr(getattr(cfg, "strategy", None), "default_exit_profile", getattr(cfg, "default_exit_profile", None))
        if not default_name or default_name not in self._profiles:
            default_name = next(iter(self._profiles.keys()), None)
        self._default_exit_profile = default_name

    def plan(self, entry: str, timeframe: Optional[str], fb: FactorBank, bag) -> Optional[RiskPlan]:
        """Return RiskPlan for the given entry if a recipe+profile exists."""

        recipe = self._entry_to_recipe.get(entry)
        if recipe:
            plan = self._plan_with_profile(
                profile_name=recipe.exit_profile,
                timeframe=timeframe,
                fb=fb,
                bag=bag,
                recipe_name=recipe.name,
                recipe_min_rr=recipe.min_rr,
                recipe_min_edge=recipe.min_edge,
            )
            if plan:
                return plan

        if self._default_exit_profile:
            return self._plan_with_profile(
                profile_name=self._default_exit_profile,
                timeframe=timeframe,
                fb=fb,
                bag=bag,
                recipe_name=None,
                recipe_min_rr=None,
                recipe_min_edge=None,
            )
        return None

    def _plan_with_profile(
        self,
        profile_name: Optional[str],
        timeframe: Optional[str],
        fb: FactorBank,
        bag,
        recipe_name: Optional[str],
        recipe_min_rr: Optional[float],
        recipe_min_edge: Optional[float],
    ) -> Optional[RiskPlan]:
        if not profile_name:
            return None
        profile = self._profiles.get(profile_name)
        if not profile:
            return None
        atr_pct = self._resolve_atr_pct(profile, timeframe, fb, bag)
        if atr_pct is None or atr_pct <= 0:
            return None

        plan = compute_plan_from_atr(profile_name, profile, atr_pct)
        if not plan or plan.sl_pct <= 0 or plan.tp_pct <= 0:
            return None

        return RiskPlan(
            sl_pct=float(plan.sl_pct),
            tp_pct=float(plan.tp_pct),
            exit_profile=profile_name,
            recipe=recipe_name,
            min_rr=recipe_min_rr,
            min_edge=recipe_min_edge,
            plan_timeframe=getattr(plan, "timeframe", None),
            plan_atr_pct=getattr(plan, "atr_pct", None),
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

__all__ = ["RiskEstimator", "RiskPlan"]
