# -*- coding: utf-8 -*-
"""Facade that centralises exit planning and router orchestration."""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

from ...config.v29_config import V29Config
from ..tier import TierManager
from .profile_planner import ProfilePlan, atr_pct_from_dp, compute_plan_from_atr
from .router import EXIT_ROUTER


class ExitFacade:
    """Provide a single entrypoint for ATR sourcing, planning, and routing."""

    def __init__(self, cfg: V29Config, tier_mgr: TierManager, router=EXIT_ROUTER) -> None:
        self.cfg = cfg
        self.tier_mgr = tier_mgr
        self.router = router
        self.dp = None
        self.strategy = None
        self.analytics = None

    def attach_strategy(self, strategy) -> None:
        self.strategy = strategy
        self.analytics = getattr(strategy, "analytics", None)

    def set_dataprovider(self, dp) -> None:
        self.dp = dp

    # ---------- ATR helpers ----------
    def atr_pct(self, pair: str, timeframe: Optional[str], current_time) -> Optional[float]:
        """Return ATR% for the requested timeframe using cache -> DP fallback."""

        tf = timeframe or getattr(self.cfg, "timeframe", None)
        if not tf:
            return None
        strategy = self.strategy
        if strategy:
            try:
                value = strategy.get_informative_value(pair, tf, "atr_pct", None)
                if value and value > 0:
                    return float(value)
            except Exception:
                pass
        return atr_pct_from_dp(self.dp, pair, tf, current_time)

    # ---------- Planning ----------
    def resolve_trade_plan(
        self,
        pair: str,
        trade,
        current_time,
        pair_state=None,
    ) -> Tuple[Optional[str], Optional[Any], Optional[ProfilePlan]]:
        """Return (profile_name, profile_def, plan) for a live trade."""

        profile_name, profile_def = self._resolve_profile(pair, trade, pair_state)
        if not profile_name or not profile_def:
            return profile_name, None, None
        plan = self._plan_from_profile(profile_name, profile_def, pair, current_time)
        return profile_name, profile_def, plan

    def plan_for_profile(
        self, profile_name: str, pair: str, current_time
    ) -> Optional[ProfilePlan]:
        profile = self.cfg.exit_profiles.get(profile_name)
        if not profile:
            return None
        return self._plan_from_profile(profile_name, profile, pair, current_time)

    # ---------- Router glue ----------
    def apply_vector(self, df, metadata: dict, state, timeframe_col: str | None = None):
        """Apply registered vector exits and log aggregated tags."""

        if not self.router:
            return df
        result = self.router.apply_vector_exits(
            df=df,
            metadata=metadata,
            dp=self.dp,
            cfg=self.cfg,
            state=state,
            timeframe_col=timeframe_col,
            strategy=self.strategy,
        )
        pair = metadata.get("pair")
        if pair and self.analytics and hasattr(result, "__getitem__"):
            try:
                tags = result["exit_tag"]
            except Exception:
                tags = None
            if tags is not None:
                try:
                    self.analytics.log_exit_tag_series(pair, tags)
                except Exception:
                    pass
        return result

    # ---------- Internals ----------
    def _plan_from_profile(
        self,
        profile_name: str,
        profile,
        pair: str,
        current_time,
    ) -> Optional[ProfilePlan]:
        atr_tf = getattr(profile, "atr_timeframe", None) or getattr(self.cfg, "timeframe", None)
        atr_pct = self.atr_pct(pair, atr_tf, current_time)
        if atr_pct is None or atr_pct <= 0:
            return None
        return compute_plan_from_atr(profile_name, profile, atr_pct)

    def _resolve_profile(self, pair: str, trade, pair_state=None) -> Tuple[Optional[str], Optional[Any]]:
        profile = None
        if trade and hasattr(trade, "get_custom_data"):
            try:
                profile = trade.get_custom_data("exit_profile")
            except Exception:
                profile = None
        if not profile and trade:
            try:
                user_data = getattr(trade, "user_data", None)
                if user_data:
                    profile = user_data.get("exit_profile")
            except Exception:
                profile = None
        state = pair_state or self._pair_state(pair)
        tid = (
            str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
            if trade is not None
            else None
        )
        if not profile and state and getattr(state, "active_trades", None) and tid:
            meta = state.active_trades.get(tid)
            if meta:
                profile = getattr(meta, "exit_profile", None)
        if not profile and trade and getattr(trade, "entry_tag", None):
            try:
                tag = json.loads(trade.entry_tag)
                profile = tag.get("exit_profile")
            except Exception:
                profile = None
        if not profile and state:
            profile = getattr(state, "last_exit_profile", None)
        if not profile and state:
            profile = self._tier_default_profile(state)
        if not profile:
            profile = getattr(self.cfg, "default_exit_profile", None)
        if profile:
            return profile, self.cfg.exit_profiles.get(profile)
        return None, None

    def _tier_default_profile(self, pair_state) -> Optional[str]:
        if not pair_state:
            return None
        closs = getattr(pair_state, "closs", None)
        if closs is None:
            return None
        try:
            policy = self.tier_mgr.get(closs)
        except Exception:
            return None
        return getattr(policy, "default_exit_profile", None)

    def _pair_state(self, pair: str):
        state = getattr(self.strategy, "state", None) if self.strategy else None
        if not state or not hasattr(state, "get_pair_state"):
            return None
        try:
            return state.get_pair_state(pair)
        except Exception:
            return None


__all__ = ["ExitFacade"]
