# -*- coding: utf-8 -*-
"""Facade that centralises exit planning and router orchestration."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

from ...config.v30_config import V30Config
from ..portfolio.tier import TierManager
from .profiles import ProfilePlan, atr_pct_from_dp, atr_pct_from_rows, compute_plan_from_atr
from .router import EXIT_ROUTER, ImmediateContext


class ExitTags:
    """Static string constants for exit reasons / tags."""

    CLOSE_FILLED = "close_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TP_HIT = "tp_hit"
    ICU_TIMEOUT = "icu_timeout"
    RISK_OFF = "risk_off"
    BREAKEVEN = "breakeven_lock"
    ATR_TRAIL = "atr_trail_follow"
    HARD_STOP = "hard_stop"
    HARD_TP = "hard_takeprofit"
    FLIP_PREFIX = "flip_"
    VECTOR_TAGS = {
        TP_HIT,
        ICU_TIMEOUT,
        RISK_OFF,
        BREAKEVEN,
        ATR_TRAIL,
        HARD_STOP,
        HARD_TP,
    }

    @staticmethod
    def flip(direction: str | None) -> str:
        """Return flip tag for the provided direction (long/short)."""

        token = (direction or "").strip().lower()
        if token not in {"long", "short"}:
            token = "unknown"
        return f"{ExitTags.FLIP_PREFIX}{token}"

    @classmethod
    def is_vector_tag(cls, tag: str | None) -> bool:
        """Return True if the tag belongs to vector exit instrumentation."""

        if not tag:
            return False
        return tag in cls.VECTOR_TAGS or tag.startswith(cls.FLIP_PREFIX)


class ExitFacade:
    """Provide a single entrypoint for ATR sourcing, planning, and routing."""

    def __init__(self, cfg: V30Config, tier_mgr: TierManager, router=EXIT_ROUTER) -> None:
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
        """Backward-compatible ATR% lookup (delegates to authoritative source)."""

        return self.get_authoritative_atr(pair, timeframe, current_time)

    def get_authoritative_atr(
        self, pair: str, timeframe: Optional[str], current_time
    ) -> Optional[float]:
        """Return the canonical ATR% for sizing/exit decisions."""

        tf = timeframe or getattr(getattr(self.cfg, "system", None), "timeframe", getattr(self.cfg, "timeframe", None))
        if not tf:
            return None

        provider = self.dp or getattr(self.strategy, "dp", None)
        dp_value = atr_pct_from_dp(provider, pair, tf, current_time)
        if dp_value and dp_value > 0:
            return float(dp_value)

        strategy = self.strategy
        if strategy:
            try:
                value = strategy.get_informative_value(pair, tf, "atr_pct", None)
                if value and value > 0:
                    return float(value)
            except Exception:
                pass
        if strategy and getattr(strategy, "state", None):
            try:
                pst = strategy.state.get_pair_state(pair)
                val = float(getattr(pst, "last_atr_pct", 0.0))
                if val > 0 and math.isfinite(val):
                    return val
            except Exception:
                pass
        return dp_value

    # ---------- Planning ----------
    def resolve_entry_plan(
        self,
        pair: str,
        candidate,
        main_row,
        informative_rows: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Any], Optional[ProfilePlan]]:
        """Resolve the profile plan for a fresh candidate before order placement."""

        if not candidate:
            return None, None, None
        profiles = self._profiles()
        profile_name = getattr(candidate, "exit_profile", None)
        profile_def = profiles.get(profile_name) if profile_name else None
        pair_state = self._pair_state(pair)
        if not profile_def and pair_state:
            fallback = self._tier_default_profile(pair_state)
            profile_def = profiles.get(fallback) if fallback else None
            profile_name = fallback
        if not profile_def:
            fallback = self._default_profile_name()
            profile_def = profiles.get(fallback) if fallback else None
            profile_name = fallback
        if not profile_def:
            return profile_name, None, None

        tf_hint = getattr(candidate, "plan_timeframe", None)
        atr_hint = getattr(candidate, "plan_atr_pct", None)
        if atr_hint is None and main_row is not None:
            atr_hint = atr_pct_from_rows(
                main_row,
                informative_rows or {},
                target_timeframe=tf_hint or getattr(profile_def, "atr_timeframe", None),
                main_timeframe=getattr(getattr(self.cfg, "system", None), "timeframe", getattr(self.cfg, "timeframe", None)),
            )

        current_time = getattr(main_row, "name", None) if main_row is not None else None
        plan = self._plan_from_profile(
            profile_name,
            profile_def,
            pair,
            current_time,
            plan_timeframe=tf_hint,
            atr_hint=atr_hint,
        )
        return profile_name, profile_def, plan

    def resolve_trade_plan(
        self,
        pair: str,
        trade,
        current_time,
        pair_state=None,
    ) -> Tuple[Optional[str], Optional[Any], Optional[ProfilePlan]]:
        """Return (profile_name, profile_def, plan) for a live trade."""

        profile_name, profile_def, meta = self._resolve_profile(pair, trade, pair_state)
        if not profile_name or not profile_def:
            return profile_name, None, None
        tf_hint, atr_hint = self._plan_hints_from_sources(trade, meta, pair_state or self._pair_state(pair))
        plan = self._plan_from_profile(
            profile_name,
            profile_def,
            pair,
            current_time,
            plan_timeframe=tf_hint,
            atr_hint=atr_hint,
        )
        return profile_name, profile_def, plan

    def plan_for_profile(
        self, profile_name: str, pair: str, current_time
    ) -> Optional[ProfilePlan]:
        profiles = self._profiles()
        profile = profiles.get(profile_name)
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

    # ---------- Immediate exits ----------
    def decide_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit_pct: Optional[float],
        pair_state=None,
    ) -> Optional[str]:
        trade_id = str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
        profit = float(current_profit_pct or 0.0) if current_profit_pct is not None else 0.0

        if self.router is not None and ImmediateContext is not None:
            try:
                ctx = ImmediateContext(
                    pair=pair,
                    trade=trade,
                    now=current_time,
                    profit=profit,
                    dp=self.dp,
                    cfg=self.cfg,
                    state=getattr(self.strategy, "state", None),
                    strategy=self.strategy,
                )
                reason = self.router.close_now_reason(ctx)
                if reason:
                    return reason
            except Exception:
                pass

        state = getattr(self.strategy, "state", None)
        pst = pair_state or (state.get_pair_state(pair) if state else None)
        meta = None
        if pst and getattr(pst, "active_trades", None):
            meta = pst.active_trades.get(trade_id)

        if meta:
            if (
                getattr(meta, "tp_pct", None)
                and current_profit_pct is not None
                and current_profit_pct >= float(meta.tp_pct)
            ):
                return ExitTags.TP_HIT
            if meta.icu_bars_left is not None and meta.icu_bars_left <= 0:
                return ExitTags.ICU_TIMEOUT
            if (
                pst
                and pst.last_dir
                and pst.last_dir != meta.direction
                and getattr(pst, "last_score", 0.0) > 0.0
            ):
                return ExitTags.flip(pst.last_dir)

        if current_profit_pct is not None and current_profit_pct < 0 and state is not None:
            if float(getattr(state, "debt_pool", 0.0)) > 0.0:
                return ExitTags.RISK_OFF

        return self._atr_entry_exit_reason(pair, trade, current_time, current_rate, pair_state=pst)

    # ---------- Internals ----------
    def _plan_from_profile(
        self,
        profile_name: str,
        profile,
        pair: str,
        current_time,
        *,
        plan_timeframe: Optional[str] = None,
        atr_hint: Optional[float] = None,
    ) -> Optional[ProfilePlan]:
        atr_tf = (
            self._normalize_tf(plan_timeframe)
            or self._normalize_tf(getattr(profile, "atr_timeframe", None))
            or self._normalize_tf(getattr(getattr(self.cfg, "system", None), "timeframe", getattr(self.cfg, "timeframe", None)))
        )
        atr_pct = self._coerce_positive_float(atr_hint)
        if atr_pct is None or atr_pct <= 0:
            atr_pct = self.atr_pct(pair, atr_tf, current_time)
        if atr_pct is None or atr_pct <= 0:
            return None
        return compute_plan_from_atr(profile_name, profile, atr_pct)

    def _atr_entry_exit_reason(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        pair_state=None,
    ) -> Optional[str]:
        if trade is None:
            return None
        try:
            _, _, plan = self.resolve_trade_plan(pair, trade, current_time, pair_state=pair_state)
        except Exception:
            plan = None
        if not plan:
            return None
        sl_pct = float(getattr(plan, "sl_pct", 0.0) or 0.0)
        tp_pct = float(getattr(plan, "tp_pct", 0.0) or 0.0)
        if sl_pct <= 0 and tp_pct <= 0:
            return None

        entry = float(getattr(trade, "open_rate", 0.0) or 0.0)
        if entry <= 0:
            return None
        price = float(current_rate or entry)
        is_short = bool(getattr(trade, "is_short", False))

        if not is_short:
            if sl_pct > 0:
                sl_price = entry * (1.0 - sl_pct)
                if price <= sl_price:
                    return "atr_entry_sl"
            if tp_pct > 0:
                tp_price = entry * (1.0 + tp_pct)
                if price >= tp_price:
                    return "atr_entry_tp"
        else:
            if sl_pct > 0:
                sl_price = entry * (1.0 + sl_pct)
                if price >= sl_price:
                    return "atr_entry_sl"
            if tp_pct > 0:
                tp_price = entry * (1.0 - tp_pct)
                if price <= tp_price:
                    return "atr_entry_tp"
        return None

    def _resolve_profile(
        self, pair: str, trade, pair_state=None
    ) -> Tuple[Optional[str], Optional[Any], Optional[Any]]:
        profile = self._coerce_str(self._read_trade_custom(trade, "exit_profile"))
        state = pair_state or self._pair_state(pair)
        tid = (
            str(getattr(trade, "trade_id", getattr(trade, "id", "NA")))
            if trade is not None
            else None
        )
        meta = None
        if state and getattr(state, "active_trades", None) and tid:
            meta = state.active_trades.get(tid)
            if meta and not profile:
                profile = self._coerce_str(getattr(meta, "exit_profile", None))
        if not profile and state:
            profile = self._tier_default_profile(state)
        if not profile:
            profile = self._default_profile_name()
        if profile:
            profiles = self._profiles()
            return profile, profiles.get(profile), meta
        return None, None, meta

    def _plan_hints_from_sources(self, trade, meta, pair_state) -> Tuple[Optional[str], Optional[float]]:
        tf_hint = self._coerce_str(self._read_trade_custom(trade, "plan_timeframe"))
        atr_hint = self._coerce_positive_float(self._read_trade_custom(trade, "atr_pct"))
        if meta:
            if not tf_hint:
                tf_hint = self._coerce_str(getattr(meta, "plan_timeframe", None))
            if atr_hint is None:
                atr_hint = self._coerce_positive_float(getattr(meta, "plan_atr_pct", None))
        provider_available = self.dp or getattr(self.strategy, "dp", None)
        if atr_hint is None and pair_state is not None and not provider_available:
            atr_hint = self._coerce_positive_float(getattr(pair_state, "last_atr_pct", None))
        return tf_hint, atr_hint

    @staticmethod
    def _read_trade_custom(trade, key: str):
        if not trade or not hasattr(trade, "get_custom_data"):
            return None
        try:
            return trade.get_custom_data(key)
        except Exception:
            return None

    @staticmethod
    def _coerce_str(value) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_positive_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if num <= 0:
            return None
        return num

    def _profiles(self) -> Dict[str, Any]:
        return (
            getattr(getattr(self.cfg, "strategy", None), "exit_profiles", getattr(self.cfg, "exit_profiles", {}))
            or {}
        )

    def _default_profile_name(self) -> Optional[str]:
        return getattr(
            getattr(self.cfg, "strategy", None),
            "default_exit_profile",
            getattr(self.cfg, "default_exit_profile", None),
        )

    def _normalize_tf(self, timeframe: Optional[str]) -> Optional[str]:
        if timeframe is None:
            return None
        token = str(timeframe).strip()
        if not token:
            return None
        lowered = token.lower()
        if lowered in {"primary", "main", "base"}:
            return None
        return token

    def _tier_default_profile(self, pair_state) -> Optional[str]:
        if not pair_state:
            return None
        closs = getattr(pair_state, "closs", None)
        if closs is None:
            return None
        return self.tier_mgr.default_profile_for_closs(closs)

    def _pair_state(self, pair: str):
        state = getattr(self.strategy, "state", None) if self.strategy else None
        if not state or not hasattr(state, "get_pair_state"):
            return None
        try:
            return state.get_pair_state(pair)
        except Exception:
            return None


__all__ = ["ExitFacade", "ExitTags"]
