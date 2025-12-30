# -*- coding: utf-8 -*-
"""Shared helper routines used by thin strategy wrappers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import math
import uuid
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from user_data.strategies.agents.signals import indicators, schemas, factors
from user_data.strategies.core.rejections import RejectReason

try:
    from freqtrade.strategy import informative
except Exception:  # pragma: no cover
    def informative(_timeframe: str):  # type: ignore
        def decorator(func):
            return func
        return decorator


def tf_to_sec(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    return 300


def timeframe_suffix_token(timeframe: Optional[str]) -> str:
    token = (timeframe or "").strip()
    return token.replace("/", "_") if token else ""


def derive_informative_timeframes(
    manual: tuple[str, ...], inferred: tuple[str, ...], base_timeframe: str
) -> tuple[str, ...]:
    ordered: list[str] = []
    for tf in (*manual, *inferred):
        if not tf or tf == base_timeframe:
            continue
        normalized = str(tf)
        if normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)


def ensure_informative_method(cls, timeframe: str) -> None:
    func_name = f"populate_indicators_{timeframe.replace('/', '_')}"
    if hasattr(cls, func_name):
        return

    @informative(timeframe)
    def _informative_populator(self, dataframe, metadata, tf=timeframe):
        needs = getattr(self.__class__, "_indicator_requirements_map", {}).get(tf)
        return indicators.compute_indicators(
            dataframe,
            self.cfg,
            suffix=None,
            required=needs or set(),
            duplicate_ohlc=True,
        )

    _informative_populator.__name__ = func_name
    setattr(cls, func_name, _informative_populator)


def register_informative_methods(strategy) -> None:
    if not getattr(strategy, "_informative_timeframes", None):
        return
    for tf in strategy._informative_timeframes:
        ensure_informative_method(strategy.__class__, tf)


def gc_informative_cache(strategy, current_whitelist: List[str]) -> None:
    if not current_whitelist:
        return
    allowed = {str(pair) for pair in current_whitelist if pair}
    if not allowed:
        return
    stale_pairs = [pair for pair in list(strategy._informative_cache.keys()) if pair not in allowed]
    for pair in stale_pairs:
        strategy._informative_cache.pop(pair, None)
        strategy._informative_last.pop(pair, None)


def get_informative_dataframe(strategy, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    cached = strategy._informative_cache.get(pair, {}).get(timeframe)
    if cached is not None and not cached.empty:
        return cached
    getter = getattr(strategy.dp, "get_informative_dataframe", None)
    if callable(getter):
        result = getter(pair, timeframe)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, pd.DataFrame) and not result.empty:
            prepared = strategy._prepare_informative_frame(result, timeframe)
            strategy._informative_cache.setdefault(pair, {})[timeframe] = prepared
            return prepared
        return strategy._prepare_informative_frame(result, timeframe)
    result = strategy.dp.get_analyzed_dataframe(pair, timeframe)
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, pd.DataFrame) and not result.empty:
        prepared = strategy._prepare_informative_frame(result, timeframe)
        strategy._informative_cache.setdefault(pair, {})[timeframe] = prepared
        return prepared
    return strategy._prepare_informative_frame(result, timeframe)


def informative_required_columns(strategy, timeframe: str) -> list[str]:
    required: set[str] = set()
    factors_needed = strategy._factor_requirements.get(timeframe, set())
    derived_base_cols = {
        "DELTA_CLOSE_EMAFAST_PCT": {"close", "ema_fast"},
        "EMA_TREND": {"ema_fast", "ema_slow"},
    }
    for base in factors_needed:
        base = str(base).upper()
        if base in factors.BASE_FACTOR_SPECS:
            required.add(factors.BASE_FACTOR_SPECS[base].column)
        elif base in factors.DERIVED_FACTOR_SPECS:
            required.update(derived_base_cols.get(base, set()))
            for indicator_name in factors.DERIVED_FACTOR_SPECS[base].indicators:
                spec = indicators.INDICATOR_SPECS.get(str(indicator_name).upper())
                if spec:
                    required.update(spec.columns)
    indicator_needs = strategy._indicator_requirements.get(timeframe, set())
    for indicator_name in indicator_needs:
        spec = indicators.INDICATOR_SPECS.get(str(indicator_name).upper())
        if spec:
            required.update(spec.columns)
    return sorted(required)


def derived_factor_columns_missing(df: pd.DataFrame, timeframes: Iterable[Optional[str]]) -> bool:
    if df is None or df.empty:
        return False
    for tf in timeframes:
        suffix = (tf or "").strip().replace("/", "_")
        for base in ("delta_close_emafast_pct", "ema_trend"):
            col = f"{base}_{suffix}" if suffix else base
            if col not in df.columns:
                return True
    return False


def merge_informative_columns_into_base(strategy, df: pd.DataFrame, pair: str) -> None:
    if df is None or df.empty:
        return
    base_time = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    left = pd.DataFrame({"_t": base_time})
    for tf in getattr(strategy, "_informative_timeframes", []):
        info_df = strategy._get_informative_dataframe(pair, tf)
        if info_df is None or info_df.empty:
            continue
        cols = informative_required_columns(strategy, tf)
        cols = [col for col in cols if col in info_df.columns]
        if not cols:
            continue
        if "date" in info_df.columns and "date" not in cols:
            cols.append("date")
        suffix = str(tf).replace("/", "_")
        missing = [col for col in cols if f"{col}_{suffix}" not in df.columns]
        if not missing:
            continue
        right = info_df.loc[:, cols]
        right = right.assign(
            _tinfo=pd.to_datetime(right["date"]) if "date" in right.columns else pd.to_datetime(right.index)
        )
        merged = pd.merge_asof(
            left.sort_values("_t"),
            right.sort_values("_tinfo"),
            left_on="_t",
            right_on="_tinfo",
            direction="backward",
        )
        merged.index = df.index
        for col in cols:
            target = f"{col}_{suffix}"
            if target in df.columns:
                continue
            if col in merged.columns:
                df[target] = merged[col]


def aligned_informative_for_df(strategy, pair: str, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    system_cfg = getattr(strategy.cfg, "system", None)
    if system_cfg and getattr(system_cfg, "merge_informative_into_base", False) and strategy._is_backtest_like_runmode():
        return out
    base_time = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
    last_ts = base_time.iloc[-1] if len(base_time) else None
    max_entries = int(getattr(system_cfg, "aligned_info_cache_max_entries", 0) or 0) if system_cfg else 0
    for tf in getattr(strategy, "_informative_timeframes", []):
        cache_key = (pair, tf, len(df), str(last_ts))
        cached = strategy._aligned_info_cache.get(cache_key)
        if cached is not None:
            if hasattr(strategy._aligned_info_cache, "move_to_end"):
                try:
                    strategy._aligned_info_cache.move_to_end(cache_key)
                except Exception:
                    pass
            out[tf] = cached
            continue
        info_df = strategy._get_informative_dataframe(pair, tf)
        if info_df is None or info_df.empty:
            continue
        left = pd.DataFrame({"_t": base_time})
        right = info_df
        right = right.assign(
            _tinfo=pd.to_datetime(right["date"]) if "date" in right.columns else pd.to_datetime(right.index)
        )
        merged = pd.merge_asof(
            left.sort_values("_t"),
            right.sort_values("_tinfo"),
            left_on="_t",
            right_on="_tinfo",
            direction="backward",
        )
        merged.index = df.index
        out[tf] = merged
        strategy._aligned_info_cache[cache_key] = merged
        if max_entries > 0:
            while len(strategy._aligned_info_cache) > max_entries:
                try:
                    strategy._aligned_info_cache.popitem(last=False)
                except Exception:
                    break
    return out


def informative_rows_for_index(aligned_info: Dict[str, pd.DataFrame], idx) -> Dict[str, pd.Series]:
    rows: Dict[str, pd.Series] = {}
    for tf, frame in aligned_info.items():
        try:
            rows[tf] = frame.loc[idx]
        except Exception:
            continue
    return rows


def candidate_with_plan(strategy, pair: str, candidate: Optional[schemas.Candidate], row: pd.Series, inf_rows: Dict[str, pd.Series]) -> Optional[schemas.Candidate]:
    if not candidate or not getattr(strategy, "exit_facade", None):
        return candidate
    try:
        _, _, plan = strategy.exit_facade.resolve_entry_plan(pair, candidate, row, inf_rows or {})
    except Exception:
        plan = None
    if not plan:
        return candidate
    try:
        return schemas.Candidate(
            direction=candidate.direction,
            kind=candidate.kind,
            raw_score=candidate.raw_score,
            rr_ratio=candidate.rr_ratio,
            win_prob=candidate.win_prob,
            expected_edge=candidate.expected_edge,
            squad=candidate.squad,
            sl_pct=float(plan.sl_pct) if getattr(plan, "sl_pct", None) else candidate.sl_pct,
            tp_pct=float(plan.tp_pct) if getattr(plan, "tp_pct", None) else candidate.tp_pct,
            exit_profile=candidate.exit_profile,
            recipe=candidate.recipe,
            timeframe=candidate.timeframe,
            plan_timeframe=getattr(plan, "timeframe", getattr(candidate, "plan_timeframe", None)),
            plan_atr_pct=getattr(plan, "atr_pct", getattr(candidate, "plan_atr_pct", None)),
        )
    except Exception:
        return candidate


def candidate_allowed_by_policy(policy, candidate: Optional[schemas.Candidate]) -> bool:
    if not policy or not candidate:
        return False
    if not policy.permits(kind=candidate.kind, squad=candidate.squad, recipe=candidate.recipe):
        return False
    if candidate.raw_score < policy.min_raw_score:
        return False
    if candidate.rr_ratio < policy.min_rr_ratio:
        return False
    if candidate.expected_edge < policy.min_edge:
        return False
    return True


def candidate_allowed_any_tier(strategy, candidate: Optional[schemas.Candidate]) -> bool:
    if not candidate:
        return False
    try:
        policies = list(strategy.tier_mgr.policies())
    except Exception:
        policies = []
    if not policies:
        policies = [strategy.tier_mgr.get(0)]
    return any(candidate_allowed_by_policy(pol, candidate) for pol in policies)


def group_candidates_by_direction(candidates: List[schemas.Candidate]) -> Dict[str, List[schemas.Candidate]]:
    grouped: Dict[str, List[schemas.Candidate]] = {"long": [], "short": []}
    for candidate in candidates:
        grouped[candidate.direction].append(candidate)
    return grouped


def trim_candidate_pool(strategy, grouped: Dict[str, List[schemas.Candidate]]) -> Dict[str, List[schemas.Candidate]]:
    limited: Dict[str, List[schemas.Candidate]] = {"long": [], "short": []}
    for direction, candidates in grouped.items():
        candidates = sorted(
            candidates, key=lambda c: (c.expected_edge, c.raw_score), reverse=True
        )
        limited[direction] = candidates[: strategy._candidate_pool_limit]
    return limited


def candidate_to_payload(candidate: schemas.Candidate) -> dict[str, Any]:
    return {
        "direction": candidate.direction,
        "kind": candidate.kind,
        "raw_score": candidate.raw_score,
        "rr_ratio": candidate.rr_ratio,
        "expected_edge": candidate.expected_edge,
        "sl_pct": candidate.sl_pct,
        "tp_pct": candidate.tp_pct,
        "plan_atr_pct": getattr(candidate, "plan_atr_pct", None),
        "timeframe": getattr(candidate, "timeframe", None),
    }


def update_last_signal(strategy, pair: str, candidate: Optional[schemas.Candidate], row: pd.Series) -> None:
    pst = strategy.state.get_pair_state(pair)
    if candidate:
        pst.last_dir = candidate.direction
        pst.last_squad = candidate.squad
        pst.last_score = float(candidate.expected_edge)
        pst.last_sl_pct = float(candidate.sl_pct)
        pst.last_tp_pct = float(candidate.tp_pct)
        pst.last_kind = str(candidate.kind)
        pst.last_exit_profile = candidate.exit_profile
        pst.last_recipe = candidate.recipe
        pst.last_atr_pct = float(candidate.plan_atr_pct) if candidate.plan_atr_pct is not None else 0.0
    else:
        pst.last_dir = None
        pst.last_squad = None
        pst.last_score = 0.0
        pst.last_sl_pct = 0.0
        pst.last_tp_pct = 0.0
        pst.last_exit_profile = None
        pst.last_recipe = None
        pst.last_atr_pct = 0.0
    strategy._last_signal[pair] = candidate


def reserve_risk_resources(
    strategy,
    pair: str,
    stake: float,
    risk: float,
    bucket: str,
    sl: float,
    tp: float,
    direction: str,
    current_rate: float,
    meta: Dict[str, Any],
) -> bool:
    meta_payload = dict(meta or {})
    meta_payload.update(
        {
            "sl_pct": sl,
            "tp_pct": tp,
            "stake_final": stake,
            "risk_final": risk,
            "bucket": bucket,
            "entry_price": current_rate,
            "dir": direction,
        }
    )

    if strategy._is_backtest_like_runmode():
        strategy._pending_entry_meta[pair] = meta_payload
        return True

    if getattr(strategy, "global_backend", None):
        backend_reserved = strategy._reserve_backend_risk(pair, risk)
        if not backend_reserved:
            return False

    rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
    strategy.reservation.reserve(pair, rid, risk, bucket)
    meta_payload["reservation_id"] = rid
    strategy._pending_entry_meta[pair] = meta_payload
    return True


def reserve_backend_risk(strategy, pair: str, risk: float) -> bool:
    cap_abs = 0.0
    try:
        equity_now = strategy.eq_provider.get_equity()
        cap_pct = strategy.state.get_dynamic_portfolio_cap_pct(equity_now)
        cap_abs = cap_pct * equity_now
        reserved = bool(strategy.global_backend.add_risk_usage(risk, cap_abs))
    except Exception as exc:
        reserved = False
        try:
            strategy.logger.error(
                f"Global backend reservation failed for {pair}, risk={risk:.4f}, cap_abs={cap_abs:.4f}",
                exc_info=exc,
            )
        except Exception:
            print(f"[backend] failed to reserve {pair}: {exc}")

    if not reserved:
        msg = f"Global Gatekeeper: CAP reached for {pair}, risk={risk:.4f}, cap_abs={cap_abs:.4f}"
        try:
            strategy.logger.info(msg)
        except Exception:
            print(msg)
        return False
    return True


def candidate_from_meta(strategy, pair: str, meta: Dict[str, Any]) -> Optional[schemas.Candidate]:
    signal_id = meta.get("signal_id")
    if not signal_id:
        return None
    try:
        signal_id = int(signal_id)
    except Exception:
        return None
    meta_info = strategy.hub.meta_for_id(signal_id)
    if not meta_info:
        return None
    try:
        raw_score = float(meta.get("raw_score", float("nan")))
        rr_ratio = float(meta.get("rr_ratio", float("nan")))
        expected_edge = float(meta.get("expected_edge", float("nan")))
        sl_pct = float(meta.get("sl_pct", float("nan")))
        tp_pct = float(meta.get("tp_pct", float("nan")))
    except Exception:
        return None
    if not all(map(math.isfinite, (raw_score, rr_ratio, expected_edge, sl_pct, tp_pct))):
        return None
    return schemas.Candidate(
        direction=meta_info.direction,
        kind=meta_info.name,
        raw_score=raw_score,
        rr_ratio=rr_ratio,
        win_prob=expected_edge,
        expected_edge=expected_edge,
        squad=str(meta_info.squad),
        sl_pct=sl_pct,
        tp_pct=tp_pct,
        exit_profile=meta_info.exit_profile,
        recipe=meta_info.recipe,
        timeframe=meta_info.timeframe,
        plan_timeframe=meta_info.plan_timeframe,
        plan_atr_pct=float(meta.get("plan_atr_pct"))
        if meta.get("plan_atr_pct") is not None
        else None,
    )


def get_entry_meta(strategy, pair: str, meta: Dict[str, Any], side: str) -> Dict[str, Any]:
    meta_payload = dict(meta or {})
    direction = meta_payload.get("dir")
    if not direction:
        direction = "long" if str(side).lower() in ("buy", "long") else "short"
    meta_payload["dir"] = direction
    meta_payload["score"] = meta_payload.get(
        "score", meta_payload.get("expected_edge", meta_payload.get("raw_score", 0.0))
    )
    return meta_payload


def update_rejection(strategy, reason: str, pair: str, context: Dict[str, Any]) -> None:
    tracker = getattr(strategy, "rejections", None)
    if not tracker:
        return
    try:
        tracker.record(reason, pair=pair, context=context)
    except Exception:
        return
