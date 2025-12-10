# ===========================
# File: agents/exits/router.py
# ===========================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd

# ---- Data contracts for threshold/immediate channels ----
@dataclass(frozen=True)
class SLContext:
    pair: str
    trade: Any
    now: Any
    profit: float
    dp: Any
    cfg: Any
    state: Any
    strategy: Any | None = None

@dataclass(frozen=True)
class TPContext:
    pair: str
    trade: Any
    now: Any
    profit: float
    dp: Any
    cfg: Any
    state: Any
    strategy: Any | None = None

@dataclass(frozen=True)
class ImmediateContext:
    pair: str
    trade: Any
    now: Any
    profit: float
    dp: Any
    cfg: Any
    state: Any
    strategy: Any | None = None

# ---- New: Data contract for vector exit channel ----
@dataclass(frozen=True)
class VectorContext:
    """Context for vectorized (dataframe) exit rules."""
    pair: str
    timeframe: str
    dp: Any
    cfg: Any
    state: Any
    metadata: Dict[str, Any]
    strategy: Any | None = None

# Function signatures
SLRuleFn = Callable[[SLContext], Optional[float]]      # -> sl_pct (>0). None: no opinion
TPRuleFn = Callable[[TPContext], Optional[float]]      # -> tp_pct (>0). None: no opinion
ImmediateRuleFn = Callable[[ImmediateContext], Optional[str]]  # -> close-now reason
# Vector rule should return either:
#   - the same df with exit columns modified, OR
#   - a dict {"exit_long": Series/array/col, "exit_short": ..., "exit_tag": ...}
VectorRuleFn = Callable[[Any, VectorContext], Union[Any, Dict[str, Any]]]

class ExitRouter:
    """Single entrypoint for exit logic.

    Channels:
      - VectorExit (populate_exit_trend) – dataframe-based rules (indicator strategies)
      - ThresholdExit (SL/TP, % of entry) – custom_stoploss() / custom_roi()
      - ImmediateExit – custom_exit() close-now reasons
    """
    def __init__(self) -> None:
        self._sl_rules: List[Tuple[int, str, SLRuleFn]] = []
        self._tp_rules: List[Tuple[int, str, TPRuleFn]] = []
        self._immediate_rules: List[Tuple[int, str, ImmediateRuleFn]] = []
        self._vector_rules: List[Tuple[int, str, VectorRuleFn]] = []

    # ---------------- Registration APIs ----------------
    def register_sl(self, name: str, fn: SLRuleFn, priority: int = 100) -> None:
        self._sl_rules.append((int(priority), name, fn))
        self._sl_rules.sort(key=lambda x: (x[0], x[1]))

    def register_tp(self, name: str, fn: TPRuleFn, priority: int = 100) -> None:
        self._tp_rules.append((int(priority), name, fn))
        self._tp_rules.sort(key=lambda x: (x[0], x[1]))

    def register_immediate(self, name: str, fn: ImmediateRuleFn, priority: int = 100) -> None:
        self._immediate_rules.append((int(priority), name, fn))
        self._immediate_rules.sort(key=lambda x: (x[0], x[1]))

    def register_vector_exit(self, name: str, fn: VectorRuleFn, priority: int = 100) -> None:
        """Register a dataframe-based exit rule."""
        self._vector_rules.append((int(priority), name, fn))
        self._vector_rules.sort(key=lambda x: (x[0], x[1]))

    # Decorators
    def sl_rule(self, name: str, priority: int = 100):
        def _deco(fn: SLRuleFn):
            self.register_sl(name, fn, priority)
            return fn
        return _deco

    def tp_rule(self, name: str, priority: int = 100):
        def _deco(fn: TPRuleFn):
            self.register_tp(name, fn, priority)
            return fn
        return _deco

    def immediate_rule(self, name: str, priority: int = 100):
        def _deco(fn: ImmediateRuleFn):
            self.register_immediate(name, fn, priority)
            return fn
        return _deco

    def vector_exit_rule(self, name: str, priority: int = 100):
        """Decorator for vectorized dataframe rules used by populate_exit_trend."""
        def _deco(fn: VectorRuleFn):
            self.register_vector_exit(name, fn, priority)
            return fn
        return _deco

    # --------------- Evaluation: threshold/immediate ---------------
    def sl_best(self, ctx: SLContext, base_sl_pct: Optional[float]) -> Optional[float]:
        """Return the tightest (min) sl percent (>0) among rules, compared vs base_sl_pct.
        Router only tightens, never loosens.
        """
        best: Optional[float] = base_sl_pct if (base_sl_pct is not None and base_sl_pct > 0) else None
        for _, name, fn in self._sl_rules:
            try:
                val = fn(ctx)
            except Exception:
                val = None
            if val and val > 0:
                tightened = float(val)
                if best is None:
                    best = tightened
                else:
                    best = min(best, tightened)
        return best

    def tp_best(self, ctx: TPContext, base_tp_pct: Optional[float]) -> Optional[float]:
        """Return the most conservative (min) tp percent (>0) among rules, compared vs base_tp_pct."""
        best: Optional[float] = base_tp_pct if (base_tp_pct is not None and base_tp_pct > 0) else None
        for _, name, fn in self._tp_rules:
            try:
                val = fn(ctx)
            except Exception:
                val = None
            if val and val > 0:
                tightened = float(val)
                if best is None:
                    best = tightened
                else:
                    best = min(best, tightened)
        return best

    def close_now_reason(self, ctx: ImmediateContext) -> Optional[str]:
        for _, name, fn in self._immediate_rules:
            try:
                reason = fn(ctx)
            except Exception:
                reason = None
            if isinstance(reason, str) and reason:
                return reason
        return None

    # ----------------- Evaluation: VectorExit -----------------
    def apply_vector_exits(
        self,
        df: Any,
        metadata: Dict[str, Any],
        dp: Any,
        cfg: Any,
        state: Any,
        timeframe_col: str = None,
        strategy: Any | None = None,
    ) -> Any:
        """Apply all registered vector exit rules to df (used in populate_exit_trend).

        Rules can:
          * modify df in-place and return df; or
          * return a dict with 'exit_long'/'exit_short'/'exit_tag' columns/arrays.

        Aggregation:
          * exit_long/exit_short: take max(0/1) across rules?OR ????)
          * exit_tag: ?????????????????????????? priority ????)
        """
        if "exit_long" not in df.columns:
            df["exit_long"] = 0
        if "exit_short" not in df.columns:
            df["exit_short"] = 0
        if "exit_tag" not in df.columns:
            try:
                df["exit_tag"] = None
            except Exception:
                df["exit_tag"] = ""

        def _series_from(value):
            if isinstance(value, pd.Series):
                series = value.reindex(df.index)
            else:
                try:
                    series = pd.Series(value, index=df.index)
                except Exception:
                    series = pd.Series([value] * len(df.index), index=df.index)
            return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

        def _align_tag(value):
            if isinstance(value, pd.Series):
                return value.reindex(df.index)
            try:
                return pd.Series(value, index=df.index)
            except Exception:
                return pd.Series([value] * len(df.index), index=df.index)

        pair = metadata.get("pair", "")
        timeframe = metadata.get("timeframe", getattr(cfg, "timeframe", ""))

        vctx = VectorContext(
            pair=pair,
            timeframe=timeframe,
            dp=dp,
            cfg=cfg,
            state=state,
            metadata=metadata,
            strategy=strategy,
        )

        base_exit_long = pd.to_numeric(df["exit_long"], errors="coerce").fillna(0).astype(int)
        base_exit_short = pd.to_numeric(df["exit_short"], errors="coerce").fillna(0).astype(int)
        exit_long_signals = [base_exit_long]
        exit_short_signals = [base_exit_short]
        last_tag_out = None
        exit_tag_result = df.get("exit_tag")

        for _, name, fn in self._vector_rules:
            try:
                out = fn(df, vctx)
            except Exception:
                continue

            if out is None:
                continue

            if isinstance(out, dict):
                if "exit_long" in out:
                    exit_long_signals.append(_series_from(out["exit_long"]))
                if "exit_short" in out:
                    exit_short_signals.append(_series_from(out["exit_short"]))
                if "exit_tag" in out:
                    last_tag_out = out["exit_tag"]
                continue

            try:
                for c in ("exit_long", "exit_short"):
                    if c not in df.columns:
                        df[c] = 0
                if "exit_tag" not in df.columns:
                    df["exit_tag"] = None
                updated_long = pd.to_numeric(df["exit_long"], errors="coerce").fillna(0).astype(int)
                updated_short = pd.to_numeric(df["exit_short"], errors="coerce").fillna(0).astype(int)
                exit_long_signals.append(updated_long)
                exit_short_signals.append(updated_short)
                exit_tag_result = df.get("exit_tag", exit_tag_result)
            except Exception:
                pass

        df["exit_long"] = pd.concat(exit_long_signals, axis=1).max(axis=1).astype(int)
        df["exit_short"] = pd.concat(exit_short_signals, axis=1).max(axis=1).astype(int)

        if last_tag_out is not None:
            try:
                df["exit_tag"] = _align_tag(last_tag_out)
            except Exception:
                pass
        elif exit_tag_result is not None:
            try:
                df["exit_tag"] = exit_tag_result
            except Exception:
                try:
                    df["exit_tag"] = _align_tag(exit_tag_result)
                except Exception:
                    pass

        return df




# Global singleton + decorators
EXIT_ROUTER = ExitRouter()

def sl_rule(name: str, priority: int = 100):
    return EXIT_ROUTER.sl_rule(name, priority)

def tp_rule(name: str, priority: int = 100):
    return EXIT_ROUTER.tp_rule(name, priority)

def immediate_rule(name: str, priority: int = 100):
    return EXIT_ROUTER.immediate_rule(name, priority)

def vector_exit_rule(name: str, priority: int = 100):
    return EXIT_ROUTER.vector_exit_rule(name, priority)

__all__ = [
    # Contexts
    "SLContext", "TPContext", "ImmediateContext", "VectorContext",
    # Router
    "ExitRouter", "EXIT_ROUTER",
    # Decorators
    "sl_rule", "tp_rule", "immediate_rule", "vector_exit_rule",
]


