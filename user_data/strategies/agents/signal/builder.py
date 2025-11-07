# -*- coding: utf-8 -*-
"""信号注册表到候选列表的执行器。

该模块读取 :data:`REGISTRY` 中的所有 :class:`SignalSpec`，
并通过 :class:`FactorBank` 获取所需的因子，逐条验证条件、计算 sl/tp/raw。
最终返回与 V29.1 兼容的 :class:`Candidate` 列表。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

from .factor_spec import DEFAULT_BAG_FACTORS, apply_timeframe_to_factor
from .factors import FactorBank
from .registry import REGISTRY
from .schemas import Candidate, Condition

_OPS = {
    "<": lambda x, v: x < v,
    "<=": lambda x, v: x <= v,
    ">": lambda x, v: x > v,
    ">=": lambda x, v: x >= v,
    "==": lambda x, v: x == v,
    "between": lambda x, bounds: bounds[0] <= x <= bounds[1],
    "outside": lambda x, bounds: x <= bounds[0] or x >= bounds[1],
}


def _check_condition(fb: FactorBank, cond: Condition, timeframe: Optional[str]) -> bool:
    """验证单条条件是否通过。"""

    try:
        factor_name = apply_timeframe_to_factor(cond.factor, timeframe)
        value = fb.get(factor_name)
    except KeyError:
        return False
    if value is None or math.isnan(value):
        return False
    if cond.fn is not None:
        try:
            return bool(cond.fn(value))
        except Exception:
            return False
    if cond.op is None:
        return bool(value)
    if cond.op in {"between", "outside"}:
        if cond.value is None or cond.value_hi is None:
            return False
        return _OPS[cond.op](value, (cond.value, cond.value_hi))
    if cond.value is None:
        return False
    return _OPS[cond.op](value, cond.value)


def _safe(value: float, default: float = 0.0) -> float:
    """在信号计算中统一兜底 NaN/Inf。"""

    try:
        if value is None or math.isnan(value) or math.isinf(value):
            return default
        return float(value)
    except Exception:
        return default


class _FactorBag(dict):
    """sl/tp/raw 等函数使用的惰性取值容器。"""

    def __init__(self, fb: FactorBank, initial: Dict[str, float], timeframe: Optional[str]) -> None:
        super().__init__(initial)
        self._fb = fb
        self._timeframe = timeframe

    def __getitem__(self, key: str) -> float:  # type: ignore[override]
        if key not in self:
            resolved = apply_timeframe_to_factor(key, self._timeframe)
            self[key] = self._fb.get(resolved)
        return super().__getitem__(key)


def build_candidates(row: Any, cfg, informative: Optional[Dict[str, Any]] = None) -> List[Candidate]:
    """根据最新的行情行与（可选）informative 行生成候选。"""
    print(f"[TaxBrainV29] df_build_candidates_informative_{informative}")
    print(f"[TaxBrainV29] df_build_candidates_row_{row}")
    fb = FactorBank(row, informative=informative)
    base_cache: Dict[Optional[str], Dict[str, float]] = {}
    base_cache[None] = _prefetch_base(fb, None)
    if any(math.isnan(v) for v in base_cache[None].values()):
        return []

    results: List[Candidate] = []
    for spec in REGISTRY.all():
        tf = spec.timeframe
        if tf not in base_cache:
            base_cache[tf] = _prefetch_base(fb, tf)
        base = dict(base_cache[tf])
        bag = _FactorBag(fb, base, tf)
        if not _all_conditions_pass(fb, spec.conditions, tf):
            continue

        sl = _safe(spec.sl_fn(bag, cfg))
        tp = _safe(spec.tp_fn(bag, cfg))
        if sl <= 0 or tp <= 0:
            continue
        raw = _safe(spec.raw_fn(bag, cfg))
        win = _safe(spec.win_prob_fn(bag, cfg, raw), default=0.5)
        rr = tp / max(sl, 1e-12)
        edge = win * tp - (1.0 - win) * sl
        if rr < spec.min_rr or edge < spec.min_edge:
            continue

        results.append(
            Candidate(
                direction=spec.direction,
                kind=spec.name,
                sl_pct=sl,
                tp_pct=tp,
                raw_score=raw,
                rr_ratio=rr,
                win_prob=win,
                expected_edge=edge,
                squad=spec.squad,
            )
        )
    return results


def _prefetch_base(fb: FactorBank, timeframe: Optional[str]) -> Dict[str, float]:
    """预拉取指定 timeframe 的默认基础因子。"""

    data: Dict[str, float] = {}
    for factor in DEFAULT_BAG_FACTORS:
        resolved = apply_timeframe_to_factor(factor, timeframe)
        try:
            data[factor] = fb.get(resolved)
        except KeyError:
            data[factor] = float("nan")
    return data


def _all_conditions_pass(
    fb: FactorBank, conditions: Iterable[Condition], timeframe: Optional[str]
) -> bool:
    for cond in conditions:
        if not _check_condition(fb, cond, timeframe):
            return False
    return True
