# -*- coding: utf-8 -*-
"""信号模块的依赖分析工具。

该模块会在运行期遍历已经注册的 :class:`SignalSpec`，
计算出「需要哪些 factor / indicator / timeframe」，
用于驱动指标按需计算与自动 informative timeframe 注册。

主要入口：

* :func:`collect_factor_requirements` —— 返回每个 timeframe 需要的 factor 集合；
* :func:`collect_indicator_requirements` —— 将 factor 需求映射为指标需求；
* :func:`required_timeframes` —— 提取所有需要的 informative timeframe。

所有函数都会自动确保默认因子（例如 DEFAULT_BAG_FACTORS）被纳入，
因此调用方无需手动维护基础依赖。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional, Set

from .factor_spec import (
    DEFAULT_BAG_FACTORS,
    factor_components_with_default,
    factor_dependencies,
)
from .registry import REGISTRY

FactorMap = Dict[Optional[str], Set[str]]
IndicatorMap = Dict[Optional[str], Set[str]]


def _normalized_extra(extra: Optional[Iterable[str]]) -> Iterable[str]:
    """将用户传入的额外因子列表做一次去空/去重处理。"""

    if not extra:
        return ()
    seen: Set[str] = set()
    for item in extra:
        if not item:
            continue
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        yield key


def collect_factor_requirements(extra: Optional[Iterable[str]] = None) -> FactorMap:
    """返回按 timeframe 分类的 factor 需求字典。

    Args:
        extra: 允许调用方追加的因子名（可带 @timeframe 后缀），通常来自配置。

    Returns:
        dict: key 为 timeframe（基础周期使用 None），value 为该 timeframe 需要的 factor 集合。
    """

    mapping: Dict[Optional[str], Set[str]] = defaultdict(set)
    mapping[None].update(DEFAULT_BAG_FACTORS)
    defaults_injected: Set[Optional[str]] = {None}
    for item in _normalized_extra(extra):
        base, tf = factor_components_with_default(item, None)
        mapping[tf].add(base)

    for spec in REGISTRY.all():
        tf = spec.timeframe
        if tf not in defaults_injected:
            mapping[tf].update(DEFAULT_BAG_FACTORS)
            defaults_injected.add(tf)
        for cond in spec.conditions:
            base, resolved = factor_components_with_default(cond.factor, tf)
            mapping[resolved].add(base)
        for factor in getattr(spec, "required_factors", ()):
            base, resolved = factor_components_with_default(factor, tf)
            mapping[resolved].add(base)

    return {tf: set(values) for tf, values in mapping.items() if values}


def collect_indicator_requirements(extra: Optional[Iterable[str]] = None) -> IndicatorMap:
    """根据 factor 需求推导指标依赖。"""

    factor_map = collect_factor_requirements(extra)
    indicator_map: Dict[Optional[str], Set[str]] = defaultdict(set)
    for tf, factors in factor_map.items():
        for factor in factors:
            deps = factor_dependencies(factor)
            if deps:
                indicator_map[tf].update(deps)
    return {tf: set(values) for tf, values in indicator_map.items() if values}


def required_timeframes(extra: Optional[Iterable[str]] = None) -> Set[str]:
    """列出除基础周期以外需要注册的所有 timeframe。"""

    factor_map = collect_factor_requirements(extra)
    return {tf for tf in factor_map.keys() if tf}


__all__ = [
    "collect_factor_requirements",
    "collect_indicator_requirements",
    "required_timeframes",
]
