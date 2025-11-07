# -*- coding: utf-8 -*-
"""信号注册中心。

提供全局单例 :data:REGISTRY 用于存储所有通过装饰器注册的 :class:SignalSpec。
注册阶段通常在模块导入时完成，因此该模块不包含任何复杂逻辑，
重点是保证重复注册提示错误并对外暴露遍历接口。
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .schemas import SignalSpec


class SignalRegistry:
    """维护信号规格的简单容器。"""

    def __init__(self) -> None:
        self._specs: Dict[Tuple[str, Optional[str]], SignalSpec] = {}

    def register(self, spec: SignalSpec) -> None:
        """注册新的信号规格。

        Args:
            spec: 通过装饰器或手动构造的信号描述体。
        Raises:
            ValueError: 当重复注册相同名称的信号时抛出。
        """

        key = (spec.name, spec.timeframe)
        if key in self._specs:
            raise ValueError(f"Signal already registered: {spec.name} @ {spec.timeframe or 'primary'}")
        self._specs[key] = spec

    def all(self) -> List[SignalSpec]:
        """返回所有已注册信号的浅拷贝列表。"""

        return list(self._specs.values())


REGISTRY = SignalRegistry()


def register_signal(*, name: str, direction: str, squad: str,
                    conditions: Iterable, sl_fn, tp_fn, raw_fn, win_prob_fn,
                    min_rr: float = 0.0, min_edge: float = 0.0,
                    required_factors: Optional[Iterable[str]] = None,
                    timeframes: Optional[Iterable[Optional[str]]] = None):
    """信号插件使用的注册装饰器。

    装饰器本身不需要访问被装饰函数的逻辑，函数主体仅用于占位，
    因此在实际调用中可以写成 @register_signal(...)
def my_signal():
    pass。
    """

    tf_list = list(timeframes) if timeframes else [None]

    def decorator(fn: Callable[[], None]):
        for raw_tf in tf_list:
            normalized_tf = _normalize_timeframe(raw_tf)
            spec = SignalSpec(
                name=name,
                direction=direction,
                squad=squad,
                conditions=list(conditions),
                sl_fn=sl_fn,
                tp_fn=tp_fn,
                raw_fn=raw_fn,
                win_prob_fn=win_prob_fn,
                min_rr=min_rr,
                min_edge=min_edge,
                required_factors=tuple(required_factors or ()),
                timeframe=normalized_tf,
            )
            REGISTRY.register(spec)
        return fn

    return decorator


def _normalize_timeframe(value: Optional[object]) -> Optional[str]:
    """将装饰器传入的 timeframe 规格标准化。"""

    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        lowered = trimmed.lower()
        if lowered in {"primary", "main", "base"}:
            return None
        return trimmed
    return str(value)
