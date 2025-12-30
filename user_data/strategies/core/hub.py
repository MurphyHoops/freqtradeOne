from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import importlib
import importlib.util
import sys

from ..config.v30_config import V30Config, entries_to_recipe
from ..agents.signals import builder
from ..agents.signals.registry import REGISTRY
from ..plugins.signals import SIGNAL_PLUGIN_MAP

_LOADED_PLUGINS: set[str] = set()


@dataclass(frozen=True)
class SignalMeta:
    signal_id: int
    name: str
    timeframe: Optional[str]
    direction: str
    squad: str
    recipe: Optional[str]
    exit_profile: Optional[str]
    plan_timeframe: Optional[str]


class SignalHub:
    def __init__(self, cfg: V30Config) -> None:
        self._cfg = cfg
        self._discovered = False
        self._specs: List[Any] = []
        self._enabled_specs: List[Any] = []
        self._signal_id_map: Dict[Tuple[str, Optional[str], str], int] = {}
        self._id_meta: Dict[int, SignalMeta] = {}
        self._factor_requirements: Dict[Optional[str], set[str]] = {}
        self._indicator_requirements: Dict[Optional[str], set[str]] = {}

    def discover(self) -> None:
        system_cfg = getattr(self._cfg, "system", None)
        allow_reload = bool(getattr(system_cfg, "plugin_allow_reload", False)) if system_cfg else False
        if self._discovered and not allow_reload:
            return
        if allow_reload:
            _LOADED_PLUGINS.clear()
            REGISTRY.reset()
            self._discovered = False
            for name in list(sys.modules):
                if name.startswith("user_data.strategies.plugins."):
                    sys.modules.pop(name, None)
        logger = logging.getLogger(__name__)
        strict = bool(getattr(system_cfg, "plugin_load_strict", False)) if system_cfg else False
        enabled_signals = {
            name
            for name in (
                getattr(getattr(self._cfg, "strategy", None), "enabled_signals", getattr(self._cfg, "enabled_signals", ()))
                or ()
            )
            if name
        }
        plugin_root = Path(__file__).resolve().parents[1] / "plugins"
        if enabled_signals:
            for signal_name in sorted(enabled_signals):
                module_name = SIGNAL_PLUGIN_MAP.get(signal_name)
                if not module_name:
                    logger.warning("Signal plugin not found for: %s", signal_name)
                    if strict:
                        raise ValueError(f"Signal plugin not found for: {signal_name}")
                    continue
                module_key = f"module:{module_name}"
                if module_key in _LOADED_PLUGINS:
                    continue
                try:
                    importlib.import_module(module_name)
                    _LOADED_PLUGINS.add(module_key)
                except Exception as exc:
                    if isinstance(exc, ValueError) and "Signal already registered" in str(exc):
                        raise
                    logger.warning("Signal plugin load failed: %s (%s)", module_name, exc, exc_info=exc)
                    if strict:
                        raise
        if plugin_root.exists():
            for path in sorted(plugin_root.rglob("*.py")):
                if path.name.startswith("__"):
                    continue
                if enabled_signals and "signals" in path.parts:
                    continue
                abs_path = str(path.resolve())
                if abs_path in _LOADED_PLUGINS:
                    continue
                rel_parts = path.relative_to(plugin_root).with_suffix("").parts
                module_name = ".".join(["user_data", "strategies", "plugins", *rel_parts])
                spec = importlib.util.spec_from_file_location(module_name, str(path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)  # type: ignore[call-arg]
                        _LOADED_PLUGINS.add(abs_path)
                    except Exception as exc:
                        if isinstance(exc, ValueError) and "Signal already registered" in str(exc):
                            raise
                        logger.warning("Signal plugin load failed: %s (%s)", path, exc, exc_info=exc)
                        if strict:
                            raise
                        continue
        self._refresh_registry()
        self._discovered = True

    def _refresh_registry(self) -> None:
        specs = list(REGISTRY.all())
        specs.sort(key=lambda s: (s.origin or "", s.name, s.timeframe or "", s.direction, s.squad))
        self._specs = specs

        enabled = {
            name
            for name in (
                getattr(getattr(self._cfg, "strategy", None), "enabled_signals", getattr(self._cfg, "enabled_signals", ()))
                or ()
            )
            if name
        }
        self._enabled_specs = specs if not enabled else [spec for spec in specs if spec.name in enabled]

        self._signal_id_map.clear()
        self._id_meta.clear()
        profiles = getattr(getattr(self._cfg, "strategy", None), "exit_profiles", getattr(self._cfg, "exit_profiles", {})) or {}
        default_profile = getattr(
            getattr(self._cfg, "strategy", None), "default_exit_profile", getattr(self._cfg, "default_exit_profile", None)
        )
        for idx, spec in enumerate(specs, start=1):
            key = (spec.name, spec.timeframe, spec.direction)
            self._signal_id_map[key] = idx
            recipe = entries_to_recipe(self._cfg, spec.name)
            recipe_name = recipe.name if recipe else None
            exit_profile = recipe.exit_profile if recipe else default_profile
            profile = profiles.get(exit_profile) if exit_profile else None
            plan_tf = getattr(profile, "atr_timeframe", None) if profile else None
            self._id_meta[idx] = SignalMeta(
                signal_id=idx,
                name=spec.name,
                timeframe=spec.timeframe,
                direction=spec.direction,
                squad=spec.squad,
                recipe=recipe_name,
                exit_profile=exit_profile,
                plan_timeframe=plan_tf,
            )

        extra = getattr(self._cfg, "extra_signal_factors", None)
        self._factor_requirements = builder.collect_factor_requirements(extra, self._cfg)
        self._indicator_requirements = builder.collect_indicator_requirements(extra, self._cfg)

    @property
    def factor_requirements(self) -> Dict[Optional[str], set[str]]:
        return dict(self._factor_requirements)

    @property
    def indicator_requirements(self) -> Dict[Optional[str], set[str]]:
        return dict(self._indicator_requirements)

    @property
    def enabled_specs(self) -> List[Any]:
        return list(self._enabled_specs)

    def signal_id_for(self, name: str, timeframe: Optional[str], direction: str) -> Optional[int]:
        return self._signal_id_map.get((name, timeframe, direction))

    def meta_for_id(self, signal_id: int) -> Optional[SignalMeta]:
        return self._id_meta.get(signal_id)
