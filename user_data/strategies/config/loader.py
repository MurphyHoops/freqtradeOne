# -*- coding: utf-8 -*-
"""Config loaders and override helpers for TaxBrain V30."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import importlib.util
import json
import logging
import sys

from ..agents.exits.profiles import ExitProfile
from .models import (
    GatekeepingConfig,
    RiskConfig,
    SensorConfig,
    SizingAlgoConfig,
    SizingConfig,
    StrategyConfig,
    StrategySpec,
    SystemConfig,
    TargetRecoveryConfig,
    TierRouting,
    TierSpec,
    TradingConfig,
    TreasuryConfig,
    V30Config,
)
from .presets import (
    DEFAULT_ENABLED_SIGNALS,
    DEFAULT_EXIT_PROFILE_NAME,
    DEFAULT_EXIT_PROFILES,
    DEFAULT_PROFILE_VERSION,
    DEFAULT_STRATEGIES,
    DEFAULT_TIERS,
    DEFAULT_TIER_ROUTING_MAP,
)


def _coerce_dc(value: Any, cls):
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**{k: v for k, v in value.items() if k in {f.name for f in fields(cls)}})
    return cls()


def _coerce_exit_profiles(raw: Mapping[str, Any] | Dict[str, ExitProfile]) -> Dict[str, ExitProfile]:
    if not raw:
        return {}
    out: Dict[str, ExitProfile] = {}
    valid_fields = {f.name for f in fields(ExitProfile)}
    for name, value in raw.items():
        if isinstance(value, ExitProfile):
            out[name] = value
            continue
        if isinstance(value, Mapping):
            filtered = {k: v for k, v in value.items() if k in valid_fields}
            out[name] = ExitProfile(**filtered)
    return out


def _coerce_strategy_spec(payload: Any, name: str) -> StrategySpec:
    if isinstance(payload, StrategySpec):
        if payload.name != name:
            return replace(payload, name=name)
        return payload
    if isinstance(payload, Mapping):
        spec_fields = {f.name for f in fields(StrategySpec)}
        data = {k: v for k, v in payload.items() if k in spec_fields}
        data["name"] = name
        return StrategySpec(**data)
    raise TypeError(f"Invalid STRATEGY_SPEC payload for recipe '{name}'")


def _coerce_tier_spec(payload: Any, name: str) -> TierSpec:
    if isinstance(payload, TierSpec):
        if payload.name != name:
            return replace(payload, name=name)
        return payload
    if isinstance(payload, Mapping):
        if "allowed_entries" in payload:
            print(f"[WARN] TierSpec.allowed_entries is deprecated and ignored for tier '{name}'.")
        tier_fields = {f.name for f in fields(TierSpec)}
        data = {k: v for k, v in payload.items() if k in tier_fields}
        data.setdefault("name", name)
        return TierSpec(**data)
    raise ValueError(f"Tier '{name}' must be a TierSpec or mapping.")


def _is_default_strategies_map(strategies: Mapping[str, StrategySpec]) -> bool:
    if set(strategies.keys()) != set(DEFAULT_STRATEGIES.keys()):
        return False
    for name, spec in strategies.items():
        if not isinstance(spec, StrategySpec):
            return False
        if spec != DEFAULT_STRATEGIES.get(name):
            return False
    return True


def _is_default_tiers_map(tiers: Mapping[str, TierSpec]) -> bool:
    if set(tiers.keys()) != set(DEFAULT_TIERS.keys()):
        return False
    for name, spec in tiers.items():
        if not isinstance(spec, TierSpec):
            return False
        if spec != DEFAULT_TIERS.get(name):
            return False
    return True


def _resolve_user_data_dir(system_cfg: SystemConfig) -> Path:
    if system_cfg and getattr(system_cfg, "user_data_dir", None):
        return Path(system_cfg.user_data_dir).expanduser().resolve()
    cwd = Path.cwd()
    user_data = cwd / "user_data"
    if user_data.exists():
        return user_data.resolve()
    return cwd.resolve()


def _discover_recipe_plugins(system_cfg: SystemConfig) -> Dict[str, StrategySpec]:
    if not system_cfg.auto_discover_plugins:
        return {}
    root = _resolve_user_data_dir(system_cfg) / "strategies"
    plugin_root = root / "plugins" / "recipes"
    if not plugin_root.exists():
        return {}
    logger = logging.getLogger(__name__)
    strict = bool(getattr(system_cfg, "plugin_load_strict", False))
    out: Dict[str, StrategySpec] = {}
    for path in sorted(plugin_root.glob("*.py")):
        if path.name.startswith("__"):
            continue
        recipe_name = path.stem
        module_name = f"user_data.strategies.plugins.recipes.{recipe_name}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if not spec or not spec.loader:
                msg = f"Recipe plugin load failed (no loader): {path}"
                logger.warning(msg)
                if strict:
                    raise ValueError(msg)
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)  # type: ignore[call-arg]
            except Exception as exc:
                sys.modules.pop(module_name, None)
                logger.warning("Recipe plugin load failed: %s (%s)", path, exc, exc_info=exc)
                if strict:
                    raise
                continue
        payload = getattr(module, "STRATEGY_SPEC", None)
        if payload is None:
            msg = f"Recipe plugin missing STRATEGY_SPEC: {path}"
            logger.warning(msg)
            if strict:
                raise ValueError(msg)
            sys.modules.pop(module_name, None)
            continue
        try:
            out[recipe_name] = _coerce_strategy_spec(payload, recipe_name)
        except Exception as exc:
            logger.warning("Recipe plugin invalid STRATEGY_SPEC: %s (%s)", path, exc, exc_info=exc)
            if strict:
                raise
            sys.modules.pop(module_name, None)
            continue
    return out


def _load_presets(system_cfg: SystemConfig) -> Dict[str, Any]:
    root = _resolve_user_data_dir(system_cfg)
    preset_path = root / "configs" / "v30_presets.json"
    if not preset_path.exists():
        return {}
    logger = logging.getLogger(__name__)
    try:
        with open(preset_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        logger.warning("Preset load failed: %s (%s)", preset_path, exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("Preset payload is not a mapping: %s", preset_path)
        return {}
    return payload


def _merge_dataclass(instance, updates: Mapping[str, Any] | Any):
    if isinstance(updates, type(instance)):
        return updates
    if not is_dataclass(instance):
        return updates
    base = {f.name: getattr(instance, f.name) for f in fields(instance)}
    if isinstance(updates, Mapping):
        for key, value in updates.items():
            if key not in base:
                continue
            if is_dataclass(base[key]) and isinstance(value, Mapping):
                base[key] = _merge_dataclass(base[key], value)
            else:
                base[key] = value
    return type(instance)(**base)


def _find_field_paths(instance, field_name: str, path=()):
    if not is_dataclass(instance):
        return
    for f in fields(instance):
        value = getattr(instance, f.name)
        if f.name == field_name:
            yield path + (f.name,)
        if is_dataclass(value):
            yield from _find_field_paths(value, field_name, path + (f.name,))


def _insert_path(target: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for token in path[:-1]:
        nxt = cursor.get(token)
        if not isinstance(nxt, dict):
            nxt = {}
        cursor[token] = nxt
        cursor = nxt
    cursor[path[-1]] = value


def _normalize_overrides(cfg: V30Config, strategy_params: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in strategy_params.items():
        if "." not in key:
            existing = normalized.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged = dict(existing)
                merged.update(value)
                normalized[key] = merged
            else:
                normalized[key] = value
            continue
        root, *rest = key.split(".")
        target = normalized.setdefault(root, {})
        if not isinstance(target, dict):
            target = {}
            normalized[root] = target
        cursor = target
        for token in rest[:-1]:
            nxt = cursor.get(token)
            if not isinstance(nxt, dict):
                nxt = {}
            cursor[token] = nxt
            cursor = nxt
        cursor[rest[-1]] = value

    top_fields = {f.name for f in fields(V30Config)}
    for key in list(normalized.keys()):
        if key in top_fields:
            continue
        paths = list(_find_field_paths(cfg, key))
        if len(paths) == 1:
            value = normalized.pop(key)
            _insert_path(normalized, paths[0], value)
    return normalized


def apply_overrides(cfg: V30Config, strategy_params: Optional[Mapping[str, Any]]) -> V30Config:
    """Apply overrides sourced from Freqtrade ``strategy_params``."""

    if not strategy_params:
        return cfg

    normalized = _normalize_overrides(cfg, strategy_params)

    if "system" in normalized:
        cfg.system = _merge_dataclass(cfg.system, normalized["system"])
    if "risk" in normalized:
        cfg.risk = _merge_dataclass(cfg.risk, normalized["risk"])
    if "trading" in normalized:
        cfg.trading = _merge_dataclass(cfg.trading, normalized["trading"])
    if "sensor" in normalized:
        cfg.sensor = _merge_dataclass(cfg.sensor, normalized["sensor"])
    if "strategy" in normalized:
        strat_updates = dict(normalized["strategy"])
        if "exit_profiles" in strat_updates:
            new_profiles = _coerce_exit_profiles(strat_updates.pop("exit_profiles"))
            if new_profiles:
                merged_profiles = dict(cfg.strategy.exit_profiles or {})
                merged_profiles.update(new_profiles)
                strat_updates["exit_profiles"] = merged_profiles
        cfg.strategy = _merge_dataclass(cfg.strategy, strat_updates)

    if "sizing_algos" in normalized:
        cfg.sizing_algos = _merge_dataclass(cfg.sizing_algos, normalized["sizing_algos"])

    top_fields = {f.name for f in fields(V30Config)}
    handled = {"system", "risk", "trading", "strategy", "sizing_algos", "sensor"}
    for key, value in normalized.items():
        if key in handled or key not in top_fields:
            continue
        if key in {"stoploss", "minimal_roi"}:
            continue
        setattr(cfg, key, value)

    if "stoploss" in normalized:
        cfg.stoploss = normalized["stoploss"]
    if "minimal_roi" in normalized:
        roi = normalized["minimal_roi"] or {}
        cfg.minimal_roi = dict(roi)

    if "stoploss" not in normalized:
        cfg.stoploss = float(cfg.trading.sizing.enforce_leverage) * -0.2
    if "minimal_roi" not in normalized:
        cfg.minimal_roi = {"0": 0.50 * float(cfg.trading.sizing.enforce_leverage)}
    return cfg


def _normalize_strategy_config(cfg: V30Config) -> V30Config:
    cfg.system = _coerce_dc(getattr(cfg, "system", None), SystemConfig)

    risk_cfg = _coerce_dc(getattr(cfg, "risk", None), RiskConfig)
    risk_gate = _coerce_dc(getattr(risk_cfg, "gatekeeping", None), GatekeepingConfig)
    cfg.risk = replace(risk_cfg, gatekeeping=risk_gate)

    trading_cfg = _coerce_dc(getattr(cfg, "trading", None), TradingConfig)
    t_sizing = _coerce_dc(getattr(trading_cfg, "sizing", None), SizingConfig)
    t_treasury = _coerce_dc(getattr(trading_cfg, "treasury", None), TreasuryConfig)
    cfg.trading = replace(trading_cfg, sizing=t_sizing, treasury=t_treasury)

    cfg.sensor = _coerce_dc(getattr(cfg, "sensor", None), SensorConfig)

    algos = _coerce_dc(getattr(cfg, "sizing_algos", None), SizingAlgoConfig)
    algos_tr = _coerce_dc(getattr(algos, "target_recovery", None), TargetRecoveryConfig)
    cfg.sizing_algos = replace(algos, target_recovery=algos_tr)

    strat_cfg = _coerce_dc(getattr(cfg, "strategy", None), StrategyConfig)

    presets = _load_presets(cfg.system)
    base_profiles = dict(DEFAULT_EXIT_PROFILES)
    if presets.get("exit_profiles"):
        base_profiles.update(_coerce_exit_profiles(presets["exit_profiles"]))

    base_strategies = dict(DEFAULT_STRATEGIES)
    if presets.get("strategies"):
        for name, spec in (presets["strategies"] or {}).items():
            base_strategies[name] = _coerce_strategy_spec(spec, name)

    base_tiers = dict(DEFAULT_TIERS)
    if presets.get("tiers"):
        for name, spec in (presets["tiers"] or {}).items():
            base_tiers[name] = _coerce_tier_spec(spec, name)

    base_enabled_signals = tuple(DEFAULT_ENABLED_SIGNALS)
    if presets.get("enabled_signals"):
        base_enabled_signals = tuple(presets["enabled_signals"])

    base_routing = dict(DEFAULT_TIER_ROUTING_MAP)
    routing_payload = presets.get("tier_routing_map") or presets.get("tier_routing") or {}
    if isinstance(routing_payload, Mapping) and routing_payload:
        base_routing.update({int(k): str(v) for k, v in routing_payload.items()})

    plugin_strategies = _discover_recipe_plugins(cfg.system)
    if plugin_strategies:
        base_strategies.update(plugin_strategies)

    exit_profiles_raw = strat_cfg.exit_profiles or {}
    strategy_profiles = _coerce_exit_profiles(exit_profiles_raw)
    if not strategy_profiles:
        strategy_profiles = dict(base_profiles)
    else:
        merged_profiles = dict(base_profiles)
        merged_profiles.update(strategy_profiles)
        strategy_profiles = merged_profiles

    strategies_raw = dict(getattr(strat_cfg, "strategies", {}) or {})
    if strategies_raw:
        if _is_default_strategies_map(strategies_raw):
            strategies = dict(base_strategies)
        else:
            strategies = {name: _coerce_strategy_spec(spec, name) for name, spec in strategies_raw.items()}
    else:
        strategies = dict(base_strategies)

    tiers_raw = dict(getattr(strat_cfg, "tiers", {}) or {})
    if tiers_raw:
        if _is_default_tiers_map(tiers_raw):
            tiers = dict(base_tiers)
        else:
            tiers = {name: _coerce_tier_spec(spec, name) for name, spec in tiers_raw.items()}
    else:
        tiers = dict(base_tiers)

    tier_routing = _coerce_dc(getattr(strat_cfg, "tier_routing", None), TierRouting)
    routing_map = dict(getattr(tier_routing, "loss_tier_map", {}) or {})
    if not routing_map or routing_map == dict(DEFAULT_TIER_ROUTING_MAP):
        routing_map = dict(base_routing)
    tier_routing = TierRouting(loss_tier_map=routing_map)

    enabled_recipes = tuple(name for name in getattr(strat_cfg, "enabled_recipes", ()) or () if name)
    enabled_signals = tuple(getattr(strat_cfg, "enabled_signals", ()) or ())
    if not enabled_signals or enabled_signals == tuple(DEFAULT_ENABLED_SIGNALS):
        enabled_signals = base_enabled_signals
    exit_profile_version = getattr(strat_cfg, "exit_profile_version", None) or DEFAULT_PROFILE_VERSION
    default_exit_profile = getattr(strat_cfg, "default_exit_profile", None) or DEFAULT_EXIT_PROFILE_NAME
    if enabled_recipes:
        missing_recipes = [name for name in enabled_recipes if name not in strategies]
        if missing_recipes:
            raise ValueError(f"enabled_recipes references unknown recipes: {missing_recipes}")
        seen_entries: list[str] = []
        for recipe_name in enabled_recipes:
            for entry in strategies[recipe_name].entries:
                if entry and entry not in seen_entries:
                    seen_entries.append(entry)
        enabled_signals = tuple(seen_entries)

    cfg.strategy = replace(
        strat_cfg,
        exit_profiles=strategy_profiles,
        strategies=strategies,
        tiers=tiers,
        tier_routing=tier_routing,
        enabled_recipes=enabled_recipes,
        enabled_signals=enabled_signals,
        exit_profile_version=exit_profile_version,
        default_exit_profile=default_exit_profile,
    )

    cfg.strategy_recipes = tuple(strategies.values())

    referenced_tiers = set(routing_map.values()) if routing_map else set()
    for tier_name in referenced_tiers:
        if tier_name not in tiers:
            raise ValueError(f"TierRouting references unknown tier '{tier_name}'")
    for tier_name in referenced_tiers:
        spec = tiers[tier_name]
        if not spec.default_exit_profile:
            raise ValueError(f"Tier '{tier_name}' must declare default_exit_profile for routing")
        if spec.default_exit_profile not in strategy_profiles:
            raise ValueError(
                f"Tier '{tier_name}' references unknown exit profile '{spec.default_exit_profile}'"
            )

    for tier_name, spec in tiers.items():
        for recipe_name in spec.allowed_recipes:
            if recipe_name not in strategies:
                raise ValueError(
                    f"Tier '{tier_name}' references unknown recipe '{recipe_name}'"
                )

    if getattr(cfg, "stoploss", None) is None:
        cfg.stoploss = float(cfg.trading.sizing.enforce_leverage) * -0.2
    if getattr(cfg, "minimal_roi", None) is None:
        cfg.minimal_roi = {"0": 0.50 * float(cfg.trading.sizing.enforce_leverage)}

    return cfg


def build_v30_config(strategy_params: Optional[Mapping[str, Any]] = None) -> V30Config:
    cfg = V30Config()
    cfg = apply_overrides(cfg, strategy_params)
    return _normalize_strategy_config(cfg)


__all__ = [
    "apply_overrides",
    "build_v30_config",
]
