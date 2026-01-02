# -*- coding: utf-8 -*-
"""Query helpers for TaxBrain V30 configuration."""

from __future__ import annotations

from typing import Optional

from .models import ExitProfile, StrategySpec, TierSpec, V30Config


class ConfigService:
    """Query interface for V30 configuration."""

    def __init__(self, cfg: V30Config) -> None:
        self.cfg = cfg

    def get_exit_profile(self, name: str) -> ExitProfile:
        try:
            profiles = getattr(getattr(self.cfg, "strategy", None), "exit_profiles", None) or {}
            return profiles[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Unknown exit profile '{name}'") from exc

    def find_strategy_recipe(self, recipe_name: str) -> Optional[StrategySpec]:
        for recipe in self.cfg.strategy_recipes:
            if recipe.name == recipe_name:
                return recipe
        return None

    def entries_to_recipe(self, entry_kind: str) -> Optional[StrategySpec]:
        for recipe in self.cfg.strategy_recipes:
            if entry_kind in recipe.entries:
                return recipe
        return None

    def get_strategy(self, name: str) -> StrategySpec:
        recipe = self.find_strategy_recipe(name)
        if recipe:
            return recipe
        raise ValueError(f"Unknown strategy recipe '{name}'")

    def get_tier_spec(self, name: str) -> TierSpec:
        try:
            tiers = getattr(getattr(self.cfg, "strategy", None), "tiers", None) or {}
            return tiers[name]
        except KeyError as exc:
            raise ValueError(f"Unknown tier '{name}'") from exc

    def get_tier_for_closs(self, closs: int) -> TierSpec:
        tier_routing = getattr(getattr(self.cfg, "strategy", None), "tier_routing", None)
        tier_name = tier_routing.resolve(closs) if tier_routing else None
        if not tier_name:
            raise ValueError("Tier routing map is empty; cannot resolve tier for closs.")
        return self.get_tier_spec(tier_name)


def get_exit_profile(cfg: V30Config, name: str) -> ExitProfile:
    return ConfigService(cfg).get_exit_profile(name)


def find_strategy_recipe(cfg: V30Config, recipe_name: str) -> Optional[StrategySpec]:
    return ConfigService(cfg).find_strategy_recipe(recipe_name)


def entries_to_recipe(cfg: V30Config, entry_kind: str) -> Optional[StrategySpec]:
    return ConfigService(cfg).entries_to_recipe(entry_kind)


def get_strategy(cfg: V30Config, name: str) -> StrategySpec:
    return ConfigService(cfg).get_strategy(name)


def get_tier_spec(cfg: V30Config, name: str) -> TierSpec:
    return ConfigService(cfg).get_tier_spec(name)


def get_tier_for_closs(cfg: V30Config, closs: int) -> TierSpec:
    return ConfigService(cfg).get_tier_for_closs(closs)


__all__ = [
    "ConfigService",
    "entries_to_recipe",
    "find_strategy_recipe",
    "get_exit_profile",
    "get_strategy",
    "get_tier_for_closs",
    "get_tier_spec",
]
