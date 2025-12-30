"""Config recipe merge tests."""

import pytest

from user_data.strategies.config.v30_config import StrategySpec, TierSpec, V30Config


def test_strategy_config_overrides_defaults():
    spec = StrategySpec(
        name="CustomOnly",
        entries=("custom_signal",),
        exit_profile="ATRtrail_v1",
    )
    tier = TierSpec(
        name="T0",
        allowed_recipes=("CustomOnly",),
        default_exit_profile="ATRtrail_v1",
    )
    cfg = V30Config(
        strategy={
            "strategies": {"CustomOnly": spec},
            "tiers": {"T0": tier},
            "tier_routing": {"loss_tier_map": {0: "T0"}},
        }
    )
    assert set(cfg.strategy.strategies.keys()) == {"CustomOnly"}
    assert "NBX_fast_default" not in cfg.strategy.strategies


def test_enabled_recipes_derives_enabled_signals():
    spec = StrategySpec(
        name="CustomOnly",
        entries=("custom_signal", "custom_signal_two"),
        exit_profile="ATRtrail_v1",
    )
    tier = TierSpec(
        name="T0",
        allowed_recipes=("CustomOnly",),
        default_exit_profile="ATRtrail_v1",
    )
    cfg = V30Config(
        strategy={
            "strategies": {"CustomOnly": spec},
            "tiers": {"T0": tier},
            "tier_routing": {"loss_tier_map": {0: "T0"}},
            "enabled_recipes": ("CustomOnly",),
        }
    )
    assert cfg.strategy.enabled_signals == ("custom_signal", "custom_signal_two")


def test_enabled_recipes_requires_known_recipe():
    tier = TierSpec(
        name="T0",
        allowed_recipes=("CustomOnly",),
        default_exit_profile="ATRtrail_v1",
    )
    with pytest.raises(ValueError):
        V30Config(
            strategy={
                "strategies": {},
                "tiers": {"T0": tier},
                "tier_routing": {"loss_tier_map": {0: "T0"}},
                "enabled_recipes": ("MissingRecipe",),
            }
        )
