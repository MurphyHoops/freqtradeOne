"""Unit tests for the signal registry."""

import dataclasses

import pytest

from user_data.strategies.agents.signal.registry import SignalRegistry
from user_data.strategies.agents.signal.schemas import Condition, SignalSpec


def _dummy_spec(name: str) -> SignalSpec:
    return SignalSpec(
        name=name,
        direction="long",
        squad="TEST",
        conditions=[Condition("CLOSE", ">", 0.0)],
        raw_fn=lambda bag, cfg: 0.5,
        win_prob_fn=lambda bag, cfg, raw: 0.6,
        min_rr=1.0,
        min_edge=0.0,
    )


def test_registry_prevents_duplicate_names():
    registry = SignalRegistry()
    registry.register(_dummy_spec("alpha"))
    with pytest.raises(ValueError):
        registry.register(_dummy_spec("alpha"))
    assert len(registry.all()) == 1


def test_registry_allows_same_name_different_timeframes():
    registry = SignalRegistry()
    primary = _dummy_spec("beta")
    registry.register(primary)
    registry.register(dataclasses.replace(primary, timeframe="1h"))
    assert len(registry.all()) == 2
