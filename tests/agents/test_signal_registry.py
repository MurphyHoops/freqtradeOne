from __future__ import annotations

from user_data.strategies.agents.signals.registry import SignalRegistry
from user_data.strategies.agents.signals.schemas import SignalSpec


def test_registry_allows_same_name_different_direction():
    registry = SignalRegistry()

    spec_long = SignalSpec(
        name="mean_rev",
        direction="long",
        squad="MRL",
        conditions=[],
        raw_fn=lambda *_: 0.0,
        win_prob_fn=lambda *_: 0.5,
        timeframe=None,
    )
    spec_short = SignalSpec(
        name="mean_rev",
        direction="short",
        squad="MRS",
        conditions=[],
        raw_fn=lambda *_: 0.0,
        win_prob_fn=lambda *_: 0.5,
        timeframe=None,
    )

    registry.register(spec_long)
    registry.register(spec_short)

    specs = registry.all()
    assert len(specs) == 2
