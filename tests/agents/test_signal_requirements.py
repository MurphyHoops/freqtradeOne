"""Tests for signal requirement collection utilities."""

from __future__ import annotations

from dataclasses import replace

from user_data.strategies.agents.signals import builder as req
from user_data.strategies.agents.signals.schemas import Condition, SignalSpec
from user_data.strategies.config.v30_config import V30Config


def _spec(
    name: str, conditions: list[Condition], *, timeframe: str | None = None, required: tuple[str, ...] = ()
) -> SignalSpec:
    return SignalSpec(
        name=name,
        direction="long",
        squad="TEST",
        conditions=conditions,
        raw_fn=lambda bag, cfg: 0.5,
        win_prob_fn=lambda bag, cfg, raw: 0.6,
        min_rr=1.0,
        min_edge=0.0,
        required_factors=required,
        timeframe=timeframe,
    )


def test_collect_factor_and_indicator_requirements(monkeypatch):
    spec = _spec(
        "demo",
        [Condition("RSI", ">", 10.0), Condition("EMA_TREND", "==", 1.0)],
        timeframe="1h",
        required=("EMA_FAST",),
    )
    monkeypatch.setattr(req.REGISTRY, "all", lambda: [spec])

    factor_map = req.collect_factor_requirements()
    assert None in factor_map
    assert "CLOSE" in factor_map[None]  # default bag included
    assert "1h" in factor_map and "RSI" in factor_map["1h"]
    assert "EMA_TREND" in factor_map["1h"]
    assert "EMA_FAST" in factor_map["1h"]  # 来自 required_factors

    indicator_map = req.collect_indicator_requirements()
    assert "1h" in indicator_map and "RSI" in indicator_map["1h"]

    tfs = req.required_timeframes()
    assert "1h" in tfs


def test_collect_factor_requirements_honors_extra(monkeypatch):
    monkeypatch.setattr(req.REGISTRY, "all", lambda: [])
    factor_map = req.collect_factor_requirements(extra=["ADX@4h"])
    assert "4h" in factor_map and "ADX" in factor_map["4h"]


def test_collect_factor_requirements_respects_enabled_signals(monkeypatch):
    keep = _spec("keep", [Condition("RSI", ">", 10.0)])
    skip = _spec("skip", [Condition("EMA_FAST", ">", 0.0)], timeframe="4h")
    monkeypatch.setattr(req.REGISTRY, "all", lambda: [keep, skip])
    cfg = V30Config()
    cfg.strategy = replace(cfg.strategy, enabled_signals=("keep",))

    factor_map = req.collect_factor_requirements(cfg=cfg)

    assert None in factor_map and "RSI" in factor_map[None]
    assert "4h" not in factor_map  # skip is not enabled
