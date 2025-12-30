from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from user_data.strategies.config.v30_config import V30Config
from user_data.strategies.agents.exits.facade import ExitFacade
from user_data.strategies.agents.exits.exit import ExitTags
from user_data.strategies.agents.exits.rules_threshold import (
    atr_trail_from_profile,
    breakeven_lock_from_profile,
    hard_sl_from_entry,
    hard_tp_from_entry,
)
from user_data.strategies.agents.exits.router import SLContext, TPContext
from user_data.strategies.agents.portfolio.tier import TierManager


class _DummyTrade:
    def __init__(self, trade_id: int = 1, custom: dict | None = None):
        self.trade_id = trade_id
        self.open_rate = 100.0
        self._custom = custom or {}

    def get_custom_data(self, key):
        return self._custom.get(key)

    def set_custom_data(self, key, value):
        self._custom[key] = value


class _DummyDP:
    def get_analyzed_dataframe(self, *args, **kwargs):
        return pd.DataFrame({"atr": [2.0], "close": [100.0]})


class _DummyState:
    def __init__(self, pair_state):
        self._pair_state = pair_state

    def get_pair_state(self, pair: str):
        return self._pair_state


class _DummyStrategy:
    def __init__(self, state):
        self.state = state
        self.analytics = None
        self.timeframe = "5m"

    def get_informative_value(self, *args, **kwargs):
        return None


def test_exit_facade_resolve_trade_plan_default_profile():
    cfg = V30Config()
    tier_mgr = TierManager(cfg)
    facade = ExitFacade(cfg, tier_mgr)

    pair_state = SimpleNamespace(closs=0, active_trades={})
    strategy = _DummyStrategy(_DummyState(pair_state))
    facade.attach_strategy(strategy)
    facade.set_dataprovider(_DummyDP())

    trade = _DummyTrade(trade_id=42)
    profile_name, profile_def, plan = facade.resolve_trade_plan("BTC/USDT", trade, None)

    assert profile_name == tier_mgr.default_profile_for_closs(0)
    assert profile_def is cfg.strategy.exit_profiles[profile_name]
    assert plan is not None
    assert plan.sl_pct > 0
    assert pytest.approx(plan.atr_pct) == plan.atr_pct  # sanity: float convertible


def test_exit_facade_respects_tier_default_profile(tmp_path):
    cfg = V30Config()
    cfg.strategy = replace(
        cfg.strategy,
        exit_profiles={
            **cfg.strategy.exit_profiles,
            "ALT_PROFILE": replace(cfg.strategy.exit_profiles["ATRtrail_v1"]),
        },
        tiers={
            **cfg.strategy.tiers,
            "T12_recovery": replace(cfg.strategy.tiers["T12_recovery"], default_exit_profile="ALT_PROFILE"),
        },
    )
    tier_mgr = TierManager(cfg)
    facade = ExitFacade(cfg, tier_mgr)

    pair_state = SimpleNamespace(closs=2, active_trades={})
    strategy = _DummyStrategy(_DummyState(pair_state))
    facade.attach_strategy(strategy)
    facade.set_dataprovider(_DummyDP())

    trade = _DummyTrade(trade_id=99)
    profile_name, profile_def, plan = facade.resolve_trade_plan("BTC/USDT", trade, None)

    assert profile_name == "ALT_PROFILE"
    assert profile_def is cfg.strategy.exit_profiles["ALT_PROFILE"]
    assert plan is not None and plan.profile_name == profile_name


def test_rules_threshold_hard_sl_tp_from_entry():
    cfg = V30Config()

    trade = _DummyTrade(custom={"sl_pct": 0.05, "tp_pct": 0.1})
    strategy = SimpleNamespace(exit_facade=None)
    ctx_sl = SLContext(
        pair="BTC/USDT",
        trade=trade,
        now=None,
        profit=0.0,
        dp=None,
        cfg=cfg,
        state=None,
        strategy=strategy,
    )
    ctx_tp = TPContext(
        pair="BTC/USDT",
        trade=trade,
        now=None,
        profit=0.0,
        dp=None,
        cfg=cfg,
        state=None,
        strategy=strategy,
    )

    assert hard_sl_from_entry(ctx_sl) == pytest.approx(0.05)
    assert hard_tp_from_entry(ctx_tp) == pytest.approx(0.1)

    stub_plan = SimpleNamespace(sl_pct=0.03, tp_pct=0.06)

    class StubFacade:
        def resolve_trade_plan(self, pair, trade, now):
            return ("profile", object(), stub_plan)

    trade_missing = _DummyTrade(custom={})
    ctx_sl_plan = ctx_sl.__class__(
        pair="BTC/USDT",
        trade=trade_missing,
        now=None,
        profit=0.0,
        dp=None,
        cfg=cfg,
        state=None,
        strategy=SimpleNamespace(exit_facade=StubFacade()),
    )
    ctx_tp_plan = ctx_tp.__class__(
        pair="BTC/USDT",
        trade=trade_missing,
        now=None,
        profit=0.0,
        dp=None,
        cfg=cfg,
        state=None,
        strategy=SimpleNamespace(exit_facade=StubFacade()),
    )

    assert hard_sl_from_entry(ctx_sl_plan) == pytest.approx(stub_plan.sl_pct)
    assert hard_tp_from_entry(ctx_tp_plan) == pytest.approx(stub_plan.tp_pct)


def test_exit_tags_vector_membership():
    assert ExitTags.is_vector_tag(ExitTags.TP_HIT)
    assert ExitTags.is_vector_tag(ExitTags.HARD_STOP)
    assert not ExitTags.is_vector_tag(ExitTags.ORDER_CANCELLED)
    assert not ExitTags.is_vector_tag("custom_reason")


class _StubFacade:
    def __init__(self, profile, plan, atr_pct: float | None = None):
        self._profile = profile
        self._plan = plan
        self._atr_pct = atr_pct

    def resolve_trade_plan(self, pair, trade, now):
        return ("profile", self._profile, self._plan)

    def atr_pct(self, pair, timeframe, now):
        return self._atr_pct


def _sl_ctx(cfg, trade, strategy, dp=None, profit: float = 0.0, now=None):
    return SLContext(
        pair="BTC/USDT",
        trade=trade,
        now=now,
        profit=profit,
        dp=dp,
        cfg=cfg,
        state=None,
        strategy=strategy,
    )


def test_breakeven_lock_from_profile(monkeypatch):
    cfg = V30Config()
    profile = SimpleNamespace(
        breakeven_lock_frac_of_tp=0.5,
        trail_mode=None,
        trail_pct=None,
        atr_timeframe=None,
    )
    plan = SimpleNamespace(sl_pct=0.04, tp_pct=0.04, atr_pct=0.01, timeframe=None)
    trade = _DummyTrade()
    strategy = SimpleNamespace(exit_facade=_StubFacade(profile, plan))

    ctx = _sl_ctx(cfg, trade, strategy, profit=0.05)
    value = breakeven_lock_from_profile(ctx)
    assert value == pytest.approx(0.001, abs=1e-6)


def test_percent_trail_rule_tightens_sl():
    cfg = V30Config()
    profile = SimpleNamespace(
        breakeven_lock_frac_of_tp=None,
        trail_mode="percent",
        trail_pct=0.5,
        atr_timeframe=None,
    )
    plan = SimpleNamespace(sl_pct=0.04, tp_pct=0.08, atr_pct=0.01, timeframe=None)
    trade = _DummyTrade()
    strategy = SimpleNamespace(exit_facade=_StubFacade(profile, plan))

    ctx = _sl_ctx(cfg, trade, strategy, profit=0.06)
    value = atr_trail_from_profile(ctx)
    assert value == pytest.approx(0.01, abs=1e-6)


def test_chandelier_trail_uses_price_extremes():
    cfg = V30Config()
    profile = SimpleNamespace(
        breakeven_lock_frac_of_tp=None,
        trail_mode="chandelier",
        trail_pct=None,
        trail_atr_mul=0.5,
        atr_timeframe="5m",
    )
    plan = SimpleNamespace(sl_pct=0.04, tp_pct=0.08, atr_pct=0.02, timeframe="5m")
    trade = _DummyTrade()
    index = pd.date_range("2024-01-01", periods=3, freq="1min")
    df = pd.DataFrame({"high": [100.0, 103.0, 102.0], "low": [99.0, 98.5, 98.0]}, index=index)
    trade.open_date_utc = index[0]
    dp_stub = SimpleNamespace(get_analyzed_dataframe=lambda *args, **kwargs: df)
    facade = _StubFacade(profile, plan, atr_pct=0.02)
    strategy = SimpleNamespace(exit_facade=facade)

    ctx = _sl_ctx(cfg, trade, strategy, dp=dp_stub, profit=0.03, now=index[-1])
    value = atr_trail_from_profile(ctx)
    assert value == pytest.approx(0.02, abs=1e-6)
