import pandas as pd
import pytest
from types import SimpleNamespace

from user_data.strategies.config.v29_config import V29Config
from user_data.strategies.agents.exits.facade import ExitFacade
from user_data.strategies.agents.exits.exit_tags import ExitTags
from user_data.strategies.agents.exits.rules_threshold import hard_sl_from_entry, hard_tp_from_entry
from user_data.strategies.agents.exits.router import SLContext, TPContext
from user_data.strategies.agents.tier import TierManager


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
    cfg = V29Config()
    tier_mgr = TierManager(cfg)
    facade = ExitFacade(cfg, tier_mgr)

    pair_state = SimpleNamespace(closs=0, active_trades={})
    strategy = _DummyStrategy(_DummyState(pair_state))
    facade.attach_strategy(strategy)
    facade.set_dataprovider(_DummyDP())

    trade = _DummyTrade(trade_id=42)
    profile_name, profile_def, plan = facade.resolve_trade_plan("BTC/USDT", trade, None)

    assert profile_name == tier_mgr.default_profile_for_closs(0)
    assert profile_def is cfg.exit_profiles[profile_name]
    assert plan is not None
    assert plan.sl_pct > 0
    assert pytest.approx(plan.atr_pct) == plan.atr_pct  # sanity: float convertible


def test_rules_threshold_hard_sl_tp_from_entry():
    cfg = V29Config()

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
