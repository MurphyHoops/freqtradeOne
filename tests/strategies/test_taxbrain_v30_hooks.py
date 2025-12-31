from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from user_data.strategies.core.engine import PairState
from user_data.strategies.TaxBrainV30 import TaxBrainV30


class _DummyState:
    def __init__(self) -> None:
        self.per_pair: dict[str, PairState] = {}

    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]

    def record_signal(self, pair: str, candidate) -> None:
        pst = self.get_pair_state(pair)
        if candidate:
            pst.last_dir = candidate.direction
            pst.last_squad = candidate.squad
            pst.last_score = float(candidate.expected_edge)
            pst.last_sl_pct = float(candidate.sl_pct)
            pst.last_tp_pct = float(candidate.tp_pct)
            pst.last_kind = str(candidate.kind)
            pst.last_exit_profile = candidate.exit_profile
            pst.last_recipe = candidate.recipe
            pst.last_atr_pct = float(candidate.plan_atr_pct) if candidate.plan_atr_pct is not None else 0.0
        else:
            pst.last_dir = None
            pst.last_squad = None
            pst.last_score = 0.0
            pst.last_sl_pct = 0.0
            pst.last_tp_pct = 0.0
            pst.last_exit_profile = None
            pst.last_recipe = None
            pst.last_atr_pct = 0.0


def _make_stub_strategy() -> TaxBrainV30:
    strat = TaxBrainV30.__new__(TaxBrainV30)
    strat.state = _DummyState()
    strat._last_signal = {}
    strat.bridge = SimpleNamespace()
    strat.bridge.get_side_meta = lambda *args, **kwargs: None
    strat.bridge.get_row_meta = lambda *args, **kwargs: None
    strat.hub = SimpleNamespace()
    strat.engine = SimpleNamespace(
        sync_to_time=lambda *args, **kwargs: None,
        is_permitted=lambda *args, **kwargs: True,
        reserve_risk_resources=lambda **kwargs: True,
        pending_entry_meta={},
    )
    strat.sizer = SimpleNamespace(compute=lambda **kwargs: (10.0, 1.0, "long"))
    strat.rejections = SimpleNamespace(record=lambda *args, **kwargs: None)
    strat.global_backend = None
    strat._runmode_token = "backtest"
    strat._is_backtest_like_runmode = lambda: True
    return strat


def test_v30_confirm_trade_entry_respects_engine():
    strat = _make_stub_strategy()
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)
    strat.engine = SimpleNamespace(sync_to_time=lambda *args, **kwargs: None, is_permitted=lambda *args, **kwargs: False)
    assert (
        strat.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="gtc",
            current_time=datetime(2024, 1, 1),
            entry_tag=None,
            side="buy",
        )
        is False
    )


def test_v30_confirm_trade_entry_rejects_on_time_alignment():
    strat = _make_stub_strategy()
    strat.bridge.get_side_meta = lambda *args, **kwargs: None
    assert (
        strat.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="gtc",
            current_time=datetime(2024, 1, 1),
            entry_tag=None,
            side="buy",
        )
        is False
    )


def test_v30_custom_stake_amount_uses_bridge_candidate():
    strat = _make_stub_strategy()
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)

    stake = strat.custom_stake_amount(
        pair="BTC/USDT",
        current_time=datetime(2024, 1, 1),
        current_rate=100.0,
        proposed_stake=10.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag=None,
        side="buy",
    )
    assert stake == 10.0
    pst = strat.state.get_pair_state("BTC/USDT")
    assert pst.last_score == 0.6


def test_v30_populate_entry_trend_updates_last_signal():
    strat = _make_stub_strategy()
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)
    strat.bridge.get_row_meta = lambda *args, **kwargs: meta
    strat.hub.meta_for_id = lambda *_: meta_info

    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "_signal_id": [7],
            "_signal_id_long": [7],
            "_signal_id_short": [pd.NA],
            "atr_pct": [0.02],
        }
    )
    out = strat.populate_entry_trend(df, {"pair": "BTC/USDT"})
    assert out["enter_tag"].iat[0] == "7"
    pst = strat.state.get_pair_state("BTC/USDT")
    assert pst.last_kind == "mean_rev_long"
    assert pst.last_score == 0.6


def test_v30_entry_tag_requires_matching_candidate():
    strat = _make_stub_strategy()
    strat.bridge.get_side_meta = lambda *args, **kwargs: None
    assert (
        strat.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="gtc",
            current_time=datetime(2024, 1, 1),
            entry_tag="7",
            side="buy",
        )
        is False
    )
    stake = strat.custom_stake_amount(
        pair="BTC/USDT",
        current_time=datetime(2024, 1, 1),
        current_rate=100.0,
        proposed_stake=10.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="7",
        side="buy",
    )
    assert stake == 0.0


def test_v30_entry_tag_uses_matching_candidate():
    strat = _make_stub_strategy()
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)
    assert (
        strat.confirm_trade_entry(
            pair="BTC/USDT",
            order_type="limit",
            amount=1.0,
            rate=100.0,
            time_in_force="gtc",
            current_time=datetime(2024, 1, 1),
            entry_tag="7",
            side="buy",
        )
        is True
    )
    stake = strat.custom_stake_amount(
        pair="BTC/USDT",
        current_time=datetime(2024, 1, 1),
        current_rate=100.0,
        proposed_stake=10.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag="7",
        side="buy",
    )
    assert stake == 10.0
