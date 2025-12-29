"""Tests for closs/debt feedback loop in GlobalState."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from user_data.strategies.TaxBrainV29 import GlobalState
from user_data.strategies.agents.portfolio.tier import TierAgent, TierManager
from user_data.strategies.config.v29_config import V29Config
from user_data.strategies import TaxBrainV29


def test_record_trade_close_updates_closs_and_debt():
    cfg = V29Config()
    state = GlobalState(cfg)
    tier_mgr = TierManager(cfg)
    pair = "BTC/USDT"
    trade_id = "99"
    state.get_pair_state(pair)  # ensure pair exists
    state.trade_risk_ledger[trade_id] = 0.0

    # Losing trade increments closs and debt_pool
    state.record_trade_close(pair, trade_id, profit_abs=-25.0, tier_mgr=tier_mgr)
    pst = state.get_pair_state(pair)
    assert pst.closs == 1
    assert state.debt_pool == pytest.approx(25.0)

    # Winning trade taxes and reduces closs
    state.trade_risk_ledger[trade_id] = 0.0
    state.record_trade_close(pair, trade_id, profit_abs=50.0, tier_mgr=tier_mgr)
    assert pst.closs == 0
    assert state.debt_pool < 25.0


def test_candidate_pool_selection_respects_tier():
    cfg = V29Config()
    strat = TaxBrainV29.TaxBrainV29.__new__(TaxBrainV29.TaxBrainV29)
    strat.cfg = cfg
    strat.state = GlobalState(cfg)
    strat.tier_mgr = TierManager(cfg)
    strat.tier_agent = TierAgent()
    strat.bridge = SimpleNamespace(
        get_candidates=lambda *args, **kwargs: [
            {
                "signal_id": 1,
                "raw_score": 0.30,
                "rr_ratio": 0.5,
                "expected_edge": 0.005,
                "sl_pct": 0.01,
                "tp_pct": 0.02,
                "plan_atr_pct": None,
            },
            {
                "signal_id": 2,
                "raw_score": 0.25,
                "rr_ratio": 0.6,
                "expected_edge": 0.004,
                "sl_pct": 0.012,
                "tp_pct": 0.024,
                "plan_atr_pct": None,
            },
        ]
    )
    strat.hub = SimpleNamespace(
        meta_for_id=lambda sid: SimpleNamespace(
            name="newbars_breakout_long_5m" if sid == 1 else "pullback_long",
            squad="NBX" if sid == 1 else "PBL",
            recipe="NBX_fast_default" if sid == 1 else "Recovery_mix",
            exit_profile="ATRtrail_v1",
            timeframe=None,
            plan_timeframe=None,
        )
    )
    state = strat.state
    pair = "ADA/USDT"
    pst = state.get_pair_state(pair)

    # Tier-0 selects NBX candidate
    pst.closs = 0
    selected = strat._select_candidate_from_pool(pair, datetime.utcnow(), "buy")
    assert selected is not None
    assert selected.squad == "NBX"

    # Tier-1 routes to recovery squad candidate
    pst.closs = 1
    selected = strat._select_candidate_from_pool(pair, datetime.utcnow(), "buy")
    assert selected is not None
    assert selected.squad == "PBL"
