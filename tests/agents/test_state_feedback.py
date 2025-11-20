"""Tests for closs/debt feedback loop in GlobalState."""

from __future__ import annotations

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
    state = strat.state
    pair = "ADA/USDT"
    pst = state.get_pair_state(pair)
    payload = {
        "version": 2,
        "candidates": {
            "long": [
                {
                    "direction": "long",
                    "kind": "newbars_breakout_long_5m",
                    "raw_score": 0.30,
                    "rr_ratio": 0.5,
                    "expected_edge": 0.005,
                    "win_prob": 0.55,
                    "squad": "NBX",
                    "sl_pct": 0.01,
                    "tp_pct": 0.02,
                    "exit_profile": "ATRtrail_v1",
                    "recipe": "NBX_fast_default",
                    "plan_timeframe": None,
                    "plan_atr_pct": None,
                },
                {
                    "direction": "long",
                    "kind": "pullback_long",
                    "raw_score": 0.25,
                    "rr_ratio": 0.6,
                    "expected_edge": 0.004,
                    "win_prob": 0.6,
                    "squad": "PBL",
                    "sl_pct": 0.012,
                    "tp_pct": 0.024,
                    "exit_profile": "ATRtrail_v1",
                    "recipe": "Recovery_mix",
                    "plan_timeframe": None,
                    "plan_atr_pct": None,
                },
            ],
            "short": [],
        },
    }

    # Tier-0 selects NBX candidate
    pst.closs = 0
    selected = strat._resolve_candidate_from_tag(pair, payload, "buy")
    assert selected is not None
    assert selected.squad == "NBX"

    # Tier-1 routes to recovery squad candidate
    pst.closs = 1
    selected = strat._resolve_candidate_from_tag(pair, payload, "buy")
    assert selected is not None
    assert selected.squad == "PBL"
