"""Tests for closs/debt feedback loop in GlobalState."""

from __future__ import annotations

import pytest

from user_data.strategies.core.engine import GlobalState
from user_data.strategies.agents.portfolio.tier import TierAgent, TierManager
from user_data.strategies.agents.signals import schemas
from user_data.strategies.config.v29_config import V29Config


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
    tier_mgr = TierManager(cfg)
    tier_agent = TierAgent()
    candidate_nbx = schemas.Candidate(
        direction="long",
        kind="newbars_breakout_long_5m",
        raw_score=0.30,
        rr_ratio=0.5,
        win_prob=0.005,
        expected_edge=0.005,
        squad="NBX",
        sl_pct=0.01,
        tp_pct=0.02,
        exit_profile="ATRtrail_v1",
        recipe="NBX_fast_default",
        timeframe=None,
    )
    candidate_recovery = schemas.Candidate(
        direction="long",
        kind="pullback_long",
        raw_score=0.25,
        rr_ratio=0.6,
        win_prob=0.004,
        expected_edge=0.004,
        squad="PBL",
        sl_pct=0.012,
        tp_pct=0.024,
        exit_profile="ATRtrail_v1",
        recipe="Recovery_mix",
        timeframe=None,
    )

    policy_t0 = tier_mgr.get(0)
    selected = tier_agent.filter_best(policy_t0, [candidate_nbx, candidate_recovery])
    assert selected is not None
    assert selected.squad == "NBX"

    policy_t1 = tier_mgr.get(1)
    selected = tier_agent.filter_best(policy_t1, [candidate_nbx, candidate_recovery])
    assert selected is not None
    assert selected.squad == "PBL"
