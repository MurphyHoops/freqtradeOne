"""Tests for closs/debt feedback loop in GlobalState."""

from __future__ import annotations

import pytest

from user_data.strategies.TaxBrainV29 import GlobalState
from user_data.strategies.agents.portfolio.tier import TierManager
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
