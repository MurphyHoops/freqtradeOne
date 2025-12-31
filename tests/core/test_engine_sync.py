from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pytest

from user_data.strategies.core.engine import Engine, GlobalState


def _plan_stub():
    return SimpleNamespace(
        k_long=0.0,
        k_short=0.0,
        theta=0.0,
        final_r=0.0,
        available=0.0,
        bias=0.0,
        volatility=1.0,
    )


def test_engine_sync_to_time_advances_and_decays():
    cfg = SimpleNamespace(
        cycle_len_bars=10,
        risk=SimpleNamespace(
            pain_decay_per_bar=0.5,
            clear_debt_on_profitable_cycle=False,
            portfolio_cap_pct_base=0.1,
            drawdown_threshold_pct=0.2,
        ),
    )
    state = GlobalState(cfg)
    state.debt_pool = 100.0
    pst = state.get_pair_state("BTC/USDT")
    pst.cooldown_bars_left = 3

    reservation = SimpleNamespace(
        tick_ttl=mock.Mock(),
        get_total_reserved=lambda: 0.0,
        get_pair_reserved=lambda *_: 0.0,
        reservations={},
    )
    treasury_agent = SimpleNamespace(plan=lambda *args, **kwargs: _plan_stub())
    risk_agent = SimpleNamespace(
        check_invariants=mock.Mock(return_value=SimpleNamespace(to_dict=lambda: {"ok": True}))
    )
    analytics = SimpleNamespace(log_finalize=lambda **kwargs: None, log_invariant=mock.Mock())
    persist = SimpleNamespace(save=lambda *args, **kwargs: None)
    eq_provider = SimpleNamespace(get_equity=lambda: 1000.0)

    engine = Engine(
        cfg=cfg,
        state=state,
        eq_provider=eq_provider,
        treasury_agent=treasury_agent,
        reservation=reservation,
        risk_agent=risk_agent,
        analytics=analytics,
        persist=persist,
        tier_mgr=None,
        tf_sec=60,
        is_backtest_like=lambda: True,
    )

    base_ts = datetime(2024, 1, 1, 0, 0, 0)
    state.last_finalized_bar_ts = base_ts.timestamp()
    current_time = datetime(2024, 1, 1, 0, 2, 0)

    engine.sync_to_time(current_time)

    assert state.bar_tick == 2
    assert pst.cooldown_bars_left == 1
    assert state.debt_pool == pytest.approx(25.0)
    assert reservation.tick_ttl.call_count == 2
    assert risk_agent.check_invariants.call_count == 1
    assert analytics.log_invariant.call_count == 1

    engine.sync_to_time(current_time)
    assert state.bar_tick == 2
    assert risk_agent.check_invariants.call_count == 1
    assert analytics.log_invariant.call_count == 1


def test_engine_finalize_clears_debt_on_profitable_cycle():
    cfg = SimpleNamespace(
        cycle_len_bars=1,
        risk=SimpleNamespace(
            pain_decay_per_bar=1.0,
            clear_debt_on_profitable_cycle=True,
            portfolio_cap_pct_base=0.1,
            drawdown_threshold_pct=0.2,
        ),
    )
    state = GlobalState(cfg)
    reservation = SimpleNamespace(
        tick_ttl=mock.Mock(),
        get_total_reserved=lambda: 0.0,
        get_pair_reserved=lambda *_: 0.0,
        reservations={},
    )
    treasury_agent = SimpleNamespace(plan=lambda *args, **kwargs: _plan_stub())
    risk_agent = SimpleNamespace(check_invariants=lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {"ok": True}))
    analytics = SimpleNamespace(log_finalize=lambda **kwargs: None, log_invariant=lambda *args, **kwargs: None)
    persist = SimpleNamespace(save=lambda *args, **kwargs: None)
    eq_provider = SimpleNamespace(get_equity=lambda: 1000.0)

    engine = Engine(
        cfg=cfg,
        state=state,
        eq_provider=eq_provider,
        treasury_agent=treasury_agent,
        reservation=reservation,
        risk_agent=risk_agent,
        analytics=analytics,
        persist=persist,
        tier_mgr=None,
        tf_sec=60,
        is_backtest_like=lambda: False,
    )

    engine.finalize_bar()
    state.debt_pool = 10.0
    pst = state.get_pair_state("BTC/USDT")
    pst.local_loss = 5.0
    pst.closs = 2

    engine.finalize_bar()

    assert state.debt_pool == 0.0
    assert pst.local_loss == 0.0
    assert pst.closs == 0
