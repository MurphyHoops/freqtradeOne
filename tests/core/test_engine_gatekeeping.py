from __future__ import annotations

from types import SimpleNamespace

from dataclasses import replace

from user_data.strategies.core.engine import Engine, GlobalState
from user_data.strategies.core.rejections import RejectReason, RejectTracker
from user_data.strategies.config.v29_config import V29Config


class _DummyReservation:
    def __init__(self, reserved: float = 0.0) -> None:
        self._reserved = reserved

    def get_total_reserved(self) -> float:
        return self._reserved

    def get_pair_reserved(self, _pair: str) -> float:
        return 0.0

    def tick_ttl(self) -> None:
        return None


class _DummyTierMgr:
    def get(self, _closs: int):
        return SimpleNamespace(single_position_only=False)


def _make_engine(*, gate_allowed: bool, reserved: float = 0.0, open_risk: float = 0.0):
    cfg = V29Config()
    cfg.risk = replace(cfg.risk, portfolio_cap_pct_base=0.01)
    state = GlobalState(cfg)
    state.pair_risk_open["BTC"] = open_risk
    eq_provider = SimpleNamespace(get_equity=lambda: 1000.0)
    treasury_agent = SimpleNamespace(
        evaluate_signal_quality=lambda *args, **kwargs: {"allowed": gate_allowed, "reason": "blocked"}
    )
    reservation = _DummyReservation(reserved=reserved)
    risk_agent = SimpleNamespace(
        check_invariants=lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {"ok": True})
    )
    analytics = SimpleNamespace(log_finalize=lambda **kwargs: None, log_invariant=lambda *args, **kwargs: None)
    persist = SimpleNamespace(save=lambda: None)
    rejections = RejectTracker(log_enabled=False, stats_enabled=True)
    engine = Engine(
        cfg=cfg,
        state=state,
        eq_provider=eq_provider,
        treasury_agent=treasury_agent,
        reservation=reservation,
        risk_agent=risk_agent,
        analytics=analytics,
        persist=persist,
        tier_mgr=_DummyTierMgr(),
        tf_sec=300,
        is_backtest_like=lambda: True,
        rejections=rejections,
    )
    return engine, rejections


def test_engine_gatekeeping_blocks_when_score_fails():
    engine, tracker = _make_engine(gate_allowed=False, open_risk=0.0)
    assert engine.is_permitted("BTC/USDT", {"score": 0.6}) is False
    assert tracker.snapshot()[RejectReason.GATEKEEP] == 1


def test_engine_portfolio_cap_blocks_when_used_exceeds_cap():
    engine, tracker = _make_engine(gate_allowed=True, open_risk=20.0)
    assert engine.is_permitted("BTC/USDT", {"score": 0.6}) is False
    assert tracker.snapshot()[RejectReason.PORTFOLIO_CAP] == 1
