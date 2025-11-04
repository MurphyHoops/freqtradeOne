import types

import pytest

from user_data.strategies.agents.sizer import SizerAgent
from user_data.strategies.config.v29_config import V29Config


class DummyPairState:
    def __init__(self, closs: int = 0, local_loss: float = 0.0):
        self.closs = closs
        self.local_loss = local_loss


class DummyState:
    def __init__(self, debt_pool: float, cap_pct: float, total_open: float, pair_state: DummyPairState):
        self.debt_pool = debt_pool
        self._cap_pct = cap_pct
        self._total_open = total_open
        self._pair_state = pair_state
        self.pair_risk_open = {"TEST/USDT": 0.0}
        self.treasury = types.SimpleNamespace(fast_alloc_risk={}, slow_alloc_risk={})

    def get_pair_state(self, pair: str) -> DummyPairState:
        return self._pair_state

    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        return self._cap_pct

    def get_total_open_risk(self) -> float:
        return self._total_open

    def per_pair_cap_room(self, pair: str, equity: float, tier_pol, reserved: float) -> float:
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.pair_risk_open.get(pair, 0.0) + reserved
        return max(0.0, cap - used)


class DummyReservation:
    def __init__(self):
        self._total_reserved = 0.0
        self._pair_reserved = {"TEST/USDT": 0.0}

    def get_total_reserved(self) -> float:
        return self._total_reserved

    def get_pair_reserved(self, pair: str) -> float:
        return self._pair_reserved.get(pair, 0.0)


class DummyEquity:
    def __init__(self, equity: float):
        self._equity = equity

    def get_equity(self) -> float:
        return self._equity


class DummyTierMgr:
    def __init__(self, policy):
        self.policy = policy

    def get(self, _):
        return self.policy


def build_policy(**overrides):
    defaults = dict(
        name="tier",
        allowed_kinds={"mean_rev_long"},
        min_raw_score=0.0,
        min_rr_ratio=0.0,
        min_edge=0.0,
        sizing_algo="BASELINE",
        k_mult_base_pct=0.01,
        recovery_factor=1.0,
        cooldown_bars=0,
        cooldown_bars_after_win=0,
        per_pair_risk_cap_pct=1.0,
        max_stake_notional_pct=1.0,
        icu_force_exit_bars=0,
    )
    defaults.update(overrides)
    policy = types.SimpleNamespace(**defaults)
    return policy


def test_sizer_suppresses_baseline_when_stressed():
    cfg = V29Config()
    pair_state = DummyPairState()
    state = DummyState(debt_pool=500.0, cap_pct=cfg.portfolio_cap_pct_base, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(1000.0)
    policy = build_policy()
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)

    stake, risk, bucket = sizer.compute("TEST/USDT", sl_pct=0.02, tp_pct=0.04, direction="long", min_stake=None, max_stake=1_000_000)

    assert stake == pytest.approx(0.0)
    assert risk == pytest.approx(0.0)
    assert bucket == "fast"


def test_sizer_target_recovery_uses_local_loss():
    cfg = V29Config()
    cfg.suppress_baseline_when_stressed = False
    pair_state = DummyPairState(local_loss=100.0)
    state = DummyState(debt_pool=0.0, cap_pct=1.0, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(5000.0)
    policy = build_policy(sizing_algo="TARGET_RECOVERY", recovery_factor=2.0)
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)

    stake, risk, bucket = sizer.compute("TEST/USDT", sl_pct=0.02, tp_pct=0.04, direction="long", min_stake=None, max_stake=10_000)

    assert bucket == "fast"
    assert risk == pytest.approx(100.0, rel=1e-3)
    assert stake == pytest.approx(5000.0, rel=1e-3)


def test_sizer_respects_caps_and_minmax():
    cfg = V29Config()
    pair_state = DummyPairState(local_loss=0.0)
    state = DummyState(debt_pool=0.0, cap_pct=0.01, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(5000.0)
    policy = build_policy(per_pair_risk_cap_pct=0.02, max_stake_notional_pct=0.5)
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)

    stake, risk, bucket = sizer.compute(
        "TEST/USDT",
        sl_pct=0.05,
        tp_pct=0.10,
        direction="long",
        min_stake=150.0,
        max_stake=200.0,
    )

    # Portfolio cap: 0.01 * 5000 = 50 risk -> stake 1000, but max stake trims to 200, min pushes to 200.
    assert bucket == "fast"
    assert stake == pytest.approx(200.0)
    assert risk == pytest.approx(10.0)
