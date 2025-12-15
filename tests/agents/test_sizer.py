"""SizerAgent sizing logic tests."""

import types
from dataclasses import replace

import pytest

from user_data.strategies.agents.portfolio.sizer import SizerAgent
from user_data.strategies.config.v29_config import V29Config


class DummyPairState:
    """Provide closs/local_loss placeholders."""

    def __init__(self, closs: int = 0, local_loss: float = 0.0):
        self.closs = closs
        self.local_loss = local_loss
        self.last_kind = None


class DummyState:
    """Minimal GlobalState surface for SizerAgent."""

    def __init__(self, debt_pool: float, cap_pct: float, total_open: float, pair_state: DummyPairState):
        self.debt_pool = debt_pool
        self._cap_pct = cap_pct
        self._total_open = total_open
        self._pair_state = pair_state
        self.pair_risk_open = {"TEST/USDT": 0.0}
        self.pair_stake_open = {"TEST/USDT": 0.0}
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
    """Return zeroed reservations for CAP calculations."""

    def __init__(self):
        self._total_reserved = 0.0
        self._pair_reserved = {"TEST/USDT": 0.0}

    def get_total_reserved(self) -> float:
        return self._total_reserved

    def get_pair_reserved(self, pair: str) -> float:
        return self._pair_reserved.get(pair, 0.0)


class DummyEquity:
    """Fixed equity provider."""

    def __init__(self, equity: float):
        self._equity = equity

    def get_equity(self) -> float:
        return self._equity


class DummyDataProvider:
    """Minimal dataprovider exposing exchange min_notional via market()."""

    def __init__(self, min_cost: float):
        self._min_cost = min_cost

    def market(self, pair: str):
        return {"limits": {"cost": {"min": self._min_cost}, "amount": {"min": None}}}


class DummyTierMgr:
    """Return a prebuilt TierPolicy."""

    def __init__(self, policy):
        self.policy = policy

    def get(self, _):
        return self.policy


def build_policy(**overrides):
    """Build a simple TierPolicy-like object."""

    defaults = dict(
        name="tier",
        allowed_entries={"mean_rev_long"},
        allowed_squads=set(),
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
    """Baseline risk suppressed under stress should skip sizing."""

    cfg = V29Config()
    cfg.trading = replace(cfg.trading, sizing=replace(cfg.trading.sizing, enforce_leverage=1.0))
    pair_state = DummyPairState()
    state = DummyState(
        debt_pool=500.0,
        cap_pct=cfg.risk.portfolio_cap_pct_base,
        total_open=0.0,
        pair_state=pair_state,
    )
    reservation = DummyReservation()
    equity = DummyEquity(1000.0)
    policy = build_policy()
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)

    stake, risk, bucket = sizer.compute("TEST/USDT", sl_pct=0.02, tp_pct=0.04, direction="long", min_stake=None, max_stake=1_000_000)

    assert stake == pytest.approx(0.0)
    assert risk == pytest.approx(0.0)
    assert bucket == "slow"


def test_sizer_base_only_uses_base_nominal_when_baseline_zero():
    """BASE_ONLY should honor base nominal even if baseline risk is suppressed."""

    cfg = V29Config()
    cfg.trading = replace(
        cfg.trading,
        sizing=replace(
            cfg.trading.sizing,
            enforce_leverage=1.0,
            static_initial_nominal=6.0,
            initial_size_equity_pct=0.0,
            initial_max_nominal_per_trade=1_000_000.0,
            per_pair_max_nominal_static=1_000_000.0,
        ),
    )
    pair_state = DummyPairState()
    state = DummyState(debt_pool=500.0, cap_pct=1.0, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(1000.0)
    policy = build_policy(
        sizing_algo="BASE_ONLY",
        k_mult_base_pct=0.0,
        per_pair_risk_cap_pct=10.0,
        max_stake_notional_pct=10.0,
    )
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)
    sizer.set_dataprovider(DummyDataProvider(6.0))

    stake, risk, bucket = sizer.compute(
        "TEST/USDT", sl_pct=0.02, tp_pct=0.04, direction="long", min_stake=None, max_stake=1_000_000
    )

    assert bucket == "slow"
    assert stake == pytest.approx(6.0)
    assert risk == pytest.approx(0.12)


def test_sizer_target_recovery_uses_local_loss():
    """TARGET_RECOVERY sizing follows ATR-style formula when uncapped."""

    cfg = V29Config()
    cfg.suppress_baseline_when_stressed = False
    cfg.trading = replace(
        cfg.trading,
        sizing=replace(
            cfg.trading.sizing,
            enforce_leverage=1.0,
            static_initial_nominal=50.0,
            initial_size_equity_pct=0.0,
            initial_max_nominal_per_trade=1_000_000.0,
            per_pair_max_nominal_static=1_000_000.0,
            initial_max_nominal_cap=1_000_000.0,
        ),
    )
    cfg.sizing_algos = replace(
        cfg.sizing_algos,
        target_recovery=replace(cfg.sizing_algos.target_recovery, max_recovery_multiple=1_000_000.0),
    )
    pair_state = DummyPairState(local_loss=100.0)
    state = DummyState(debt_pool=0.0, cap_pct=1.0, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(5000.0)
    policy = build_policy(sizing_algo="TARGET_RECOVERY", recovery_factor=2.0, per_pair_risk_cap_pct=10.0, max_stake_notional_pct=10.0)
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)
    sizer.set_dataprovider(DummyDataProvider(50.0))

    stake, risk, bucket = sizer.compute("TEST/USDT", sl_pct=0.02, tp_pct=0.04, direction="long", min_stake=None, max_stake=1_000_000)

    expected_nominal = 50.0 + (pair_state.local_loss * policy.recovery_factor) / 0.02
    assert bucket == "slow"
    assert stake == pytest.approx(expected_nominal, rel=1e-3)
    assert risk == pytest.approx(expected_nominal * 0.02, rel=1e-3)


def test_sizer_respects_caps_and_minmax():
    """Caps and exchange constraints should clamp the nominal target."""

    cfg = V29Config()
    cfg.trading = replace(
        cfg.trading,
        sizing=replace(
            cfg.trading.sizing,
            enforce_leverage=1.0,
            static_initial_nominal=50.0,
            initial_size_equity_pct=0.0,
            initial_max_nominal_per_trade=1000.0,
            per_pair_max_nominal_static=800.0,
            initial_max_nominal_cap=1000.0,
        ),
    )
    cfg.sizing_algos = replace(
        cfg.sizing_algos,
        target_recovery=replace(cfg.sizing_algos.target_recovery, max_recovery_multiple=1000.0),
    )
    pair_state = DummyPairState(local_loss=0.0)
    state = DummyState(debt_pool=0.0, cap_pct=1.0, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(5000.0)
    policy = build_policy(per_pair_risk_cap_pct=10.0, max_stake_notional_pct=1.0)
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)
    sizer.set_dataprovider(DummyDataProvider(50.0))

    stake, risk, bucket = sizer.compute(
        "TEST/USDT",
        sl_pct=0.05,
        tp_pct=0.10,
        direction="long",
        min_stake=None,
        max_stake=200.0,
    )

    assert bucket == "slow"
    assert stake == pytest.approx(800.0)
    assert risk == pytest.approx(40.0)


def test_sizer_caps_with_proposed_stake():
    """Freqtrade proposed stake should act as a hard cap."""

    cfg = V29Config()
    cfg.trading = replace(
        cfg.trading,
        sizing=replace(
            cfg.trading.sizing,
            enforce_leverage=1.0,
            initial_size_equity_pct=0.0,
            initial_max_nominal_per_trade=1_000_000.0,
            per_pair_max_nominal_static=1_000_000.0,
        ),
    )
    cfg.sizing_algos = replace(
        cfg.sizing_algos,
        target_recovery=replace(cfg.sizing_algos.target_recovery, max_recovery_multiple=1_000_000.0),
    )
    pair_state = DummyPairState()
    state = DummyState(debt_pool=0.0, cap_pct=1.0, total_open=0.0, pair_state=pair_state)
    reservation = DummyReservation()
    equity = DummyEquity(5000.0)
    policy = build_policy()
    tier_mgr = DummyTierMgr(policy)
    sizer = SizerAgent(state, reservation, equity, cfg, tier_mgr)

    stake, risk, bucket = sizer.compute(
        "TEST/USDT",
        sl_pct=0.02,
        tp_pct=0.04,
        direction="long",
        min_stake=None,
        max_stake=1_000_000,
        proposed_stake=100.0,
    )

    assert bucket == "slow"
    assert stake == pytest.approx(100.0)
    assert risk == pytest.approx(2.0)
