"""SizerAgent 风险与仓位计算逻辑的测试。"""

import types

import pytest

from user_data.strategies.agents.sizer import SizerAgent
from user_data.strategies.config.v29_config import V29Config


class DummyPairState:
    """提供 closs 与 local_loss 的最小占位对象。"""

    def __init__(self, closs: int = 0, local_loss: float = 0.0):
        self.closs = closs
        self.local_loss = local_loss
        self.last_kind = None


class DummyState:
    """模拟 GlobalState 检索接口供 SizerAgent 使用。"""

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
    """返回固定值的预约桩，用于 CAP 计算。"""

    def __init__(self):
        self._total_reserved = 0.0
        self._pair_reserved = {"TEST/USDT": 0.0}

    def get_total_reserved(self) -> float:
        return self._total_reserved

    def get_pair_reserved(self, pair: str) -> float:
        return self._pair_reserved.get(pair, 0.0)


class DummyEquity:
    """固定返回 equity 的桩。"""

    def __init__(self, equity: float):
        self._equity = equity

    def get_equity(self) -> float:
        return self._equity


class DummyTierMgr:
    """返回预设 TierPolicy 的桩管理器。"""

    def __init__(self, policy):
        self.policy = policy

    def get(self, _):
        return self.policy


def build_policy(**overrides):
    """构造简单的 TierPolicy 替身以便调整参数。"""

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
    """压力期时 BASELINE VaR 应被压制导致拒单。"""

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
    """TARGET_RECOVERY 档位应根据 local_loss 放大风险需求。"""

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
    """组合/单票 CAP 与交易所 min/max 约束应正确生效。"""

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

    assert bucket == "fast"
    assert stake == pytest.approx(200.0)
    assert risk == pytest.approx(10.0)


def test_sizer_caps_with_proposed_stake():
    """Freqtrade �Ƽ��µ����޺�Ӧ��Ϊ����һ������Լ��."""

    cfg = V29Config()
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

    assert bucket == "fast"
    assert stake == pytest.approx(100.0)
    assert risk == pytest.approx(2.0)
