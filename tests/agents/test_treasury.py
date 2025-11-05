"""TreasuryAgent 拨款策略的单元测试。"""

import pytest

from user_data.strategies.agents.tier import TierPolicy
from user_data.strategies.agents.treasury import TreasuryAgent
from user_data.strategies.config.v29_config import V29Config


class FixedTierManager:
    """始终返回同一 TierPolicy 的桩。"""

    def __init__(self, policy: TierPolicy):
        self._policy = policy

    def get(self, _closs: int) -> TierPolicy:
        return self._policy


def make_policy(**overrides) -> TierPolicy:
    """快速构建带默认值的 TierPolicy，方便在测试中覆写参数。"""

    data = dict(
        name="test",
        allowed_kinds={"MRL", "PBL", "TRS"},
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
    data.update(overrides)
    return TierPolicy(**data)


def base_snapshot() -> dict:
    """提供最初的 Treasury 状态快照。"""

    return {
        "debt_pool": 200.0,
        "total_open_risk": 0.0,
        "reserved_portfolio_risk": 0.0,
        "pairs": {},
    }


def test_treasury_selects_best_per_squad_and_distributes_fast_slow():
    """验证 squad 代表选择、fast/slow 均分以及疼痛加权效果。"""

    cfg = V29Config()
    cfg.treasury_fast_split_pct = 0.5
    cfg.fast_topK_squads = 3
    cfg.slow_universe_pct = 1.0
    cfg.min_injection_nominal_fast = 0.0
    cfg.min_injection_nominal_slow = 0.0
    cfg.portfolio_cap_pct_base = 1.0
    policy = make_policy()
    agent = TreasuryAgent(cfg, FixedTierManager(policy))

    snapshot = base_snapshot()
    snapshot["pairs"] = {
        "A": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 3.0,
            "last_dir": "long",
            "last_squad": "MRL",
            "last_sl_pct": 0.02,
            "local_loss": 0.0,
            "closs": 0,
            "pair_open_risk": 0.0,
            "pair_reserved_risk": 0.0,
        },
        "B": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 2.0,
            "last_dir": "long",
            "last_squad": "MRL",
            "last_sl_pct": 0.02,
            "local_loss": 0.0,
            "closs": 0,
            "pair_open_risk": 0.0,
            "pair_reserved_risk": 0.0,
        },
        "C": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 1.5,
            "last_dir": "short",
            "last_squad": "PBL",
            "last_sl_pct": 0.03,
            "local_loss": 40.0,
            "closs": 0,
            "pair_open_risk": 0.0,
            "pair_reserved_risk": 0.0,
        },
    }

    plan = agent.plan(snapshot, equity=1000.0)

    assert set(plan.fast_alloc_risk.keys()) == {"A", "C"}
    assert pytest.approx(sum(plan.fast_alloc_risk.values())) == pytest.approx(100.0)
    assert set(plan.slow_alloc_risk.keys()) == {"A", "B", "C"}
    assert pytest.approx(sum(plan.slow_alloc_risk.values())) == pytest.approx(100.0)
    assert "B" not in plan.fast_alloc_risk


def test_treasury_honours_min_injection_and_cap_trim():
    """确保最小注入与单票 CAP 修剪逻辑生效。"""

    cfg = V29Config()
    cfg.treasury_fast_split_pct = 1.0
    cfg.fast_topK_squads = 1
    cfg.slow_universe_pct = 1.0
    cfg.min_injection_nominal_fast = 10.0
    cfg.min_injection_nominal_slow = 5.0
    cfg.portfolio_cap_pct_base = 1.0
    policy = make_policy(per_pair_risk_cap_pct=0.01)
    agent = TreasuryAgent(cfg, FixedTierManager(policy))

    snapshot = base_snapshot()
    snapshot["pairs"] = {
        "X": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 5.0,
            "last_dir": "long",
            "last_squad": "MRL",
            "last_sl_pct": 0.01,
            "local_loss": 0.0,
            "closs": 0,
            "pair_open_risk": 0.0,
            "pair_reserved_risk": 0.0,
        }
    }

    plan = agent.plan(snapshot, equity=1000.0)

    assert pytest.approx(plan.fast_alloc_risk["X"]) == pytest.approx(10.0)
    assert plan.slow_alloc_risk == {}
