"""TreasuryAgent unit tests."""

from dataclasses import replace

import pytest

from user_data.strategies.agents.portfolio.tier import TierPolicy
from user_data.strategies.agents.portfolio.treasury import TreasuryAgent
from user_data.strategies.config.v30_config import V30Config


class FixedTierManager:
    """Fixed TierPolicy adapter."""

    def __init__(self, policy: TierPolicy):
        self._policy = policy

    def get(self, _closs: int) -> TierPolicy:
        return self._policy


def make_policy(**overrides) -> TierPolicy:
    """Return a TierPolicy with overrides."""

    data = dict(
        name="test",
        allowed_recipes=set(),
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
    """Provide a baseline treasury snapshot."""

    return {
        "debt_pool": 200.0,
        "total_open_risk": 0.0,
        "reserved_portfolio_risk": 0.0,
        "pairs": {},
    }


def test_treasury_selects_best_per_squad_and_distributes_fast_slow():
    """Ensure per-squad selection feeds fast/slow allocations."""

    cfg = V30Config()
    cfg.trading = replace(
        cfg.trading,
        treasury=replace(
            cfg.trading.treasury,
            treasury_fast_split_pct=0.5,
            fast_topK_squads=3,
            slow_universe_pct=1.0,
            min_injection_nominal_fast=0.0,
            min_injection_nominal_slow=0.0,
        ),
    )
    cfg.risk = replace(cfg.risk, portfolio_cap_pct_base=1.0)
    policy = make_policy()
    agent = TreasuryAgent(cfg, FixedTierManager(policy))

    snapshot = base_snapshot()
    snapshot["pairs"] = {
        "A": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 3.0,
            "last_dir": "long",
            "last_kind": "mean_rev_long",
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
            "last_kind": "mean_rev_long",
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
            "last_kind": "pullback_long",
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
    """Min injection and per-pair caps should still hold."""

    cfg = V30Config()
    cfg.trading = replace(
        cfg.trading,
        treasury=replace(
            cfg.trading.treasury,
            treasury_fast_split_pct=1.0,
            fast_topK_squads=1,
            slow_universe_pct=1.0,
            min_injection_nominal_fast=10.0,
            min_injection_nominal_slow=5.0,
        ),
    )
    cfg.risk = replace(cfg.risk, portfolio_cap_pct_base=1.0)
    policy = make_policy(per_pair_risk_cap_pct=0.01)
    agent = TreasuryAgent(cfg, FixedTierManager(policy))

    snapshot = base_snapshot()
    snapshot["pairs"] = {
        "X": {
            "cooldown_bars_left": 0,
            "active_trades": 0,
            "last_score": 5.0,
            "last_dir": "long",
            "last_kind": "mean_rev_long",
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
