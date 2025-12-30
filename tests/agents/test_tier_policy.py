from user_data.strategies.agents.portfolio.tier import TierManager
from user_data.strategies.config.v30_config import V30Config


def test_tier_policy_prefers_recipes_over_squads():
    cfg = V30Config()
    tier_mgr = TierManager(cfg)
    healthy = tier_mgr.get_by_name("T0_healthy")
    recovery = tier_mgr.get_by_name("T12_recovery")

    assert healthy.permits(
        recipe="NBX_fast_default",
        kind="newbars_breakout_long_5m",
        squad="NBX",
    )

    assert not recovery.permits(
        recipe="NBX_fast_default",
        kind="newbars_breakout_long_5m",
        squad="NBX",
    )

    assert recovery.permits(
        recipe="Recovery_mix",
        kind="pullback_long",
        squad="PBL",
    )
