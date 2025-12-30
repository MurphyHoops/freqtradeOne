"""Integration tests for the plug-in candidate builder."""



from dataclasses import replace
from types import SimpleNamespace



from user_data.strategies.config.v30_config import ExitProfile, StrategyRecipe, V30Config
from user_data.strategies.agents.signals import build_candidates
from user_data.strategies.agents.signals import builder as builder_module
from user_data.strategies.agents.signals import factors as factors_module
from user_data.strategies.agents.signals.schemas import Condition, SignalSpec
from user_data.strategies.core.hub import SignalHub




def _row(**kwargs):

    return {

        "close": kwargs.get("close", 100.0),

        "ema_fast": kwargs.get("ema_fast", 100.0),

        "ema_slow": kwargs.get("ema_slow", 95.0),

        "rsi": kwargs.get("rsi", 20.0),

        "adx": kwargs.get("adx", 30.0),

        "atr": kwargs.get("atr", kwargs.get("atr_pct", 0.02) * kwargs.get("close", 100.0)),

        "atr_pct": kwargs.get("atr_pct", 0.02),

    }


def _ensure_signals_loaded():
    hub = SignalHub(V30Config())
    hub.discover()





def test_builder_emits_long_candidates_when_conditions_met():
    _ensure_signals_loaded()

    cfg = V30Config()

    row = _row(close=90.0, ema_fast=110.0, ema_slow=95.0, rsi=5.0, adx=45.0)

    cands = build_candidates(row, cfg)

    kinds = {c.kind for c in cands}

    assert "mean_rev_long" in kinds

    assert "pullback_long" in kinds

    for c in cands:

        assert c.rr_ratio > 0

        assert c.win_prob > 0

        assert c.exit_profile == cfg.strategy.default_exit_profile





def test_builder_emits_short_candidate_for_trend_short():
    _ensure_signals_loaded()

    cfg = V30Config()

    row = _row(close=130.0, ema_fast=80.0, ema_slow=120.0, rsi=60.0, adx=60.0)

    cands = build_candidates(row, cfg)

    kinds = {c.kind for c in cands}

    assert "trend_short" in kinds

    short = next(c for c in cands if c.kind == "trend_short")

    assert short.direction == "short"

    assert short.exit_profile == cfg.strategy.default_exit_profile





def test_builder_returns_empty_when_base_factors_nan():
    _ensure_signals_loaded()

    cfg = V30Config()

    row = _row(close=float("nan"))

    cands = build_candidates(row, cfg)

    assert cands == []


def test_hub_only_loads_enabled_signals():
    cfg = V30Config()
    cfg.system = replace(cfg.system, plugin_allow_reload=True)
    cfg.strategy = replace(cfg.strategy, enabled_signals=("mean_rev_long",))
    hub = SignalHub(cfg)
    hub.discover()
    names = {spec.name for spec in builder_module.REGISTRY.all()}
    assert names == {"mean_rev_long"}


def test_builder_prefers_regime_columns_over_history(monkeypatch):
    _ensure_signals_loaded()

    cfg = V30Config()

    row = _row(close=90.0, ema_fast=110.0, ema_slow=95.0, rsi=5.0, adx=45.0)
    row["hurst"] = 0.8
    row["adx_zsig"] = 0.2

    def _raise(*_args, **_kwargs):
        raise AssertionError("history regime compute should be skipped when columns exist")

    monkeypatch.setattr(builder_module, "_compute_hurst_rs", _raise)
    monkeypatch.setattr(builder_module, "_compute_adx_zsig", _raise)

    cands = build_candidates(
        row,
        cfg,
        history_close=[100.0, 101.0, 102.0],
        history_adx=[20.0, 21.0, 22.0],
    )

    assert cands





def test_builder_handles_timeframe_specific_spec(monkeypatch):

    cfg = V30Config()

    cfg.strategy = replace(
        cfg.strategy,
        exit_profiles={
            **cfg.strategy.exit_profiles,
            "tf_profile": ExitProfile(
                atr_mul_sl=1.0,
                floor_sl_pct=0.005,
                atr_mul_tp=6.0,
            ),
        },
        default_exit_profile="tf_profile",
        enabled_signals=cfg.strategy.enabled_signals + ("tf_long",),
    )

    cfg.strategy_recipes = (

        StrategyRecipe(

            name="tf_recipe",

            entries=("tf_long",),

            exit_profile="tf_profile",

            min_rr=0.0,

            min_edge=0.0,

        ),

    )


    spec = SignalSpec(

        name="tf_long",

        direction="long",

        squad="TF",

        conditions=[Condition("RSI", ">", 40.0)],

        raw_fn=lambda bag, _: bag["RSI"] / 100.0,

        win_prob_fn=lambda bag, _, raw: 0.5 + raw * 0.2,

        min_rr=1.0,

        min_edge=0.0,

        required_factors=("RSI",),

        timeframe="1h",

    )

    fake_registry = SimpleNamespace(all=lambda: [spec])

    monkeypatch.setattr(builder_module, "REGISTRY", fake_registry)



    row = _row()

    info_row = {

        "close_1h": 100.0,

        "ema_fast_1h": 101.0,

        "ema_slow_1h": 99.0,

        "rsi_1h": 60.0,

        "atr_1h": 2.0,

        "atr_pct_1h": 0.02,

        "adx_1h": 25.0,

    }

    cands = builder_module.build_candidates(row, cfg, informative={"1h": info_row})

    assert len(cands) == 1

    assert cands[0].kind == "tf_long"

    assert cands[0].exit_profile == cfg.strategy.default_exit_profile





def test_builder_uses_recipe_exit_profile_when_available(monkeypatch):

    cfg = V30Config()

    cfg.strategy = replace(
        cfg.strategy,
        exit_profiles={
            **cfg.strategy.exit_profiles,
            "custom_profile": ExitProfile(
                atr_mul_sl=1.0,
                floor_sl_pct=0.01,
                atr_mul_tp=2.5,
            ),
        },
        enabled_signals=cfg.strategy.enabled_signals + ("tf_long",),
    )

    cfg.strategy_recipes = (

        StrategyRecipe(

            name="custom_recipe",

            entries=("tf_long",),

            exit_profile="custom_profile",

            min_rr=1.2,

            min_edge=0.0,

        ),

    )


    spec = SignalSpec(

        name="tf_long",

        direction="long",

        squad="TF",

        conditions=[Condition("RSI", ">", 40.0)],

        raw_fn=lambda bag, _: bag["RSI"] / 100.0,

        win_prob_fn=lambda bag, _, raw: 0.5 + raw * 0.2,

        min_rr=1.0,

        min_edge=0.0,

        required_factors=("RSI",),

    )

    fake_registry = SimpleNamespace(all=lambda: [spec])

    monkeypatch.setattr(builder_module, "REGISTRY", fake_registry)



    row = _row(rsi=60.0, atr_pct=0.02)

    cands = builder_module.build_candidates(row, cfg)

    assert len(cands) == 1

    cand = cands[0]

    assert cand.exit_profile == "custom_profile"

    assert cand.recipe == "custom_recipe"

