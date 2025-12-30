# -*- coding: utf-8 -*-
"""Vectorized backtest consistency and caching guardrails."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import user_data.strategies.TaxBrainV30 as strat_mod
from user_data.strategies.TaxBrainV30 import TaxBrainV30
from user_data.strategies.agents.signals import builder, vectorized
from user_data.strategies.core.hub import SignalHub
from user_data.strategies.config.v30_config import V30Config


def _base_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "ema_fast": [105.0, 105.0, 105.0],
            "ema_slow": [110.0, 110.0, 110.0],
            "rsi": [20.0, 20.0, 20.0],
            "adx": [30.0, 30.0, 30.0],
            "atr": [2.0, 2.0, 2.0],
            "atr_pct": [0.02, 0.02, 0.02],
            "newbars_high": [100.0, 100.0, 100.0],
            "newbars_low": [100.0, 100.0, 100.0],
            "atr_pct_30m": [0.03, 0.03, 0.03],
            "newbars_high_30m": [100.0, 100.0, 100.0],
            "newbars_low_30m": [100.0, 100.0, 100.0],
        }
    )


def _info_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="1h")
    return pd.DataFrame(
        {
            "date": dates,
            "newbars_high": [100.0, 100.0, 100.0],
            "newbars_low": [100.0, 100.0, 100.0],
            "atr_pct": [0.03, 0.03, 0.03],
            "newbars_high_30m": [100.0, 100.0, 100.0],
            "newbars_low_30m": [100.0, 100.0, 100.0],
            "atr_pct_30m": [0.03, 0.03, 0.03],
        }
    )


class _DummyDP:
    def __init__(self, info_df: pd.DataFrame | None = None):
        self._info_df = info_df

    def get_informative_dataframe(self, pair: str, timeframe: str):
        return self._info_df.copy() if self._info_df is not None else None


class _RunmodeDP:
    def __init__(self, runmode):
        self.runmode = runmode

    def get_informative_dataframe(self, pair: str, timeframe: str):
        return pd.DataFrame()

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        return pd.DataFrame()


def _patch_runmode(monkeypatch):
    class FakeRunMode:
        BACKTEST = object()
        HYPEROPT = object()
        PLOT = object()

    monkeypatch.setattr(strat_mod, "RunMode", FakeRunMode)
    return FakeRunMode


def _ensure_signals_loaded():
    cfg = V30Config()
    cfg.system = replace(cfg.system, plugin_allow_reload=True)
    hub = SignalHub(cfg)
    hub.discover()


def test_prefilter_specs_matches_default():
    _ensure_signals_loaded()
    cfg = V30Config()
    df = _base_frame()
    vectorized.add_derived_factor_columns(df, (None,))

    default_mask = vectorized.prefilter_signal_mask(df, cfg)
    specs = builder.REGISTRY.all()
    specs_mask = vectorized.prefilter_signal_mask(df, cfg, specs=specs)

    assert default_mask.equals(specs_mask)


def test_build_candidates_specs_matches_default():
    _ensure_signals_loaded()
    cfg = V30Config()
    row = _base_frame().iloc[-1]
    default = builder.build_candidates(row, cfg)
    specs = builder.REGISTRY.all()
    filtered = builder.build_candidates(row, cfg, specs=specs)

    assert [c.kind for c in default] == [c.kind for c in filtered]


def test_vectorized_and_nonvectorized_match(monkeypatch, tmp_path):
    fake = _patch_runmode(monkeypatch)
    params = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
            "vectorized_entry_backtest": True,
            "merge_informative_into_base": True,
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    vectorized_strat = TaxBrainV30(params)
    vectorized_strat.dp = _RunmodeDP(fake.BACKTEST)

    params["strategy_params"]["vectorized_entry_backtest"] = False
    non_vectorized_strat = TaxBrainV30(params)
    non_vectorized_strat.dp = _RunmodeDP(fake.BACKTEST)

    df = _base_frame()
    vec_df = vectorized_strat.populate_indicators(df.copy(), {"pair": "BTC/USDT"})
    vec_df = vectorized_strat.populate_entry_trend(vec_df, {"pair": "BTC/USDT"})
    non_df = non_vectorized_strat.populate_indicators(df.copy(), {"pair": "BTC/USDT"})
    non_df = non_vectorized_strat.populate_entry_trend(non_df, {"pair": "BTC/USDT"})

    assert vec_df[["enter_long", "enter_short"]].equals(non_df[["enter_long", "enter_short"]])
    assert vec_df["enter_tag"].fillna("").tolist() == non_df["enter_tag"].fillna("").tolist()


def test_vectorized_skips_when_informative_unmerged(monkeypatch, tmp_path):
    fake = _patch_runmode(monkeypatch)
    params = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
            "vectorized_entry_backtest": True,
            "merge_informative_into_base": False,
            "informative_timeframes": ("1h",),
            "enabled_signals": ("mean_rev_long",),
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV30(params)
    strategy.dp = SimpleNamespace(runmode=fake.BACKTEST, get_informative_dataframe=_DummyDP(_info_frame()).get_informative_dataframe)

    def _no_vectorized(*args, **kwargs):
        raise AssertionError("vectorized prefilter should be skipped when informative is unmerged")

    monkeypatch.setattr(vectorized, "prefilter_signal_mask", _no_vectorized)

    df = _base_frame()
    out = strategy.populate_indicators(df.copy(), {"pair": "BTC/USDT"})
    strategy.populate_entry_trend(out, {"pair": "BTC/USDT"})


def test_build_signal_matrices_matches_builder():
    _ensure_signals_loaded()
    cfg = V30Config()
    df = _base_frame()
    vectorized.add_derived_factor_columns(df, (None,))
    specs = [spec for spec in builder.REGISTRY.all() if spec.name == "mean_rev_long"]
    assert specs

    matrices = vectorized.build_signal_matrices(df, cfg, specs)
    assert len(matrices) == 1
    mat = matrices[0]
    pos = len(df) - 1
    candidates = builder.build_candidates(df.iloc[-1], cfg, specs=specs)
    assert len(candidates) == 1
    cand = candidates[0]

    assert mat["valid_mask"].iat[pos]
    assert mat["raw_score"].iat[pos] == pytest.approx(cand.raw_score)
    assert mat["rr_ratio"].iat[pos] == pytest.approx(cand.rr_ratio)
    assert mat["expected_edge"].iat[pos] == pytest.approx(cand.expected_edge)
    assert mat["sl_pct"].iat[pos] == pytest.approx(cand.sl_pct)
    assert mat["tp_pct"].iat[pos] == pytest.approx(cand.tp_pct)


def test_build_signal_matrices_matches_builder_for_all_plugin_signals():
    _ensure_signals_loaded()
    cfg = V30Config()
    df = _base_frame()
    vectorized.add_derived_factor_columns(df, (None, "30m"))
    specs = builder.REGISTRY.all()

    matrices = vectorized.build_signal_matrices(df, cfg, specs)
    matrices_by_name = {mat["name"]: mat for mat in matrices}
    pos = len(df) - 1

    for spec in specs:
        if not vectorized.is_vectorizable(spec):
            continue
        mat = matrices_by_name.get(spec.name)
        assert mat is not None
        candidates = builder.build_candidates(df.iloc[-1], cfg, specs=[spec])
        if not candidates:
            assert not mat["valid_mask"].iat[pos]
            continue
        cand = candidates[0]
        assert mat["valid_mask"].iat[pos]
        assert mat["raw_score"].iat[pos] == pytest.approx(cand.raw_score)
        assert mat["rr_ratio"].iat[pos] == pytest.approx(cand.rr_ratio)
        assert mat["expected_edge"].iat[pos] == pytest.approx(cand.expected_edge)
        assert mat["sl_pct"].iat[pos] == pytest.approx(cand.sl_pct)
        assert mat["tp_pct"].iat[pos] == pytest.approx(cand.tp_pct)


def test_regime_factor_matches_builder_with_series_inputs():
    _ensure_signals_loaded()
    cfg = V30Config()
    df = _base_frame()
    df["hurst"] = [0.8] * len(df)
    df["adx_zsig"] = [0.2] * len(df)
    vectorized.add_derived_factor_columns(df, (None,))
    specs = [spec for spec in builder.REGISTRY.all() if spec.name == "mean_rev_long"]
    assert specs

    matrices = vectorized.build_signal_matrices(df, cfg, specs)
    cand = builder.build_candidates(df.iloc[-1], cfg, specs=specs)[0]
    assert matrices[0]["expected_edge"].iat[len(df) - 1] == pytest.approx(cand.expected_edge)


def test_informative_merge_matches_aligned_rows(monkeypatch, tmp_path):
    fake = _patch_runmode(monkeypatch)
    info_df = _info_frame()
    dp = SimpleNamespace(runmode=fake.BACKTEST, get_informative_dataframe=_DummyDP(info_df).get_informative_dataframe)
    params = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
            "vectorized_entry_backtest": False,
            "informative_timeframes": ("30m",),
            "enabled_signals": ("newbars_breakout_long_30m",),
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }

    merged_params = dict(params)
    merged_params["strategy_params"] = dict(params["strategy_params"], merge_informative_into_base=True)
    merged_strat = TaxBrainV30(merged_params)
    merged_strat.dp = dp

    aligned_params = dict(params)
    aligned_params["strategy_params"] = dict(params["strategy_params"], merge_informative_into_base=False)
    aligned_strat = TaxBrainV30(aligned_params)
    aligned_strat.dp = dp

    base_df = _base_frame().drop(columns=["atr_pct_30m", "newbars_high_30m", "newbars_low_30m"])
    merged_df = base_df.copy()
    merged_strat._merge_informative_columns_into_base(merged_df, "BTC/USDT")

    merged_out = merged_strat.populate_indicators(merged_df.copy(), {"pair": "BTC/USDT"})
    merged_out = merged_strat.populate_entry_trend(merged_out, {"pair": "BTC/USDT"})
    aligned_out = aligned_strat.populate_indicators(base_df.copy(), {"pair": "BTC/USDT"})
    aligned_out = aligned_strat.populate_entry_trend(aligned_out, {"pair": "BTC/USDT"})

    assert merged_out[["enter_long", "enter_short"]].equals(aligned_out[["enter_long", "enter_short"]])
    assert merged_out["enter_tag"].fillna("").tolist() == aligned_out["enter_tag"].fillna("").tolist()


def test_missing_informative_atr_pct_blocks_vectorized_and_builder():
    _ensure_signals_loaded()
    cfg = V30Config()
    df = _base_frame().drop(columns=["atr_pct_30m"])
    vectorized.add_derived_factor_columns(df, (None, "30m"))
    specs = [spec for spec in builder.REGISTRY.all() if spec.name == "newbars_breakout_long_30m"]
    assert specs

    matrices = vectorized.build_signal_matrices(df, cfg, specs)
    assert len(matrices) == 1
    mat = matrices[0]

    candidates = builder.build_candidates(df.iloc[-1], cfg, specs=specs)
    assert candidates == []
    assert not mat["valid_mask"].iat[len(df) - 1]
