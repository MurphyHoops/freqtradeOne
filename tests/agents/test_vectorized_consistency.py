# -*- coding: utf-8 -*-
"""Vectorized backtest consistency and caching guardrails."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import user_data.strategies.TaxBrainV29 as strat_mod
from user_data.strategies.TaxBrainV29 import TaxBrainV29
from user_data.strategies.agents.signals import builder, vectorized
from user_data.strategies.config.v29_config import V29Config


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
        }
    )


def _info_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=3, freq="1h")
    return pd.DataFrame({"date": dates})


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


def test_prefilter_specs_matches_default():
    cfg = V29Config()
    df = _base_frame()
    vectorized.add_derived_factor_columns(df, (None,))

    default_mask = vectorized.prefilter_signal_mask(df, cfg)
    specs = builder.REGISTRY.all()
    specs_mask = vectorized.prefilter_signal_mask(df, cfg, specs=specs)

    assert default_mask.equals(specs_mask)


def test_build_candidates_specs_matches_default():
    cfg = V29Config()
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
    vectorized_strat = TaxBrainV29(params)
    vectorized_strat.dp = _RunmodeDP(fake.BACKTEST)

    params["strategy_params"]["vectorized_entry_backtest"] = False
    non_vectorized_strat = TaxBrainV29(params)
    non_vectorized_strat.dp = _RunmodeDP(fake.BACKTEST)

    df = _base_frame()
    vec_df = vectorized_strat.populate_entry_trend(df.copy(), {"pair": "BTC/USDT"})
    non_df = non_vectorized_strat.populate_entry_trend(df.copy(), {"pair": "BTC/USDT"})

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
    strategy = TaxBrainV29(params)
    strategy.dp = SimpleNamespace(runmode=fake.BACKTEST, get_informative_dataframe=_DummyDP(_info_frame()).get_informative_dataframe)

    def _no_vectorized(*args, **kwargs):
        raise AssertionError("vectorized prefilter should be skipped when informative is unmerged")

    monkeypatch.setattr(vectorized, "prefilter_signal_mask", _no_vectorized)

    df = _base_frame()
    strategy.populate_entry_trend(df.copy(), {"pair": "BTC/USDT"})


def test_build_signal_matrices_matches_builder():
    cfg = V29Config()
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
