"""Tests for indicator helper utilities."""

from __future__ import annotations

import pandas as pd

from user_data.strategies.config.v30_config import V30Config
from user_data.strategies.agents.signals.indicators import compute_indicators

from user_data.strategies.agents.signals.indicators import _bars_since_new_extreme


def test_newbars_high_distance():
    series = pd.Series([1, 2, 3, 2, 1, 4, 2], name="high")
    result = _bars_since_new_extreme(series, mode="high")
    assert result.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 6.0, 1.0]


def test_newbars_low_distance():
    series = pd.Series([5, 4, 3, 4, 5, 2], name="low")
    result = _bars_since_new_extreme(series, mode="low")
    assert result.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 6.0]


def test_compute_indicators_skips_empty_required():
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.5, 2.5],
            "low": [0.5, 1.5],
            "close": [1.2, 2.2],
            "volume": [10.0, 12.0],
        }
    )
    cfg = V30Config()
    out = compute_indicators(df.copy(), cfg, required=set())

    assert "rsi" not in out.columns
    assert "ema_fast" not in out.columns
    assert "ema_slow" not in out.columns
