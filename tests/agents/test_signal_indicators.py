"""Tests for indicator helper utilities."""

from __future__ import annotations

import pandas as pd

from user_data.strategies.agents.signals.indicators import _bars_since_new_extreme


def test_newbars_high_distance():
    series = pd.Series([1, 2, 3, 2, 1, 4, 2], name="high")
    result = _bars_since_new_extreme(series, mode="high")
    assert result.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 6.0, 1.0]


def test_newbars_low_distance():
    series = pd.Series([5, 4, 3, 4, 5, 2], name="low")
    result = _bars_since_new_extreme(series, mode="low")
    assert result.tolist() == [1.0, 2.0, 3.0, 1.0, 1.0, 6.0]
