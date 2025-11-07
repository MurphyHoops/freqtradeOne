"""Unit tests for signal factor computations."""

import math

from user_data.strategies.agents.signal.factors import FactorBank


def test_factor_bank_atomic_and_composite_values():
    row = {
        "close": 100.0,
        "ema_fast": 98.0,
        "ema_slow": 95.0,
        "rsi": 30.0,
        "adx": 25.0,
        "atr_pct": 0.02,
    }
    fb = FactorBank(row)

    assert fb.get("CLOSE") == 100.0
    assert fb.get("EMA_TREND") == 1.0
    delta = fb.get("DELTA_CLOSE_EMAFAST_PCT")
    assert math.isclose(delta, 100.0 / 98.0 - 1.0)


def test_factor_unknown_name_raises_keyerror():
    fb = FactorBank({"close": 100.0})
    try:
        fb.get("UNKNOWN")
    except KeyError:
        assert True
    else:
        assert False, "Expected KeyError for unknown factor"


def test_factor_bank_supports_informative_timeframes():
    row = {
        "close": 100.0,
        "ema_fast": 98.0,
        "ema_slow": 95.0,
        "rsi": 30.0,
        "adx": 25.0,
        "atr_pct": 0.02,
    }
    informative = {
        "1h": {
            "close": 110.0,
            "ema_fast_1h": 105.0,
            "ema_slow_1h": 112.0,
            "rsi_1h": 55.0,
            "adx_1h": 35.0,
            "atr_pct_1h": 0.03,
        }
    }
    fb = FactorBank(row, informative=informative)

    assert fb.get("CLOSE@1h") == 110.0
    trend = fb.get("EMA_TREND@1h")
    assert trend == -1.0  # fast < slow on informative row
