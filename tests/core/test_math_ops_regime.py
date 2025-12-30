from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.core import math_ops


def test_regime_scalar_matches_vector_tail():
    close = pd.Series(np.linspace(100.0, 200.0, 80))
    hurst_vec = math_ops.calculate_hurst_vec(close, window=60, min_points=20)
    hurst_scalar = math_ops.calculate_hurst_scalar(close.tolist(), window=60, min_points=20)
    assert np.isfinite(hurst_vec.iloc[-1])
    assert hurst_vec.iloc[-1] == pytest.approx(hurst_scalar, rel=1e-6, abs=1e-6)

    adx = pd.Series(np.linspace(10.0, 40.0, 80))
    z_vec = math_ops.calculate_adx_zsig_vec(adx, window=60, min_points=20)
    z_scalar = math_ops.calculate_adx_zsig_scalar(adx.tolist(), float(adx.iloc[-1]), window=60, min_points=20)
    assert np.isfinite(z_vec.iloc[-1])
    assert z_vec.iloc[-1] == pytest.approx(z_scalar, rel=1e-6, abs=1e-6)


def test_regime_factor_vector_matches_scalar_tail():
    hurst = pd.Series([0.8] * 50)
    zsig = pd.Series([0.2] * 50)
    vec = math_ops.calculate_regime_factor_vec("mean_rev_long", hurst, zsig)
    scalar = math_ops.calculate_regime_factor_vec("mean_rev_long", 0.8, 0.2)
    assert vec.iloc[-1] == pytest.approx(float(scalar), rel=1e-6, abs=1e-6)


def test_regime_factor_nan_defaults_to_neutral():
    hurst = pd.Series([np.nan, np.nan])
    zsig = pd.Series([np.nan, np.nan])
    vec = math_ops.calculate_regime_factor_vec("trend_short", hurst, zsig)
    assert vec.iloc[-1] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    scalar = math_ops.calculate_regime_factor_vec("trend_short", float("nan"), float("nan"))
    assert float(scalar) == pytest.approx(1.0, rel=1e-6, abs=1e-6)
