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
