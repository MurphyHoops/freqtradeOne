from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from user_data.strategies.vectorized.matrix_engine import MatrixEngine


class _DummyHub:
    def __init__(self) -> None:
        self._map = {
            ("sig_a", None, "long"): 1,
            ("sig_b", None, "short"): 2,
        }

    def signal_id_for(self, name, timeframe, direction):
        return self._map.get((name, timeframe, direction))


def test_matrix_engine_signal_id_matches_top_edge():
    engine = MatrixEngine(strategy=SimpleNamespace(cfg=None), hub=_DummyHub(), bridge=SimpleNamespace())
    df = pd.DataFrame(index=[0])
    payloads_long = [
        [
            {
                "direction": "long",
                "kind": "sig_a",
                "timeframe": None,
                "expected_edge": 0.2,
            }
        ]
    ]
    payloads_short = [
        [
            {
                "direction": "short",
                "kind": "sig_b",
                "timeframe": None,
                "expected_edge": 0.5,
            }
        ]
    ]
    engine._apply_payloads(df, payloads_long, payloads_short)

    assert int(df["_signal_id"].iat[0]) == 2
    assert df["enter_tag"].iat[0] == "2"


def test_matrix_engine_signal_id_matches_top_edge_from_arrays():
    engine = MatrixEngine(strategy=SimpleNamespace(cfg=None), hub=_DummyHub(), bridge=SimpleNamespace())
    df = pd.DataFrame(index=[0])
    field_len = len(engine._pool_schema.fields)
    long_array = np.full((1, 1, field_len), np.nan, dtype=float)
    short_array = np.full((1, 1, field_len), np.nan, dtype=float)
    idx = engine._pool_schema.index
    long_array[0, 0, idx["signal_id"]] = 1
    long_array[0, 0, idx["expected_edge"]] = 0.2
    short_array[0, 0, idx["signal_id"]] = 2
    short_array[0, 0, idx["expected_edge"]] = 0.5

    engine._apply_pool_arrays(df, long_array, short_array)

    assert int(df["_signal_id"].iat[0]) == 2
    assert df["enter_tag"].iat[0] == "2"
