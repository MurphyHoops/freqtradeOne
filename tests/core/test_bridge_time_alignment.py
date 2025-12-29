from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from user_data.strategies.core.bridge import ZeroCopyBridge
from user_data.strategies.core.rejections import RejectReason, RejectTracker
from user_data.strategies.vectorized.pool_buffer import PoolBuffer


def test_bridge_strict_time_alignment_records_rejection():
    tracker = RejectTracker(log_enabled=False, stats_enabled=True)
    strategy = SimpleNamespace(
        cfg=SimpleNamespace(),
        rejections=tracker,
        _aligned_informative_for_df=lambda *args, **kwargs: {},
    )
    bridge = ZeroCopyBridge(strategy)

    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2, freq="5min")})
    bridge.bind_df("BTC/USDT", df)

    pool = PoolBuffer.from_payloads([[], []], [[], []], slots=1, signal_id_fn=lambda *_: None)
    bridge.bind_pool_buffer("BTC/USDT", pool)

    bad_time = df["date"].iloc[-1] + pd.Timedelta(minutes=1)
    assert bridge.get_candidates("BTC/USDT", bad_time, "buy") == []
    assert tracker.snapshot()[RejectReason.TIME_ALIGNMENT] == 1


def test_bridge_datetimeindex_bind_df_does_not_crash():
    tracker = RejectTracker(log_enabled=False, stats_enabled=True)
    strategy = SimpleNamespace(
        cfg=SimpleNamespace(),
        rejections=tracker,
        _aligned_informative_for_df=lambda *args, **kwargs: {},
    )
    bridge = ZeroCopyBridge(strategy)

    idx = pd.date_range("2024-01-01", periods=2, freq="5min")
    df = pd.DataFrame(index=idx)
    bridge.bind_df("BTC/USDT", df)

    pool = PoolBuffer.from_payloads([[], []], [[], []], slots=1, signal_id_fn=lambda *_: None)
    bridge.bind_pool_buffer("BTC/USDT", pool)

    good_time = idx[0]
    assert bridge.get_candidates("BTC/USDT", good_time, "buy") == []
    assert RejectReason.TIME_ALIGNMENT not in tracker.snapshot()
