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


def test_bridge_get_row_meta_reads_values():
    tracker = RejectTracker(log_enabled=False, stats_enabled=True)
    strategy = SimpleNamespace(
        cfg=SimpleNamespace(),
        rejections=tracker,
    )
    bridge = ZeroCopyBridge(strategy)

    idx = pd.date_range("2024-01-01", periods=2, freq="5min")
    df = pd.DataFrame(
        {
            "date": idx,
            "_signal_id": [1, 2],
            "_signal_score": [0.1, 0.2],
            "_signal_raw_score": [0.3, 0.4],
            "_signal_rr_ratio": [1.1, 1.2],
            "_signal_sl_pct": [0.01, 0.02],
            "_signal_tp_pct": [0.03, 0.04],
            "_signal_plan_atr_pct": [0.05, 0.06],
        }
    )
    bridge.bind_df("BTC/USDT", df)

    meta = bridge.get_row_meta("BTC/USDT", idx[1])
    assert meta["signal_id"] == 2.0
    assert meta["expected_edge"] == 0.2
    assert meta["sl_pct"] == 0.02


def test_bridge_get_side_meta_respects_direction():
    tracker = RejectTracker(log_enabled=False, stats_enabled=True)
    hub = SimpleNamespace(meta_for_id=lambda *_: SimpleNamespace(direction="long"))
    strategy = SimpleNamespace(cfg=SimpleNamespace(), rejections=tracker, hub=hub)
    bridge = ZeroCopyBridge(strategy)

    idx = pd.date_range("2024-01-01", periods=2, freq="5min")
    df = pd.DataFrame({"date": idx})
    bridge.bind_df("BTC/USDT", df)

    payloads_long = [[{"signal_id": 1, "raw_score": 0.2, "rr_ratio": 1.0, "expected_edge": 0.5, "sl_pct": 0.01, "tp_pct": 0.02, "plan_atr_pct": 0.03}], []]
    payloads_short = [[], []]
    pool = PoolBuffer.from_payloads(payloads_long, payloads_short, slots=1, signal_id_fn=lambda p, _: p.get("signal_id"))
    bridge.bind_pool_buffer("BTC/USDT", pool)

    assert bridge.get_side_meta("BTC/USDT", idx[0], "sell", None) is None
    meta = bridge.get_side_meta("BTC/USDT", idx[0], "buy", "1")
    assert meta is not None
    assert meta[0]["signal_id"] == 1
