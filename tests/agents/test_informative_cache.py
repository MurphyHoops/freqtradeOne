"""验证 TaxBrainV29 多周期缓存/对齐逻辑。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from user_data.strategies.TaxBrainV29 import TaxBrainV29


class _DummyDataProvider:
    def __init__(self, info_df: pd.DataFrame) -> None:
        self._info_df = info_df
        self.info_calls = 0

    def current_whitelist(self):
        return ["BTC/USDT"]

    def get_informative_dataframe(self, pair: str, timeframe: str):
        assert pair == "BTC/USDT"
        assert timeframe == "1h"
        self.info_calls += 1
        return self._info_df.copy()


def _base_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="5min")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100 + i for i in range(5)],
            "high": [101 + i for i in range(5)],
            "low": [99 + i for i in range(5)],
            "close": [100.5 + i for i in range(5)],
            "volume": [1000] * 5,
        }
    )


def _info_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-12-31", periods=3, freq="1h")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100] * 3,
            "high": [101] * 3,
            "low": [99] * 3,
            "close": [100.5] * 3,
            "volume": [500] * 3,
            "atr": [1.0, 1.2, 1.1],
            "atr_pct": [0.01, 0.012, 0.011],
        }
    )


def test_informative_cache_reuses_frames(tmp_path: Path):
    params = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV29(params)
    strategy._informative_timeframes = ("1h",)

    info_df = _info_frame()
    dp = _DummyDataProvider(info_df)
    strategy.dp = dp

    base_df = _base_frame()
    strategy.populate_indicators(base_df.copy(), {"pair": "BTC/USDT"})

    assert dp.info_calls == 1
    assert "BTC/USDT" in strategy._informative_cache
    cached = strategy._informative_cache["BTC/USDT"]["1h"]
    assert cached.equals(info_df.tail(len(cached)))

    aligned = strategy._aligned_informative_for_df("BTC/USDT", base_df.copy())
    assert dp.info_calls == 1, "Aligned fetch should reuse cached dataframe"
    assert "1h" in aligned
    assert len(aligned["1h"]) == len(base_df)


def test_aligned_info_cache_lru_eviction(tmp_path: Path):
    params = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
            "aligned_info_cache_max_entries": 2,
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV29(params)
    strategy._informative_timeframes = ("1h",)

    info_df = _info_frame()
    dp = _DummyDataProvider(info_df)
    strategy.dp = dp

    base_df = _base_frame()
    df1 = base_df.copy()
    df2 = base_df.copy()
    df2["date"] = df2["date"] + pd.Timedelta(minutes=5)
    df3 = base_df.copy()
    df3["date"] = df3["date"] + pd.Timedelta(minutes=10)

    strategy._aligned_informative_for_df("BTC/USDT", df1)
    strategy._aligned_informative_for_df("BTC/USDT", df2)
    strategy._aligned_informative_for_df("BTC/USDT", df3)

    cache = strategy._aligned_info_cache
    assert len(cache) <= 2
    key2 = ("BTC/USDT", "1h", len(df2), str(df2["date"].iloc[-1]))
    key3 = ("BTC/USDT", "1h", len(df3), str(df3["date"].iloc[-1]))
    assert key2 in cache
    assert key3 in cache
