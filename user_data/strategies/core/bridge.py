from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from .rejections import RejectReason

from ..vectorized.pool_buffer import PoolBuffer


class ZeroCopyBridge:
    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy
        self._frames: Dict[str, pd.DataFrame] = {}
        self._times: Dict[str, np.ndarray] = {}
        self._row_map: Dict[str, Dict[int, int]] = {}
        self._pool_buffers: Dict[str, PoolBuffer] = {}

    def align_informative(self, df: pd.DataFrame, pair: str) -> Dict[str, pd.DataFrame]:
        return self._strategy._aligned_informative_for_df(pair, df)

    def bind_df(self, pair: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self._frames.pop(pair, None)
            self._times.pop(pair, None)
            self._row_map.pop(pair, None)
            self._pool_buffers.pop(pair, None)
            return
        self._frames[pair] = df
        time_series = df["date"] if "date" in df.columns else df.index
        ts = pd.to_datetime(time_series, errors="coerce")
        ts_int = ts.astype("int64", copy=False)
        self._times[pair] = np.asarray(ts_int)
        row_map: Dict[int, int] = {}
        for idx, ts_val in enumerate(self._times[pair]):
            if pd.isna(ts_val):
                continue
            row_map[int(ts_val)] = idx
        self._row_map[pair] = row_map

    def bind_pool_buffer(self, pair: str, buffer: PoolBuffer) -> None:
        self._pool_buffers[pair] = buffer

    def _row_from_time(self, pair: str, current_time) -> Optional[int]:
        row_map = self._row_map.get(pair)
        if not row_map:
            return None
        try:
            target = pd.Timestamp(current_time).value
        except Exception:
            return None
        row = row_map.get(int(target))
        if row is None:
            tracker = getattr(self._strategy, "rejections", None)
            if tracker is not None:
                try:
                    tracker.record(
                        RejectReason.TIME_ALIGNMENT,
                        pair=pair,
                        context={"time": str(current_time)},
                    )
                except Exception:
                    pass
        return row

    def get_candidates(
        self,
        pair: str,
        current_time,
        side: str,
    ) -> list[Dict[str, float]]:
        row = self._row_from_time(pair, current_time)
        if row is None:
            return []
        buffer = self._pool_buffers.get(pair)
        if buffer is None:
            return []
        out = buffer.candidates_for(row, side)
        return out
