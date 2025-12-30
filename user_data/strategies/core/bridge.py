from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import time

import pandas as pd
import numpy as np

from .rejections import RejectReason

from ..vectorized.pool_buffer import PoolBuffer

_META_COLUMNS: tuple[str, ...] = (
    "_signal_id",
    "_signal_score",
    "_signal_raw_score",
    "_signal_rr_ratio",
    "_signal_sl_pct",
    "_signal_tp_pct",
    "_signal_plan_atr_pct",
)


class ZeroCopyBridge:
    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy
        self._frames: Dict[str, pd.DataFrame] = {}
        self._times: Dict[str, np.ndarray] = {}
        self._row_map: Dict[str, Dict[int, int]] = {}
        self._pool_buffers: Dict[str, PoolBuffer] = {}
        self._views: Dict[str, np.ndarray] = {}
        self._col_index: Dict[str, Dict[str, int]] = {}
        self._col_names: Dict[str, tuple[str, ...]] = {}
        self._meta_views: Dict[str, np.ndarray] = {}
        self._meta_col_index: Dict[str, Dict[str, int]] = {}
        self._meta_col_names: Dict[str, tuple[str, ...]] = {}
        self._informative_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._informative_cache_order: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._informative_last: Dict[str, Dict[str, pd.Series]] = {}
        self._aligned_info_cache: OrderedDict[tuple, pd.DataFrame] = OrderedDict()
        self._informative_gc_last_ts: float = 0.0
        self._informative_gc_interval_sec: int = 900

    def align_informative(self, df: pd.DataFrame, pair: str) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if df is None or df.empty:
            return out
        system_cfg = getattr(self._strategy.cfg, "system", None)
        merge_info = bool(getattr(system_cfg, "merge_informative_into_base", False)) if system_cfg else False
        if merge_info and self._strategy._is_backtest_like_runmode():
            return out
        base_time = pd.to_datetime(df["date"]) if "date" in df.columns else pd.to_datetime(df.index)
        last_ts = base_time.iloc[-1] if len(base_time) else None
        max_entries = int(getattr(system_cfg, "aligned_info_cache_max_entries", 0) or 0) if system_cfg else 0
        for tf in getattr(self._strategy, "_informative_timeframes", []):
            cache_key = (pair, tf, len(df), str(last_ts))
            cached = self._aligned_info_cache.get(cache_key)
            if cached is not None:
                try:
                    self._aligned_info_cache.move_to_end(cache_key)
                except Exception:
                    pass
                out[tf] = cached
                continue
            info_df = self.get_informative_dataframe(pair, tf)
            if info_df is None or info_df.empty:
                continue
            left = pd.DataFrame({"_t": base_time})
            right = info_df.assign(
                _tinfo=pd.to_datetime(info_df["date"]) if "date" in info_df.columns else pd.to_datetime(info_df.index)
            )
            merged = pd.merge_asof(
                left.sort_values("_t"),
                right.sort_values("_tinfo"),
                left_on="_t",
                right_on="_tinfo",
                direction="backward",
            )
            merged.index = df.index
            out[tf] = merged
            self._aligned_info_cache[cache_key] = merged
            if max_entries > 0:
                while len(self._aligned_info_cache) > max_entries:
                    try:
                        self._aligned_info_cache.popitem(last=False)
                    except Exception:
                        break
        return out

    def bind_df(self, pair: str, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self._frames.pop(pair, None)
            self._times.pop(pair, None)
            self._row_map.pop(pair, None)
            self._pool_buffers.pop(pair, None)
            self._views.pop(pair, None)
            self._col_index.pop(pair, None)
            self._col_names.pop(pair, None)
            self._meta_views.pop(pair, None)
            self._meta_col_index.pop(pair, None)
            self._meta_col_names.pop(pair, None)
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
        self._views[pair] = df.to_numpy(copy=False)
        col_names = tuple(df.columns)
        if col_names != self._col_names.get(pair):
            self._col_index[pair] = {name: idx for idx, name in enumerate(col_names)}
            self._col_names[pair] = col_names
        meta_cols = [col for col in _META_COLUMNS if col in df.columns]
        if meta_cols:
            meta_names = tuple(meta_cols)
            self._meta_views[pair] = df.loc[:, meta_cols].to_numpy(copy=False)
            if meta_names != self._meta_col_names.get(pair):
                self._meta_col_index[pair] = {name: idx for idx, name in enumerate(meta_cols)}
                self._meta_col_names[pair] = meta_names
        else:
            self._meta_views.pop(pair, None)
            self._meta_col_index.pop(pair, None)
            self._meta_col_names.pop(pair, None)

    def bind_pool_buffer(self, pair: str, buffer: PoolBuffer) -> None:
        self._pool_buffers[pair] = buffer

    def get_informative_dataframe(self, pair: str, timeframe: str) -> Optional[pd.DataFrame]:
        system_cfg = getattr(self._strategy.cfg, "system", None)
        max_entries = int(getattr(system_cfg, "informative_cache_max_entries", 0) or 0) if system_cfg else 0
        cached = self._informative_cache.get(pair, {}).get(timeframe)
        if cached is not None and not cached.empty:
            self._touch_informative_key(pair, timeframe, max_entries)
            return cached
        getter = getattr(self._strategy.dp, "get_informative_dataframe", None)
        if callable(getter):
            result = getter(pair, timeframe)
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, pd.DataFrame) and not result.empty:
                prepared = self._strategy._prepare_informative_frame(result, timeframe)
                self._informative_cache.setdefault(pair, {})[timeframe] = prepared
                self._touch_informative_key(pair, timeframe, max_entries)
                return prepared
            return self._strategy._prepare_informative_frame(result, timeframe)
        result = self._strategy.dp.get_analyzed_dataframe(pair, timeframe)
        if isinstance(result, tuple):
            result = result[0]
        if isinstance(result, pd.DataFrame) and not result.empty:
            prepared = self._strategy._prepare_informative_frame(result, timeframe)
            self._informative_cache.setdefault(pair, {})[timeframe] = prepared
            self._touch_informative_key(pair, timeframe, max_entries)
            return prepared
        return self._strategy._prepare_informative_frame(result, timeframe)

    def set_informative_last(self, pair: str, rows: Dict[str, pd.Series]) -> None:
        if rows:
            self._informative_last[pair] = rows
        else:
            self._informative_last.pop(pair, None)

    def clear_informative_pair(self, pair: str) -> None:
        self._informative_cache.pop(pair, None)
        self._informative_last.pop(pair, None)
        for key in [key for key in self._informative_cache_order.keys() if key[0] == pair]:
            self._informative_cache_order.pop(key, None)

    def get_informative_row(self, pair: str, timeframe: str) -> Optional[pd.Series]:
        return self._informative_last.get(pair, {}).get(timeframe)

    def gc_informative_cache(self, current_whitelist) -> None:
        if not current_whitelist:
            return
        allowed = {str(pair) for pair in current_whitelist if pair}
        if not allowed:
            return
        stale_pairs = [pair for pair in list(self._informative_cache.keys()) if pair not in allowed]
        for pair in stale_pairs:
            self.clear_informative_pair(pair)

    def _touch_informative_key(self, pair: str, timeframe: str, max_entries: int) -> None:
        if max_entries <= 0:
            return
        key = (pair, timeframe)
        if key in self._informative_cache_order:
            self._informative_cache_order.move_to_end(key)
        else:
            self._informative_cache_order[key] = None
        while len(self._informative_cache_order) > max_entries:
            stale_key, _ = self._informative_cache_order.popitem(last=False)
            stale_pair, stale_tf = stale_key
            pair_cache = self._informative_cache.get(stale_pair)
            if pair_cache:
                pair_cache.pop(stale_tf, None)
                if not pair_cache:
                    self._informative_cache.pop(stale_pair, None)

    def maybe_gc_informative_cache(self, current_whitelist) -> None:
        interval = self._informative_gc_interval_sec
        now_ts = time.time()
        if (now_ts - self._informative_gc_last_ts) < interval:
            return
        try:
            self.gc_informative_cache(current_whitelist)
        finally:
            self._informative_gc_last_ts = now_ts

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

    def get_row_index(self, pair: str, current_time) -> Optional[int]:
        return self._row_from_time(pair, current_time)

    def get_candidates(
        self,
        pair: str,
        current_time,
        side: str,
        row_idx: Optional[int] = None,
    ) -> list[Dict[str, float]]:
        row = row_idx if row_idx is not None else self._row_from_time(pair, current_time)
        if row is None:
            return []
        buffer = self._pool_buffers.get(pair)
        if buffer is None:
            return []
        out = buffer.candidates_for(row, side)
        return out

    def get_candidate_by_id(
        self,
        pair: str,
        current_time,
        signal_id: int,
        side: str,
        row_idx: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        row = row_idx if row_idx is not None else self._row_from_time(pair, current_time)
        if row is None:
            return None
        buffer = self._pool_buffers.get(pair)
        if buffer is None:
            return None
        for cand in buffer.candidates_for(row, side):
            if int(cand.get("signal_id", 0)) == int(signal_id):
                return cand
        return None

    def get_side_meta(
        self,
        pair: str,
        current_time,
        side: str,
        entry_tag: Optional[str],
    ) -> Optional[Tuple[Dict[str, float], Any]]:
        row = self._row_from_time(pair, current_time)
        if row is None:
            return None
        buffer = self._pool_buffers.get(pair)
        if buffer is None:
            return None
        desired_dir = "long" if str(side).lower() in ("buy", "long") else "short"
        tag_id = None
        if entry_tag:
            try:
                tag_id = int(entry_tag)
            except Exception:
                return None
        if tag_id:
            candidate = None
            for cand in buffer.candidates_for(row, side):
                if int(cand.get("signal_id", 0)) == tag_id:
                    candidate = cand
                    break
        else:
            candidates = buffer.candidates_for(row, side)
            candidate = candidates[0] if candidates else None
        if not candidate:
            return None
        meta_info = None
        try:
            meta_info = self._strategy.hub.meta_for_id(int(candidate.get("signal_id", 0)))
        except Exception:
            meta_info = None
        if not meta_info:
            return None
        if str(getattr(meta_info, "direction", "")).lower() != desired_dir:
            return None
        return (dict(candidate), meta_info)

    def get_best_candidate(
        self,
        pair: str,
        current_time,
        side: str,
        row_idx: Optional[int] = None,
    ) -> Optional[Dict[str, float]]:
        row = row_idx if row_idx is not None else self._row_from_time(pair, current_time)
        if row is None:
            return None
        buffer = self._pool_buffers.get(pair)
        if buffer is None:
            return None
        candidates = buffer.candidates_for(row, side)
        return candidates[0] if candidates else None

    def get_row_meta(self, pair: str, current_time) -> Optional[Dict[str, float | None]]:
        row = self._row_from_time(pair, current_time)
        if row is None:
            return None
        view = self._meta_views.get(pair)
        col_index = self._meta_col_index.get(pair)
        if view is None or col_index is None:
            view = self._views.get(pair)
            col_index = self._col_index.get(pair)
        if view is None or col_index is None:
            return None
        def _take(name: str) -> Optional[float]:
            idx = col_index.get(name)
            if idx is None:
                return None
            try:
                value = float(view[row, idx])
            except Exception:
                return None
            if not np.isfinite(value):
                return None
            return value

        return {
            "signal_id": _take("_signal_id"),
            "expected_edge": _take("_signal_score"),
            "raw_score": _take("_signal_raw_score"),
            "rr_ratio": _take("_signal_rr_ratio"),
            "sl_pct": _take("_signal_sl_pct"),
            "tp_pct": _take("_signal_tp_pct"),
            "plan_atr_pct": _take("_signal_plan_atr_pct"),
        }
