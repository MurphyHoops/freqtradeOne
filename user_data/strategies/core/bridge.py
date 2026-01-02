from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict
import ctypes
import os
import platform
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
        self._aligned_base_ts_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._aligned_info_ts_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._informative_gc_last_ts: float = 0.0
        self._informative_gc_interval_sec: int = 900
        system_cfg = getattr(getattr(strategy, "cfg", None), "system", None)
        self._informative_gc_mem_pct = float(
            getattr(system_cfg, "informative_gc_mem_pct", 0.85) or 0.0
        ) if system_cfg else 0.85
        self._informative_gc_force_pct = float(
            getattr(system_cfg, "informative_gc_force_pct", 0.92) or 0.0
        ) if system_cfg else 0.92
        if self._informative_gc_mem_pct > 1.0:
            self._informative_gc_mem_pct /= 100.0
        if self._informative_gc_force_pct > 1.0:
            self._informative_gc_force_pct /= 100.0

    def align_informative(self, df: pd.DataFrame, pair: str) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if df is None or df.empty:
            return out
        system_cfg = getattr(self._strategy.cfg, "system", None)
        merge_info = bool(getattr(system_cfg, "merge_informative_into_base", False)) if system_cfg else False
        if merge_info and self._strategy._is_backtest_like_runmode():
            return out
        base_source = df["date"] if "date" in df.columns else df.index
        if len(base_source):
            try:
                last_ts = base_source.iloc[-1]
            except Exception:
                last_ts = base_source[-1]
        else:
            last_ts = None
        max_entries = int(getattr(system_cfg, "aligned_info_cache_max_entries", 0) or 0) if system_cfg else 0
        base_cache_key = (pair, len(df), str(last_ts))
        base_ts = self._aligned_base_ts_cache.get(base_cache_key)
        if base_ts is not None:
            try:
                self._aligned_base_ts_cache.move_to_end(base_cache_key)
            except Exception:
                pass
        else:
            base_time = pd.to_datetime(base_source)
            base_ts = base_time.astype("int64", copy=False).to_numpy(copy=False)
            self._aligned_base_ts_cache[base_cache_key] = base_ts
            if max_entries > 0:
                while len(self._aligned_base_ts_cache) > max_entries:
                    try:
                        self._aligned_base_ts_cache.popitem(last=False)
                    except Exception:
                        break
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
            info_source = info_df["date"] if "date" in info_df.columns else info_df.index
            if len(info_source):
                try:
                    last_info_ts = info_source.iloc[-1]
                except Exception:
                    last_info_ts = info_source[-1]
            else:
                last_info_ts = None
            info_cache_key = (pair, tf, len(info_df), str(last_info_ts))
            info_ts = self._aligned_info_ts_cache.get(info_cache_key)
            if info_ts is not None:
                try:
                    self._aligned_info_ts_cache.move_to_end(info_cache_key)
                except Exception:
                    pass
            else:
                info_time = pd.to_datetime(info_source)
                info_ts = info_time.astype("int64", copy=False).to_numpy(copy=False)
                self._aligned_info_ts_cache[info_cache_key] = info_ts
                if max_entries > 0:
                    while len(self._aligned_info_ts_cache) > max_entries:
                        try:
                            self._aligned_info_ts_cache.popitem(last=False)
                        except Exception:
                            break
            info_vals = info_df.to_numpy(copy=False)
            min_int64 = np.iinfo("int64").min
            info_valid = info_ts != min_int64
            if not info_valid.all():
                info_ts = info_ts[info_valid]
                info_vals = info_vals[info_valid]
            if info_ts.size == 0:
                continue
            if info_ts.size > 1 and np.any(info_ts[1:] < info_ts[:-1]):
                order = np.argsort(info_ts, kind="mergesort")
                info_ts = info_ts[order]
                info_vals = info_vals[order]
            idxs = np.searchsorted(info_ts, base_ts, side="right") - 1
            aligned_vals = np.full((len(base_ts), info_vals.shape[1]), np.nan, dtype=info_vals.dtype)
            valid = idxs >= 0
            if valid.any():
                aligned_vals[valid] = info_vals[idxs[valid]]
            aligned = pd.DataFrame(aligned_vals, columns=info_df.columns, index=df.index)
            out[tf] = aligned
            self._aligned_info_cache[cache_key] = aligned
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
        self._times[pair] = np.asarray(ts_int, dtype="int64")
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
        for key in [key for key in list(self._aligned_info_cache.keys()) if key[0] == pair]:
            self._aligned_info_cache.pop(key, None)
        for key in [key for key in list(self._aligned_base_ts_cache.keys()) if key[0] == pair]:
            self._aligned_base_ts_cache.pop(key, None)
        for key in [key for key in list(self._aligned_info_ts_cache.keys()) if key[0] == pair]:
            self._aligned_info_ts_cache.pop(key, None)

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

    def gc_pair_views(self, current_whitelist) -> None:
        if not current_whitelist:
            return
        allowed = {str(pair) for pair in current_whitelist if pair}
        if not allowed:
            return
        stale_pairs = [pair for pair in list(self._frames.keys()) if pair not in allowed]
        if not stale_pairs:
            return
        for pair in stale_pairs:
            self._frames.pop(pair, None)
            self._times.pop(pair, None)
            self._pool_buffers.pop(pair, None)
            self._views.pop(pair, None)
            self._col_index.pop(pair, None)
            self._col_names.pop(pair, None)
            self._meta_views.pop(pair, None)
            self._meta_col_index.pop(pair, None)
            self._meta_col_names.pop(pair, None)
            self.clear_informative_pair(pair)
        for key in [key for key in list(self._aligned_info_cache.keys()) if key[0] in stale_pairs]:
            self._aligned_info_cache.pop(key, None)
        for key in [key for key in list(self._aligned_base_ts_cache.keys()) if key[0] in stale_pairs]:
            self._aligned_base_ts_cache.pop(key, None)
        for key in [key for key in list(self._aligned_info_ts_cache.keys()) if key[0] in stale_pairs]:
            self._aligned_info_ts_cache.pop(key, None)

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

    def _force_gc_informative_cache(self) -> None:
        self._informative_cache.clear()
        self._informative_last.clear()
        self._informative_cache_order.clear()
        self._aligned_info_cache.clear()
        self._aligned_base_ts_cache.clear()
        self._aligned_info_ts_cache.clear()

    def _memory_pressure_ratio(self) -> Optional[float]:
        try:
            system = platform.system().lower()
        except Exception:
            system = ""
        if system == "windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            try:
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) == 0:
                    return None
            except Exception:
                return None
            return max(0.0, min(1.0, float(status.dwMemoryLoad) / 100.0))

        meminfo = "/proc/meminfo"
        if os.path.isfile(meminfo):
            total_kb = None
            avail_kb = None
            try:
                with open(meminfo, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.startswith("MemTotal:"):
                            total_kb = float(line.split()[1])
                        elif line.startswith("MemAvailable:"):
                            avail_kb = float(line.split()[1])
                        if total_kb is not None and avail_kb is not None:
                            break
            except Exception:
                return None
            if total_kb and avail_kb is not None and total_kb > 0:
                used = max(0.0, total_kb - avail_kb)
                return max(0.0, min(1.0, used / total_kb))
        return None

    def maybe_gc_informative_cache(self, current_whitelist) -> None:
        now_ts = time.time()
        mem_ratio = self._memory_pressure_ratio()
        if (
            mem_ratio is not None
            and self._informative_gc_force_pct > 0
            and mem_ratio >= self._informative_gc_force_pct
        ):
            self._force_gc_informative_cache()
            self.gc_pair_views(current_whitelist)
            self._informative_gc_last_ts = now_ts
            return
        if (
            mem_ratio is not None
            and self._informative_gc_mem_pct > 0
            and mem_ratio >= self._informative_gc_mem_pct
        ):
            self.gc_informative_cache(current_whitelist)
            self.gc_pair_views(current_whitelist)
            self._informative_gc_last_ts = now_ts
            return

        interval = self._informative_gc_interval_sec
        if (now_ts - self._informative_gc_last_ts) < interval:
            return
        try:
            self.gc_informative_cache(current_whitelist)
            self.gc_pair_views(current_whitelist)
        finally:
            self._informative_gc_last_ts = now_ts

    def _row_from_time(self, pair: str, current_time) -> Optional[int]:
        times = self._times.get(pair)
        if times is None or len(times) == 0:
            return None
        try:
            target = int(pd.Timestamp(current_time).value)
        except Exception:
            return None
        idx = int(np.searchsorted(times, target))
        row = idx if idx < len(times) and times[idx] == target else None
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
        values = np.full(len(_META_COLUMNS), np.nan, dtype=float)
        if view is not None and col_index is not None:
            idxs = np.array([col_index.get(name, -1) for name in _META_COLUMNS], dtype=int)
            row_vals = view[row]
            valid = (idxs >= 0) & (idxs < row_vals.shape[0])
            if valid.any():
                values[valid] = row_vals[idxs[valid]]
            values = np.where(np.isfinite(values), values, np.nan)
        else:
            view = self._views.get(pair)
            col_index = self._col_index.get(pair)
            if view is None or col_index is None:
                return None
            row_vals = view[row]
            for idx, name in enumerate(_META_COLUMNS):
                col_idx = col_index.get(name, -1)
                if col_idx < 0:
                    continue
                try:
                    raw = row_vals[col_idx]
                except Exception:
                    continue
                try:
                    value = float(raw)
                except Exception:
                    continue
                if np.isfinite(value):
                    values[idx] = value
        keys = (
            "signal_id",
            "expected_edge",
            "raw_score",
            "rr_ratio",
            "sl_pct",
            "tp_pct",
            "plan_atr_pct",
        )
        out: Dict[str, float | None] = {}
        for idx, key in enumerate(keys):
            value = values[idx]
            out[key] = float(value) if np.isfinite(value) else None
        return out
