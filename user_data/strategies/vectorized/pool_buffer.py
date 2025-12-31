from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional
import math

import numpy as np


DEFAULT_POOL_FIELDS = (
    "signal_id",
    "raw_score",
    "rr_ratio",
    "expected_edge",
    "sl_pct",
    "tp_pct",
    "plan_atr_pct",
)


@dataclass(frozen=True)
class PoolSchema:
    fields: tuple[str, ...] = DEFAULT_POOL_FIELDS

    @property
    def index(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.fields)}


class PoolBuffer:
    def __init__(self, rows: int, slots: int, schema: PoolSchema | None = None) -> None:
        self.schema = schema or PoolSchema()
        self.rows = int(rows)
        self.slots = int(slots)
        field_len = len(self.schema.fields)
        self._data: Dict[str, np.ndarray] = {
            "long": np.full((self.rows, self.slots, field_len), np.nan, dtype="float64"),
            "short": np.full((self.rows, self.slots, field_len), np.nan, dtype="float64"),
        }

    @classmethod
    def from_payloads(
        cls,
        payloads_long: list[list[dict]],
        payloads_short: list[list[dict]],
        slots: int,
        signal_id_fn: Callable[[dict, str], Optional[int]],
        schema: PoolSchema | None = None,
    ) -> "PoolBuffer":
        rows = max(len(payloads_long), len(payloads_short))
        buffer = cls(rows, slots, schema=schema)
        buffer._fill("long", payloads_long, signal_id_fn)
        buffer._fill("short", payloads_short, signal_id_fn)
        return buffer

    @classmethod
    def from_array_data(
        cls,
        long_data: np.ndarray,
        short_data: np.ndarray,
        schema: PoolSchema | None = None,
    ) -> "PoolBuffer":
        if long_data.ndim != 3 or short_data.ndim != 3:
            raise ValueError("PoolBuffer arrays must be 3D (rows, slots, fields).")
        if long_data.shape != short_data.shape:
            raise ValueError("PoolBuffer arrays must share the same shape.")
        rows, slots, _ = long_data.shape
        buffer = cls(rows, slots, schema=schema)
        buffer._data["long"] = long_data
        buffer._data["short"] = short_data
        return buffer

    def _fill(
        self,
        direction: str,
        payloads: list[list[dict]],
        signal_id_fn: Callable[[dict, str], Optional[int]],
    ) -> None:
        idx = self.schema.index
        out = self._data[direction]
        for row_idx, row_payloads in enumerate(payloads):
            if not row_payloads:
                continue
            for slot in range(min(self.slots, len(row_payloads))):
                payload = row_payloads[slot]
                sig_id = signal_id_fn(payload, direction)
                if not sig_id:
                    continue
                out[row_idx, slot, idx["signal_id"]] = float(sig_id)
                out[row_idx, slot, idx["raw_score"]] = _safe_float(payload.get("raw_score"))
                out[row_idx, slot, idx["rr_ratio"]] = _safe_float(payload.get("rr_ratio"))
                out[row_idx, slot, idx["expected_edge"]] = _safe_float(payload.get("expected_edge"))
                out[row_idx, slot, idx["sl_pct"]] = _safe_float(payload.get("sl_pct"))
                out[row_idx, slot, idx["tp_pct"]] = _safe_float(payload.get("tp_pct"))
                atr_val = payload.get("plan_atr_pct")
                out[row_idx, slot, idx["plan_atr_pct"]] = (
                    _safe_float(atr_val) if atr_val is not None else np.nan
                )

    def candidates_for(self, row_idx: int, side: str) -> list[Dict[str, float]]:
        direction = "long" if str(side).lower() in ("buy", "long") else "short"
        idx = self.schema.index
        out: list[Dict[str, float]] = []
        if direction not in self._data:
            return out
        data = self._data[direction]
        if row_idx < 0 or row_idx >= data.shape[0]:
            return out
        row = data[row_idx]
        for slot in range(self.slots):
            sig_id = float(row[slot, idx["signal_id"]])
            if not math.isfinite(sig_id) or sig_id <= 0:
                continue
            out.append(
                {
                    "signal_id": int(sig_id),
                    "raw_score": float(row[slot, idx["raw_score"]]),
                    "rr_ratio": float(row[slot, idx["rr_ratio"]]),
                    "expected_edge": float(row[slot, idx["expected_edge"]]),
                    "sl_pct": float(row[slot, idx["sl_pct"]]),
                    "tp_pct": float(row[slot, idx["tp_pct"]]),
                    "plan_atr_pct": float(row[slot, idx["plan_atr_pct"]]),
                }
            )
        return out


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")
