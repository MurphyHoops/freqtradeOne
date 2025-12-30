# -*- coding: utf-8 -*-
"""Reservation pool management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from ...config.v30_config import V30Config
from .global_backend import GlobalRiskBackend

if TYPE_CHECKING:
    from .analytics import AnalyticsAgent


@dataclass
class ReservationRecord:
    """Snapshot for a single reservation entry."""

    pair: str
    risk: float
    bucket: str
    ttl_bars: int


class ReservationAgent:
    """Manage reservation, release, and TTL advancement."""

    def __init__(self, cfg: V30Config, analytics: Optional["AnalyticsAgent"] = None, backend: Optional[GlobalRiskBackend] = None) -> None:
        """Initialize reservation agent.

        Args:
            cfg: V30Config for TTL and reservation sizing.
            analytics: Optional AnalyticsAgent for logging reservation events.
        """

        self.cfg = cfg
        self.analytics = analytics
        self.backend = backend
        self.reservations: Dict[str, ReservationRecord] = {}
        self.reserved_pair_risk: Dict[str, float] = {}
        self.reserved_bucket_risk: Dict[str, float] = {"long": 0.0, "short": 0.0}
        self.reserved_portfolio_risk: float = 0.0
        self._released_ids_since_cycle: set[str] = set()

    def reserve(self, pair: str, rid: str, risk: float, bucket: str, record_backend: bool = True) -> None:
        """Reserve risk for a single pair."""

        if rid in self.reservations:
            return
        ttl = int(self.cfg.reservation_ttl_bars)
        record = ReservationRecord(pair=pair, risk=float(risk), bucket=bucket, ttl_bars=ttl)
        self.reservations[rid] = record
        self.reserved_portfolio_risk += record.risk
        self.reserved_pair_risk[pair] = self.reserved_pair_risk.get(pair, 0.0) + record.risk
        self.reserved_bucket_risk[bucket] = self.reserved_bucket_risk.get(bucket, 0.0) + record.risk
        if self.backend and record_backend:
            self.backend.add_risk_usage(record.risk)
        if self.analytics:
            self.analytics.log_reservation("create", rid, pair, bucket, record.risk)

    def release(self, rid: str, event: str = "release") -> Tuple[str, float, str]:
        """Release reserved risk without rolling back treasury (V30 revision #5)."""

        record = self.reservations.pop(rid, None)
        if not record:
            return ("", 0.0, "slow")
        self.reserved_portfolio_risk = max(0.0, self.reserved_portfolio_risk - record.risk)
        pair_total = max(0.0, self.reserved_pair_risk.get(record.pair, 0.0) - record.risk)
        self.reserved_pair_risk[record.pair] = pair_total
        bucket_total = max(0.0, self.reserved_bucket_risk.get(record.bucket, 0.0) - record.risk)
        self.reserved_bucket_risk[record.bucket] = bucket_total
        self._released_ids_since_cycle.add(rid)
        if self.backend:
            self.backend.release_risk_usage(record.risk)
        if self.analytics:
            self.analytics.log_reservation(event, rid, record.pair, record.bucket, record.risk)
        return (record.pair, record.risk, record.bucket)

    def tick_ttl(self) -> None:
        """Advance TTL and expire reservations."""

        expired: list[str] = []
        for rid, rec in list(self.reservations.items()):
            rec.ttl_bars -= 1
            self.reservations[rid] = rec
            if rec.ttl_bars <= 0:
                expired.append(rid)
        for rid in expired:
            self.release(rid, event="expire")

    def get_total_reserved(self) -> float:
        """Return total reserved risk for the portfolio."""

        return self.reserved_portfolio_risk

    def get_pair_reserved(self, pair: str) -> float:
        """Return reserved risk for a specific pair."""

        return self.reserved_pair_risk.get(pair, 0.0)

    def get_bucket_reserved(self, bucket: str) -> float:
        """Return reserved risk for a specific bucket."""

        return self.reserved_bucket_risk.get(bucket, 0.0)

    def drain_recent_releases(self) -> Iterable[str]:
        """Return release ids since last call and clear the buffer."""

        ids = tuple(self._released_ids_since_cycle)
        self._released_ids_since_cycle.clear()
        return ids

    def to_snapshot(self) -> Dict[str, Any]:
        """Serialize reservation pool state."""

        return {
            "reservations": {
                rid: {
                    "pair": rec.pair,
                    "risk": rec.risk,
                    "bucket": rec.bucket,
                    "ttl_bars": rec.ttl_bars,
                }
                for rid, rec in self.reservations.items()
            },
            "reserved_pair_risk": self.reserved_pair_risk,
            "reserved_bucket_risk": self.reserved_bucket_risk,
            "reserved_portfolio_risk": self.reserved_portfolio_risk,
        }

    def restore_snapshot(self, snap: Optional[dict]) -> None:
        """Restore reservation pool state from snapshot."""

        snap = snap or {}
        self.reservations = {}
        for rid, payload in snap.get("reservations", {}).items():
            self.reservations[rid] = ReservationRecord(
                pair=str(payload.get("pair", "")),
                risk=float(payload.get("risk", 0.0)),
                bucket=str(payload.get("bucket", "long")),
                ttl_bars=int(payload.get("ttl_bars", self.cfg.reservation_ttl_bars)),
            )
        self.reserved_pair_risk = {
            k: float(v) for k, v in snap.get("reserved_pair_risk", {}).items()
        }
        self.reserved_bucket_risk = {
            k: float(v) for k, v in snap.get("reserved_bucket_risk", {}).items()
        }
        self.reserved_portfolio_risk = float(snap.get("reserved_portfolio_risk", 0.0))
        self._released_ids_since_cycle.clear()
