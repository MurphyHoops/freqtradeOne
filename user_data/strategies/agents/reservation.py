from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from ..config.v29_config import V29Config

if TYPE_CHECKING:
    from .analytics import AnalyticsAgent


@dataclass
class ReservationRecord:
    """ReservationRecord 的职责说明。"""
    pair: str
    risk: float
    bucket: str
    ttl_bars: int


class ReservationAgent:
    """ReservationAgent 的职责说明。"""
    def __init__(self, cfg: V29Config, analytics: Optional["AnalyticsAgent"] = None) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.cfg = cfg
        self.analytics = analytics
        self.reservations: Dict[str, ReservationRecord] = {}
        self.reserved_pair_risk: Dict[str, float] = {}
        self.reserved_bucket_risk: Dict[str, float] = {"fast": 0.0, "slow": 0.0}
        self.reserved_portfolio_risk: float = 0.0
        self._released_ids_since_cycle: set[str] = set()

    def reserve(self, pair: str, rid: str, risk: float, bucket: str) -> None:
        """处理 reserve 的主要逻辑。"""
        if rid in self.reservations:
            return
        ttl = int(self.cfg.reservation_ttl_bars)
        record = ReservationRecord(pair=pair, risk=float(risk), bucket=bucket, ttl_bars=ttl)
        self.reservations[rid] = record
        self.reserved_portfolio_risk += record.risk
        self.reserved_pair_risk[pair] = self.reserved_pair_risk.get(pair, 0.0) + record.risk
        self.reserved_bucket_risk[bucket] = self.reserved_bucket_risk.get(bucket, 0.0) + record.risk
        if self.analytics:
            self.analytics.log_reservation("create", rid, pair, bucket, record.risk)

    def release(self, rid: str, event: str = "release") -> Tuple[str, float, str]:
        """处理 release 的主要逻辑。"""
        record = self.reservations.pop(rid, None)
        if not record:
            return ("", 0.0, "slow")
        self.reserved_portfolio_risk = max(0.0, self.reserved_portfolio_risk - record.risk)
        pair_total = max(0.0, self.reserved_pair_risk.get(record.pair, 0.0) - record.risk)
        self.reserved_pair_risk[record.pair] = pair_total
        bucket_total = max(0.0, self.reserved_bucket_risk.get(record.bucket, 0.0) - record.risk)
        self.reserved_bucket_risk[record.bucket] = bucket_total
        self._released_ids_since_cycle.add(rid)
        if self.analytics:
            self.analytics.log_reservation(event, rid, record.pair, record.bucket, record.risk)
        return (record.pair, record.risk, record.bucket)

    def tick_ttl(self) -> None:
        """处理 tick_ttl 的主要逻辑。"""
        expired: list[str] = []
        for rid, rec in list(self.reservations.items()):
            rec.ttl_bars -= 1
            self.reservations[rid] = rec
            if rec.ttl_bars <= 0:
                expired.append(rid)
        for rid in expired:
            # V29.1 #5: only release reservation, no fiscal rollback here.
            self.release(rid, event="expire")

    def get_total_reserved(self) -> float:
        """处理 get_total_reserved 的主要逻辑。"""
        return self.reserved_portfolio_risk

    def get_pair_reserved(self, pair: str) -> float:
        """处理 get_pair_reserved 的主要逻辑。"""
        return self.reserved_pair_risk.get(pair, 0.0)

    def get_bucket_reserved(self, bucket: str) -> float:
        """处理 get_bucket_reserved 的主要逻辑。"""
        return self.reserved_bucket_risk.get(bucket, 0.0)

    def drain_recent_releases(self) -> Iterable[str]:
        """处理 drain_recent_releases 的主要逻辑。"""
        ids = tuple(self._released_ids_since_cycle)
        self._released_ids_since_cycle.clear()
        return ids

    def to_snapshot(self) -> Dict[str, Any]:
        """处理 to_snapshot 的主要逻辑。"""
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
        """处理 restore_snapshot 的主要逻辑。"""
        snap = snap or {}
        self.reservations = {}
        for rid, payload in snap.get("reservations", {}).items():
            self.reservations[rid] = ReservationRecord(
                pair=str(payload.get("pair", "")),
                risk=float(payload.get("risk", 0.0)),
                bucket=str(payload.get("bucket", "slow")),
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
