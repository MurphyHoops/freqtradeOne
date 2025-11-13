# -*- coding: utf-8 -*-
"""风险预约池管理模块。

ReservationAgent 负责在下单前锁定名义风险额度，并在成交/撤单/TTL 过期时
释放额度，确保财务拨款与实际下单保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

from ...config.v29_config import V29Config

if TYPE_CHECKING:
    from .analytics import AnalyticsAgent


@dataclass
class ReservationRecord:
    """描述单条预约的结构信息。"""

    pair: str
    risk: float
    bucket: str
    ttl_bars: int


class ReservationAgent:
    """管理预约、释放与 TTL 推进的风险代理。"""

    def __init__(self, cfg: V29Config, analytics: Optional["AnalyticsAgent"] = None) -> None:
        """初始化预约代理。

        Args:
            cfg: V29Config，用于读取预约 TTL 等参数。
            analytics: 可选的 AnalyticsAgent，用于记录预约事件日志。
        """

        self.cfg = cfg
        self.analytics = analytics
        self.reservations: Dict[str, ReservationRecord] = {}
        self.reserved_pair_risk: Dict[str, float] = {}
        self.reserved_bucket_risk: Dict[str, float] = {"fast": 0.0, "slow": 0.0}
        self.reserved_portfolio_risk: float = 0.0
        self._released_ids_since_cycle: set[str] = set()

    def reserve(self, pair: str, rid: str, risk: float, bucket: str) -> None:
        """为某交易对锁定风险额度。"""

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
        """释放预约风险，仅调整预约池不回滚财政（V29.1 修订 #5）。"""

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
        """推进预约的 TTL，自动释放过期预约。"""

        expired: list[str] = []
        for rid, rec in list(self.reservations.items()):
            rec.ttl_bars -= 1
            self.reservations[rid] = rec
            if rec.ttl_bars <= 0:
                expired.append(rid)
        for rid in expired:
            self.release(rid, event="expire")

    def get_total_reserved(self) -> float:
        """返回组合层面已预约的风险额度。"""

        return self.reserved_portfolio_risk

    def get_pair_reserved(self, pair: str) -> float:
        """返回指定交易对已被预约的风险。"""

        return self.reserved_pair_risk.get(pair, 0.0)

    def get_bucket_reserved(self, bucket: str) -> float:
        """返回指定拨款桶已被预约的风险。"""

        return self.reserved_bucket_risk.get(bucket, 0.0)

    def drain_recent_releases(self) -> Iterable[str]:
        """导出自上次调用以来已释放的预约 ID 并清空记录。"""

        ids = tuple(self._released_ids_since_cycle)
        self._released_ids_since_cycle.clear()
        return ids

    def to_snapshot(self) -> Dict[str, Any]:
        """构造可持久化的预约池快照。"""

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
        """从快照恢复预约池状态。"""

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
