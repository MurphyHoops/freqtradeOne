from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config.v29_config import V29Config
from .reservation import ReservationAgent
from .tier import TierManager


@dataclass
class InvariantViolation:
    """InvariantViolation 的职责说明。"""
    code: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """处理 to_dict 的主要逻辑。"""
        return {"code": self.code, "details": self.details}


@dataclass
class InvariantReport:
    """InvariantReport 的职责说明。"""
    ok: bool
    violations: List[InvariantViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """处理 to_dict 的主要逻辑。"""
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
        }


class RiskAgent:
    """RiskAgent 的职责说明。"""
    def __init__(self, cfg: V29Config, reservation: ReservationAgent, tier_mgr: TierManager) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.cfg = cfg
        self.reservation = reservation
        self.tier_mgr = tier_mgr
        self._released_seen: set[str] = set()
        self._ttl_snapshot: Dict[str, int] = {}

    def check_invariants(self, state, equity: float, cap_pct: float) -> Dict[str, Any]:
        """处理 check_invariants 的主要逻辑。"""
        violations: List[InvariantViolation] = []

        def add(code: str, **details: Any) -> None:
            """处理 add 的主要逻辑。"""
            violations.append(InvariantViolation(code=code, details=details))

        cap_abs = cap_pct * equity
        total_open = state.get_total_open_risk()
        total_reserved = self.reservation.get_total_reserved()
        if total_reserved < -1e-6:
            add("NEGATIVE_RESERVED_PORTFOLIO", value=total_reserved)
        if total_open < -1e-6:
            add("NEGATIVE_OPEN_PORTFOLIO", value=total_open)
        if total_open + total_reserved > cap_abs + 1e-6:
            add(
                "PORTFOLIO_CAP_EXCEEDED",
                total_open=total_open,
                reserved=total_reserved,
                cap_abs=cap_abs,
            )

        pairs = set(state.per_pair.keys()) | set(self.reservation.reserved_pair_risk.keys())
        for pair in pairs:
            pst = state.get_pair_state(pair)
            tier_pol = self.tier_mgr.get(pst.closs)
            cap = tier_pol.per_pair_risk_cap_pct * equity
            open_risk = state.pair_risk_open.get(pair, 0.0)
            reserved = self.reservation.get_pair_reserved(pair)
            if reserved < -1e-6:
                add("NEGATIVE_RESERVED_PAIR", pair=pair, value=reserved)
            if open_risk < -1e-6:
                add("NEGATIVE_OPEN_PAIR", pair=pair, value=open_risk)
            if open_risk + reserved > cap + 1e-6:
                add(
                    "PAIR_CAP_EXCEEDED",
                    pair=pair,
                    open_risk=open_risk,
                    reserved=reserved,
                    cap_abs=cap,
                )

        ttl_snapshot_next: Dict[str, int] = {}
        for rid, rec in self.reservation.reservations.items():
            ttl = int(rec.ttl_bars)
            if ttl < 0:
                add("RESERVATION_TTL_NEGATIVE", reservation_id=rid, ttl=ttl)
            prev = self._ttl_snapshot.get(rid)
            if prev is not None and ttl > prev:
                add("RESERVATION_TTL_NOT_DECREASING", reservation_id=rid, previous=prev, current=ttl)
            ttl_snapshot_next[rid] = ttl
        self._ttl_snapshot = ttl_snapshot_next

        for rid in self.reservation.drain_recent_releases():
            if rid in self._released_seen:
                add("RESERVATION_DUPLICATE_RELEASE", reservation_id=rid)
            self._released_seen.add(rid)

        report = InvariantReport(ok=not violations, violations=violations)
        return report.to_dict()
