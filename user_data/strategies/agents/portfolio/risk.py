# -*- coding: utf-8 -*-
"""Risk invariant checks for the portfolio."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ...config.v30_config import V30Config
from .reservation import ReservationAgent
from .tier import TierManager
from .global_backend import GlobalRiskBackend


@dataclass
class InvariantViolation:
    """Single invariant violation entry."""

    code: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""

        return {"code": self.code, "details": self.details}


@dataclass
class InvariantReport:
    """Risk invariant report."""

    ok: bool
    violations: List[InvariantViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dict."""

        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
        }


class RiskAgent:
    """Check portfolio caps and reservation consistency."""

    def __init__(
        self,
        cfg: V30Config,
        reservation: ReservationAgent,
        tier_mgr: TierManager,
        backend: Optional[GlobalRiskBackend] = None,
    ) -> None:
        """Initialize risk agent.

        Args:
            cfg: V30Config for cap thresholds.
            reservation: ReservationAgent to read reservation usage.
            tier_mgr: TierManager providing per-pair caps.
        """

        self.cfg = cfg
        self.reservation = reservation
        self.tier_mgr = tier_mgr
        self.backend = backend
        self._released_seen: set[str] = set()
        self._ttl_snapshot: Dict[str, int] = {}

    def check_invariants(self, state, equity: float, cap_pct: float) -> InvariantReport:
        """Run invariant checks and return a structured report."""

        violations: List[InvariantViolation] = []

        def add(code: str, **details: Any) -> None:
            """Append a violation entry."""

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

        if self.backend:
            snap = self.backend.get_snapshot()
            backend_risk = getattr(snap, "risk_used", 0.0)
            local_risk = total_open + total_reserved
            tol = 0.01 * max(abs(local_risk), abs(backend_risk), 1e-6)
            if abs(backend_risk - local_risk) > tol:
                add(
                    "GLOBAL_STATE_MISMATCH",
                    backend_risk=backend_risk,
                    local_risk=local_risk,
                    tolerance=tol,
                )

        report = InvariantReport(ok=not violations, violations=violations)
        return report
