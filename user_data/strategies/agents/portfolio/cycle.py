"""TaxBrainV30 cycle coordinator.

CycleAgent drives:
1. advance bar_tick, debt decay, and cooldown counters,
2. build allocation plan via TreasuryAgent,
3. apply profit repayment (V30 revision #2),
4. run invariant checks, logging, and persistence.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from ...config.v30_config import V30Config
from .analytics import AnalyticsAgent
from .reservation import ReservationAgent
from .risk import RiskAgent
from .treasury import AllocationPlan, TreasuryAgent
from .global_backend import GlobalRiskBackend


class CycleAgent:
    """Coordinate bar-level cadence and treasury sync."""

    def __init__(
        self,
        cfg: V30Config,
        state,
        reservation: ReservationAgent,
        treasury: TreasuryAgent,
        risk: RiskAgent,
        analytics: AnalyticsAgent,
        persist,
        tier_mgr,
        backend: Optional[GlobalRiskBackend] = None,
        engine: Any | None = None,
    ) -> None:
        """Construct cycle agent and inject dependencies.

        Args:
            cfg: Runtime config (cycle length, decay parameters).
            state: GlobalState instance for portfolio risk state.
            reservation: Reservation agent for TTL and reservation tracking.
            treasury: Treasury agent to build allocation plan.
            risk: Risk agent for invariants.
            analytics: Analytics agent for finalize/invariant events.
            persist: StateStore wrapper for persistence.
            tier_mgr: TierManager for tier and cooldown rules.
        """

        self.cfg = cfg
        self.state = state
        self.reservation = reservation
        self.treasury = treasury
        self.risk = risk
        self.analytics = analytics
        self.persist = persist
        self.tier_mgr = tier_mgr
        self.backend = backend
        self.engine = engine

    def finalize(self, eq_provider) -> AllocationPlan:
        """Delegate finalize to Engine."""

        if not self.engine:
            raise RuntimeError("CycleAgent.engine is not attached")
        return self.engine.finalize_bar()

    def maybe_finalize(
        self,
        pair: str,
        bar_ts: float,
        whitelist: Iterable[str],
        timeframe_sec: int,
        eq_provider,
    ) -> None:
        """Decide whether to run finalize based on progress and timeout.

        Args:
            pair: Pair that just finished populate_indicators.
            bar_ts: Bar timestamp (seconds).
            whitelist: Current whitelist to confirm all reported.
            timeframe_sec: Timeframe in seconds (for timeout threshold).
            eq_provider: EquityProvider for finalize when triggered.
        """

        now = time.time()
        if self.state.current_cycle_ts is None or bar_ts > float(self.state.current_cycle_ts):
            self.state.current_cycle_ts = float(bar_ts)
            self.state.reported_pairs_for_current_cycle = set()
        self.state.reported_pairs_for_current_cycle.add(pair)

        all_reported = all(
            p in self.state.reported_pairs_for_current_cycle for p in whitelist
        )
        newer_than_last = (
            self.state.last_finalized_bar_ts is None
            or float(self.state.current_cycle_ts) > float(self.state.last_finalized_bar_ts)
        )
        timeout = (now - self.state.last_finalize_walltime) >= (
            self.cfg.force_finalize_mult * timeframe_sec
        )

        if newer_than_last and (all_reported or timeout):
            self.finalize(eq_provider)
            self.state.last_finalize_walltime = now
            self.state.last_finalized_bar_ts = float(self.state.current_cycle_ts)
            self.state.reported_pairs_for_current_cycle = set()
