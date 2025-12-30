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
        """Finalize the cycle, delegating to Engine when available."""

        if self.engine:
            return self.engine.finalize_bar()
        return self._finalize_without_engine(eq_provider)

    def _decay_and_cooldowns(self, bars_passed: int) -> None:
        if bars_passed <= 0:
            return
        for pst in getattr(self.state, "per_pair", {}).values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left = max(0, pst.cooldown_bars_left - bars_passed)
            for meta in getattr(pst, "active_trades", {}).values():
                if meta.icu_bars_left is not None and meta.icu_bars_left > 0:
                    meta.icu_bars_left = max(0, meta.icu_bars_left - bars_passed)
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        self.state.debt_pool *= (float(getattr(risk_cfg, "pain_decay_per_bar", 1.0)) ** bars_passed)
        for _ in range(bars_passed):
            self.reservation.tick_ttl()

    def _build_snapshot(self) -> dict:
        pairs_payload = {}
        for pair, pst in getattr(self.state, "per_pair", {}).items():
            pairs_payload[pair] = {
                "cooldown_bars_left": pst.cooldown_bars_left,
                "active_trades": len(pst.active_trades),
                "last_score": pst.last_score,
                "last_kind": getattr(pst, "last_kind", None),
                "last_dir": pst.last_dir,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "local_loss": pst.local_loss,
                "closs": pst.closs,
                "pair_open_risk": getattr(self.state, "pair_risk_open", {}).get(pair, 0.0),
                "pair_reserved_risk": self.reservation.get_pair_reserved(pair),
            }
        return {
            "debt_pool": self.state.debt_pool,
            "total_open_risk": self.state.get_total_open_risk(),
            "reserved_portfolio_risk": self.reservation.get_total_reserved(),
            "pairs": pairs_payload,
        }

    def _finalize_without_engine(self, eq_provider) -> AllocationPlan:
        self.state.bar_tick += 1
        self._decay_and_cooldowns(1)

        equity = eq_provider.get_equity()
        snapshot = self._build_snapshot()
        plan = self.treasury.plan(snapshot, equity)

        if plan is not None:
            self.state.treasury.k_long = getattr(plan, "k_long", 0.0)
            self.state.treasury.k_short = getattr(plan, "k_short", 0.0)
            self.state.treasury.theta = getattr(plan, "theta", 0.0)
            self.state.treasury.final_r = getattr(plan, "final_r", 0.0)
            self.state.treasury.available = getattr(plan, "available", 0.0)
            self.state.treasury.bias = getattr(plan, "bias", 0.0)
            self.state.treasury.volatility = getattr(plan, "volatility", 1.0)

        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        pnl_since_cycle_start = equity - float(self.state.treasury.cycle_start_equity)
        cycle_completed = (
            self.state.bar_tick - self.state.treasury.cycle_start_tick
        ) >= int(self.cfg.cycle_len_bars)
        cycle_cleared = False
        if cycle_completed:
            if pnl_since_cycle_start >= 0 and bool(
                getattr(
                    getattr(self.cfg, "risk", None),
                    "clear_debt_on_profitable_cycle",
                    getattr(self.cfg, "clear_debt_on_profitable_cycle", False),
                )
            ):
                cycle_cleared = True
                self.state.debt_pool = 0.0
                for pst in getattr(self.state, "per_pair", {}).values():
                    pst.local_loss = 0.0
                    pst.closs = 0
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        cap_abs = cap_pct * equity
        used_risk = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        cap_used_pct = (used_risk / cap_abs) if cap_abs > 0 else 0.0
        reservations_count = len(getattr(self.reservation, "reservations", {}))

        tier_summary = {}
        for pair, pst in getattr(self.state, "per_pair", {}).items():
            try:
                tier_pol = self.tier_mgr.get(pst.closs) if self.tier_mgr else None
                tier_name = getattr(tier_pol, "name", None) if tier_pol else None
            except Exception:
                tier_name = None
            recipes = sorted(
                {meta.recipe for meta in pst.active_trades.values() if getattr(meta, "recipe", None)}
            )
            profiles = sorted(
                {meta.exit_profile for meta in pst.active_trades.values() if getattr(meta, "exit_profile", None)}
            )
            tier_summary[pair] = {
                "tier": tier_name,
                "active_trades": len(pst.active_trades),
                "active_recipes": recipes,
                "active_exit_profiles": profiles,
            }

        self.analytics.log_finalize(
            bar_tick=self.state.bar_tick,
            pnl=pnl_since_cycle_start,
            debt_pool=self.state.debt_pool,
            k_long=getattr(plan, "k_long", 0.0),
            k_short=getattr(plan, "k_short", 0.0),
            theta=getattr(plan, "theta", 0.0),
            final_r=getattr(plan, "final_r", 0.0),
            cap_used_pct=cap_used_pct,
            reservations=reservations_count,
            cycle_cleared=cycle_cleared,
            tier_summary=tier_summary,
        )

        report = self.risk.check_invariants(self.state, equity, cap_pct)
        report_payload = report.to_dict() if hasattr(report, "to_dict") else {"ok": True}
        self.analytics.log_invariant(report_payload)
        if not report_payload.get("ok", True):
            print("[WARN] Risk invariant breach:", report_payload)

        self.persist.save()
        return plan

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
