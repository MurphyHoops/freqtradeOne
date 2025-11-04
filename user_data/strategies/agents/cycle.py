from __future__ import annotations

import time
from typing import Iterable

from ..config.v29_config import V29Config
from .analytics import AnalyticsAgent
from .reservation import ReservationAgent
from .risk import RiskAgent
from .treasury import AllocationPlan, TreasuryAgent


class CycleAgent:
    """CycleAgent 的职责说明。"""
    def __init__(
        self,
        cfg: V29Config,
        state,
        reservation: ReservationAgent,
        treasury: TreasuryAgent,
        risk: RiskAgent,
        analytics: AnalyticsAgent,
        persist,
        tier_mgr,
    ) -> None:
        """处理 __init__ 的主要逻辑。"""
        self.cfg = cfg
        self.state = state
        self.reservation = reservation
        self.treasury = treasury
        self.risk = risk
        self.analytics = analytics
        self.persist = persist
        self.tier_mgr = tier_mgr

    def finalize(self, eq_provider) -> AllocationPlan:
        """处理 finalize 的主要逻辑。"""
        self.state.bar_tick += 1
        self._decay_and_cooldowns()

        equity = eq_provider.get_equity()
        snapshot = self._build_snapshot()
        plan = self.treasury.plan(snapshot, equity)
        self.state.treasury.fast_alloc_risk = plan.fast_alloc_risk
        self.state.treasury.slow_alloc_risk = plan.slow_alloc_risk

        if self.state.treasury.cycle_start_tick == 0:
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        pnl_since_cycle_start = equity - float(self.state.treasury.cycle_start_equity)
        cycle_completed = (
            self.state.bar_tick - self.state.treasury.cycle_start_tick
        ) >= int(self.cfg.cycle_len_bars)
        cycle_cleared = False
        if cycle_completed:
            if pnl_since_cycle_start >= 0 and bool(self.cfg.clear_debt_on_profitable_cycle):
                cycle_cleared = True
                self.state.debt_pool = 0.0
                for pst in self.state.per_pair.values():
                    pst.local_loss = 0.0
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        fast_alloc_size = sum(plan.fast_alloc_risk.values())
        slow_alloc_size = sum(plan.slow_alloc_risk.values())
        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        cap_abs = cap_pct * equity
        used_risk = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        cap_used_pct = (used_risk / cap_abs) if cap_abs > 0 else 0.0
        reservations_count = len(self.reservation.reservations)

        self.analytics.log_finalize(
            bar_tick=self.state.bar_tick,
            pnl=pnl_since_cycle_start,
            debt_pool=self.state.debt_pool,
            fast_alloc_size=fast_alloc_size,
            slow_alloc_size=slow_alloc_size,
            cap_used_pct=cap_used_pct,
            reservations=reservations_count,
            cycle_cleared=cycle_cleared,
        )

        report = self.risk.check_invariants(self.state, equity, cap_pct)
        if isinstance(report, dict):
            report_payload = report
        else:
            report_payload = getattr(report, "__dict__", {"ok": True})
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
        """处理 maybe_finalize 的主要逻辑。"""
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

    def _decay_and_cooldowns(self) -> None:
        """处理 _decay_and_cooldowns 的主要逻辑。"""
        for pst in self.state.per_pair.values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left -= 1
            for meta in pst.active_trades.values():
                if meta.icu_bars_left is not None and meta.icu_bars_left > 0:
                    meta.icu_bars_left -= 1
        self.state.debt_pool *= self.cfg.pain_decay_per_bar
        self.reservation.tick_ttl()

    def _build_snapshot(self) -> dict:
        """处理 _build_snapshot 的主要逻辑。"""
        pairs_payload = {}
        for pair, pst in self.state.per_pair.items():
            pairs_payload[pair] = {
                "cooldown_bars_left": pst.cooldown_bars_left,
                "active_trades": len(pst.active_trades),
                "last_score": pst.last_score,
                "last_dir": pst.last_dir,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "local_loss": pst.local_loss,
                "closs": pst.closs,
                "pair_open_risk": self.state.pair_risk_open.get(pair, 0.0),
                "pair_reserved_risk": self.reservation.get_pair_reserved(pair),
            }
        return {
            "debt_pool": self.state.debt_pool,
            "total_open_risk": self.state.get_total_open_risk(),
            "reserved_portfolio_risk": self.reservation.get_total_reserved(),
            "pairs": pairs_payload,
        }
