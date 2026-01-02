from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Iterable
import time
import uuid

from ..config.v30_config import V30Config
from ..agents.portfolio.global_backend import GlobalRiskBackend
from ..agents.portfolio.schemas import ActiveTradeMeta, PairState
from .rejections import RejectReason


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    thresholds: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    debt: Optional[float] = None
    closs: Optional[int] = None

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any], default_closs: Optional[int]) -> "GateResult":
        thresholds = payload.get("thresholds") if isinstance(payload, dict) else {}
        return cls(
            allowed=bool(payload.get("allowed", True)) if isinstance(payload, dict) else True,
            thresholds=thresholds if isinstance(thresholds, dict) else {},
            reason=str(payload.get("reason", "")) if isinstance(payload, dict) else "",
            debt=payload.get("debt") if isinstance(payload, dict) else None,
            closs=payload.get("closs", default_closs) if isinstance(payload, dict) else default_closs,
        )


@dataclass
class TreasuryState:
    k_long: float = 0.0
    k_short: float = 0.0
    theta: float = 0.0
    final_r: float = 0.0
    available: float = 0.0
    bias: float = 0.0
    volatility: float = 1.0
    cycle_start_tick: int = 0
    cycle_start_equity: float = 0.0

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "k_long": self.k_long,
            "k_short": self.k_short,
            "theta": self.theta,
            "final_r": self.final_r,
            "available": self.available,
            "bias": self.bias,
            "volatility": self.volatility,
            "cycle_start_tick": self.cycle_start_tick,
            "cycle_start_equity": self.cycle_start_equity,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        self.k_long = float(payload.get("k_long", 0.0))
        self.k_short = float(payload.get("k_short", 0.0))
        self.theta = float(payload.get("theta", 0.0))
        self.final_r = float(payload.get("final_r", 0.0))
        self.available = float(payload.get("available", 0.0))
        self.bias = float(payload.get("bias", 0.0))
        self.volatility = float(payload.get("volatility", 1.0))
        self.cycle_start_tick = int(payload.get("cycle_start_tick", 0))
        self.cycle_start_equity = float(payload.get("cycle_start_equity", 0.0))


class GlobalState:
    def __init__(self, cfg: V30Config, backend: Optional[GlobalRiskBackend] = None) -> None:
        self.cfg = cfg
        self.backend = backend
        self.per_pair: Dict[str, PairState] = {}
        self.debt_pool: float = 0.0
        self.trade_risk_ledger: Dict[str, float] = {}
        self.pair_risk_open: Dict[str, float] = {}
        self.treasury = TreasuryState()
        self.bar_tick: int = 0
        self.current_cycle_ts: Optional[float] = None
        self.last_finalized_bar_ts: Optional[float] = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime: float = time.time()

        self.trade_stake_ledger: Dict[str, float] = {}
        self.pair_stake_open: Dict[str, float] = {}

    def _canonical_pair(self, pair: str | None) -> str:
        if not pair:
            return ""
        return str(pair).split(":")[0]

    def get_pair_state(self, pair: str) -> PairState:
        key = self._canonical_pair(pair)
        if key not in self.per_pair:
            self.per_pair[key] = PairState()
        return self.per_pair[key]

    def get_loss_tier_state(self, pair: str) -> int:
        return self.get_pair_state(pair).closs

    def record_signal(self, pair: str, candidate) -> None:
        pst = self.get_pair_state(pair)
        if candidate:
            pst.last_dir = candidate.direction
            pst.last_squad = candidate.squad
            pst.last_score = float(candidate.expected_edge)
            pst.last_sl_pct = float(candidate.sl_pct)
            pst.last_tp_pct = float(candidate.tp_pct)
            pst.last_kind = str(candidate.kind)
            pst.last_exit_profile = candidate.exit_profile
            pst.last_recipe = candidate.recipe
            pst.last_atr_pct = float(candidate.plan_atr_pct) if candidate.plan_atr_pct is not None else 0.0
        else:
            pst.last_dir = None
            pst.last_squad = None
            pst.last_score = 0.0
            pst.last_sl_pct = 0.0
            pst.last_tp_pct = 0.0
            pst.last_exit_profile = None
            pst.last_recipe = None
            pst.last_atr_pct = 0.0

    def get_total_open_risk(self) -> float:
        return sum(self.pair_risk_open.values())

    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        base = risk_cfg.portfolio_cap_pct_base
        if equity <= 0:
            return base * 0.5
        if (self.debt_pool / equity) > risk_cfg.drawdown_threshold_pct:
            return base * 0.5
        return base

    def per_pair_cap_room(self, pair: str, equity: float, tier_pol, reserved: float) -> float:
        key = self._canonical_pair(pair)
        cap = tier_pol.per_pair_risk_cap_pct * equity
        used = self.pair_risk_open.get(key, 0.0) + reserved
        return max(0.0, cap - used)

    def record_trade_open(
        self,
        pair: str,
        trade_id: str,
        real_risk: float,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        bucket: str,
        entry_price: float,
        tier_pol,
        exit_profile: Optional[str] = None,
        recipe: Optional[str] = None,
        plan_timeframe: Optional[str] = None,
        plan_atr_pct: Optional[float] = None,
        tier_name: Optional[str] = None,
        stake_nominal: Optional[float] = None,
    ) -> None:
        pst = self.get_pair_state(pair)
        pair_key = self._canonical_pair(pair)
        tier_name = tier_name or (getattr(tier_pol, "name", None) if tier_pol else None)
        self.trade_risk_ledger[trade_id] = float(real_risk)
        self.pair_risk_open[pair_key] = self.pair_risk_open.get(pair_key, 0.0) + float(real_risk)

        if stake_nominal is None:
            if sl_pct and sl_pct > 0:
                stake_nominal = real_risk / sl_pct
            else:
                stake_nominal = 0.0

        stake_nominal = float(max(stake_nominal or 0.0, 0.0))
        self.trade_stake_ledger[trade_id] = stake_nominal
        self.pair_stake_open[pair_key] = self.pair_stake_open.get(pair_key, 0.0) + stake_nominal

        icu_left = tier_pol.icu_force_exit_bars if tier_pol.icu_force_exit_bars > 0 else None
        pst.active_trades[trade_id] = ActiveTradeMeta(
            sl_pct=float(sl_pct),
            tp_pct=float(tp_pct),
            direction=str(direction),
            entry_bar_tick=self.bar_tick,
            entry_price=float(entry_price),
            bucket=str(bucket),
            icu_bars_left=icu_left,
            exit_profile=exit_profile,
            recipe=recipe,
            plan_timeframe=plan_timeframe,
            plan_atr_pct=float(plan_atr_pct) if plan_atr_pct is not None else None,
            tier_name=tier_name,
        )
        pst.cooldown_bars_left = max(pst.cooldown_bars_left, tier_pol.cooldown_bars)

    def record_trade_close(self, pair: str, trade_id: str, profit_abs: float, tier_mgr) -> None:
        pst = self.get_pair_state(pair)
        pair_key = self._canonical_pair(pair)

        was_risk = self.trade_risk_ledger.pop(trade_id, 0.0)
        self.pair_risk_open[pair_key] = max(0.0, self.pair_risk_open.get(pair_key, 0.0) - was_risk)
        if self.pair_risk_open.get(pair_key, 0.0) <= 1e-12:
            self.pair_risk_open[pair_key] = 0.0

        was_stake = self.trade_stake_ledger.pop(trade_id, 0.0)
        self.pair_stake_open[pair_key] = max(0.0, self.pair_stake_open.get(pair_key, 0.0) - was_stake)
        if self.pair_stake_open.get(pair_key, 0.0) <= 1e-12:
            self.pair_stake_open[pair_key] = 0.0

        pst.active_trades.pop(trade_id, None)
        prev_closs = pst.closs

        routing_map = getattr(tier_mgr, "_routing_map", None) or {}
        max_closs = max(routing_map.keys()) if routing_map else 3

        if profit_abs >= 0:
            excess_profit = pst.local_loss - profit_abs
            pst.local_loss = 0
            self.debt_pool = max(0, self.debt_pool + excess_profit)

            if self.backend:
                self.backend.atomic_update_debt(excess_profit)

            pst.closs = 0
            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars_after_win)
            return

        loss = abs(profit_abs)
        if loss > 0:
            self.debt_pool += loss
            if self.backend:
                self.backend.atomic_update_debt(loss)

        if prev_closs >= max_closs:
            pst.local_loss = 0.0
            pst.closs = 0

            pol = tier_mgr.get(pst.closs)
            pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars)
            return

        pst.local_loss += loss

        pst.closs = prev_closs + 1
        pol = tier_mgr.get(pst.closs)
        pst.cooldown_bars_left = max(pst.cooldown_bars_left, pol.cooldown_bars)

    def to_snapshot(self) -> Dict[str, Any]:
        per_pair_snap: Dict[str, Any] = {}
        for pair, pst in self.per_pair.items():
            per_pair_snap[pair] = {
                "closs": pst.closs,
                "local_loss": pst.local_loss,
                "cooldown_bars_left": pst.cooldown_bars_left,
                "last_dir": pst.last_dir,
                "last_score": pst.last_score,
                "last_kind": pst.last_kind,
                "last_squad": pst.last_squad,
                "last_sl_pct": pst.last_sl_pct,
                "last_tp_pct": pst.last_tp_pct,
                "last_atr_pct": pst.last_atr_pct,
                "last_exit_profile": pst.last_exit_profile,
                "last_recipe": pst.last_recipe,
                "active_trades": {
                    tid: {
                        "sl_pct": meta.sl_pct,
                        "tp_pct": meta.tp_pct,
                        "direction": meta.direction,
                        "entry_bar_tick": meta.entry_bar_tick,
                        "entry_price": meta.entry_price,
                        "bucket": meta.bucket,
                        "icu_bars_left": meta.icu_bars_left,
                        "exit_profile": meta.exit_profile,
                        "recipe": meta.recipe,
                    }
                    for tid, meta in pst.active_trades.items()
                },
            }
        return {
            "debt_pool": self.debt_pool,
            "per_pair": per_pair_snap,
            "trade_risk_ledger": self.trade_risk_ledger,
            "pair_risk_open": self.pair_risk_open,
            "treasury": self.treasury.to_snapshot(),
            "bar_tick": self.bar_tick,
            "current_cycle_ts": self.current_cycle_ts,
            "last_finalized_bar_ts": self.last_finalized_bar_ts,
            "last_finalize_walltime": self.last_finalize_walltime,
        }

    def restore_snapshot(self, payload: Dict[str, Any]) -> None:
        self.debt_pool = float(payload.get("debt_pool", 0.0))
        self.trade_risk_ledger = {k: float(v) for k, v in payload.get("trade_risk_ledger", {}).items()}
        self.pair_risk_open = {k: float(v) for k, v in payload.get("pair_risk_open", {}).items()}
        self.bar_tick = int(payload.get("bar_tick", 0))
        self.current_cycle_ts = payload.get("current_cycle_ts")
        self.last_finalized_bar_ts = payload.get("last_finalized_bar_ts")
        self.last_finalize_walltime = float(payload.get("last_finalize_walltime", time.time()))
        self.per_pair = {}
        for pair, pst_payload in payload.get("per_pair", {}).items():
            pst = PairState(
                closs=int(pst_payload.get("closs", 0)),
                local_loss=float(pst_payload.get("local_loss", 0.0)),
                cooldown_bars_left=int(pst_payload.get("cooldown_bars_left", 0)),
                last_dir=pst_payload.get("last_dir"),
                last_score=float(pst_payload.get("last_score", 0.0)),
                last_kind=pst_payload.get("last_kind"),
                last_squad=pst_payload.get("last_squad"),
                last_sl_pct=float(pst_payload.get("last_sl_pct", 0.0)),
                last_tp_pct=float(pst_payload.get("last_tp_pct", 0.0)),
                last_atr_pct=float(pst_payload.get("last_atr_pct", 0.0)),
                last_exit_profile=pst_payload.get("last_exit_profile"),
                last_recipe=pst_payload.get("last_recipe"),
            )
            for tid, meta_payload in pst_payload.get("active_trades", {}).items():
                pst.active_trades[tid] = ActiveTradeMeta(
                    sl_pct=float(meta_payload.get("sl_pct", 0.0)),
                    tp_pct=float(meta_payload.get("tp_pct", 0.0)),
                    direction=str(meta_payload.get("direction", "")),
                    entry_bar_tick=int(meta_payload.get("entry_bar_tick", 0)),
                    entry_price=float(meta_payload.get("entry_price", 0.0)),
                    bucket=str(meta_payload.get("bucket", "long")),
                    icu_bars_left=(
                        int(meta_payload["icu_bars_left"])
                        if meta_payload.get("icu_bars_left") is not None
                        else None
                    ),
                    exit_profile=meta_payload.get("exit_profile"),
                    recipe=meta_payload.get("recipe"),
                )
            self.per_pair[pair] = pst
        self.treasury.restore_snapshot(payload.get("treasury", {}))

    def reset_cycle_after_restore(self) -> None:
        self.current_cycle_ts = None
        self.reported_pairs_for_current_cycle = set()
        self.last_finalize_walltime = time.time()


class Engine:
    def __init__(
        self,
        cfg: V30Config,
        state: GlobalState,
        eq_provider: Any,
        treasury_agent: Any,
        reservation: Any,
        risk_agent: Any,
        analytics: Any,
        persist: Any,
        tier_mgr: Any,
        tf_sec: int,
        is_backtest_like: Callable[[], bool],
        guard_bus: Optional[Iterable[Callable[[str, Dict[str, Any]], bool]]] = None,
        rejections: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.state = state
        self.eq_provider = eq_provider
        self.treasury_agent = treasury_agent
        self.reservation = reservation
        self.risk_agent = risk_agent
        self.analytics = analytics
        self.persist = persist
        self.tier_mgr = tier_mgr
        self._tf_sec = int(tf_sec)
        self._is_backtest_like = is_backtest_like
        self._guards = list(guard_bus or [])
        self.rejections = rejections
        self.pending_entry_meta: Dict[str, Dict[str, Any]] = {}

    def set_guard_bus(self, guard_bus: Iterable[Callable[[str, Dict[str, Any]], bool]]) -> None:
        self._guards = list(guard_bus or [])

    def _record_reject(self, reason: str, pair: str, context: Optional[Dict[str, Any]]) -> None:
        if self.rejections is None:
            return
        try:
            self.rejections.record(reason, pair=pair, context=context or {})
        except Exception:
            return

    def evaluate_gatekeeping(self, pair: str, score: float, closs: int) -> GateResult:
        try:
            raw = self.treasury_agent.evaluate_signal_quality(pair, score, closs=closs)
        except Exception:
            raw = {}
        return GateResult.from_mapping(raw, default_closs=closs) if raw else GateResult(
            allowed=True, thresholds={}, reason="", debt=None, closs=closs
        )

    def reserve_backend_risk(self, pair: str, risk: float) -> bool:
        backend = getattr(self.state, "backend", None)
        if backend is None:
            return True
        cap_abs = 0.0
        try:
            equity_now = self.eq_provider.get_equity()
            cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity_now)
            cap_abs = cap_pct * equity_now
            reserved = bool(backend.add_risk_usage(risk, cap_abs))
        except Exception as exc:
            reserved = False
            try:
                print(
                    f"[backend] reserve failed for {pair}: risk={risk:.4f} cap_abs={cap_abs:.4f} err={exc}"
                )
            except Exception:
                pass

        if not reserved:
            msg = f"Global Gatekeeper: CAP reached for {pair}, risk={risk:.4f}, cap_abs={cap_abs:.4f}"
            try:
                print(msg)
            except Exception:
                pass
            return False
        return True

    def reserve_risk_resources(
        self,
        pair: str,
        stake: float,
        risk: float,
        bucket: str,
        sl: float,
        tp: float,
        direction: str,
        current_rate: float,
        meta: Dict[str, Any],
    ) -> bool:
        meta_payload = dict(meta or {})
        meta_payload.update(
            {
                "sl_pct": sl,
                "tp_pct": tp,
                "stake_final": stake,
                "risk_final": risk,
                "bucket": bucket,
                "entry_price": current_rate,
                "dir": direction,
            }
        )

        if self._is_backtest_like():
            self.pending_entry_meta[pair] = meta_payload
            return True

        backend = getattr(self.state, "backend", None)
        if backend:
            backend_reserved = self.reserve_backend_risk(pair, risk)
            if not backend_reserved:
                return False

        rid = f"{pair}:{bucket}:{uuid.uuid4().hex}"
        self.reservation.reserve(pair, rid, risk, bucket)
        meta_payload["reservation_id"] = rid
        self.pending_entry_meta[pair] = meta_payload
        return True

    def is_permitted(self, pair: str, context: Optional[Dict[str, Any]] = None) -> bool:
        pst = self.state.get_pair_state(pair)
        tier_pol = self.tier_mgr.get(pst.closs) if self.tier_mgr else None
        if tier_pol and context:
            kind = context.get("kind")
            squad = context.get("squad")
            recipe = context.get("recipe")
            if any(item is not None for item in (kind, squad, recipe)):
                if not tier_pol.permits(kind=kind, squad=squad, recipe=recipe):
                    self._record_reject(RejectReason.TIER_REJECT, pair, context)
                    return False
        if tier_pol and getattr(tier_pol, "single_position_only", False) and pst.active_trades:
            self._record_reject(RejectReason.SINGLE_POSITION, pair, context)
            return False
        if pst.cooldown_bars_left > 0:
            self._record_reject(RejectReason.COOLDOWN, pair, context)
            return False
        try:
            equity = float(self.eq_provider.get_equity())
        except Exception:
            equity = 0.0
        treasury_cfg = getattr(getattr(self.cfg, "trading", None), "treasury", getattr(self.cfg, "treasury", None))
        debt_cap_pct = float(getattr(treasury_cfg, "debt_pool_cap_pct", 0.0) or 0.0) if treasury_cfg else 0.0
        if debt_cap_pct > 0 and equity > 0:
            debt_cap_abs = debt_cap_pct * equity
            if float(getattr(self.state, "debt_pool", 0.0)) >= debt_cap_abs:
                self._record_reject(RejectReason.DEBT_CAP, pair, context)
                return False
        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        cap_abs = cap_pct * equity
        used = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        if cap_abs > 0 and used >= cap_abs:
            self._record_reject(RejectReason.PORTFOLIO_CAP, pair, context)
            return False
        if tier_pol:
            pair_reserved = self.reservation.get_pair_reserved(pair)
            pair_room = self.state.per_pair_cap_room(pair, equity, tier_pol, pair_reserved)
            if pair_room <= 0:
                self._record_reject(RejectReason.PAIR_CAP, pair, context)
                return False
        for guard in self._guards:
            try:
                if not guard(pair, context or {}):
                    self._record_reject(RejectReason.GUARD, pair, context)
                    return False
            except Exception:
                self._record_reject(RejectReason.GUARD, pair, context)
                return False
        score = None
        if context:
            score = context.get("score")
        if score is not None:
            try:
                score_val = float(score)
            except Exception:
                score_val = 0.0
            gate = self.evaluate_gatekeeping(pair, score_val, pst.closs)
            if not gate.allowed:
                self._record_reject(
                    RejectReason.GATEKEEP,
                    pair,
                    dict(context or {}, reason=gate.reason),
                )
                return False
        return True

    def _decay_and_cooldowns(self, bars_passed: int) -> None:
        if bars_passed <= 0:
            return
        for pst in self.state.per_pair.values():
            if pst.cooldown_bars_left > 0:
                pst.cooldown_bars_left = max(0, pst.cooldown_bars_left - bars_passed)
            for meta in pst.active_trades.values():
                if meta.icu_bars_left is not None and meta.icu_bars_left > 0:
                    meta.icu_bars_left = max(0, meta.icu_bars_left - bars_passed)
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        self.state.debt_pool *= (float(getattr(risk_cfg, "pain_decay_per_bar", 1.0)) ** bars_passed)
        for _ in range(bars_passed):
            self.reservation.tick_ttl()

    def _build_snapshot(self) -> Dict[str, Any]:
        pairs_payload = {}
        for pair, pst in self.state.per_pair.items():
            pairs_payload[pair] = {
                "cooldown_bars_left": pst.cooldown_bars_left,
                "active_trades": len(pst.active_trades),
                "last_score": pst.last_score,
                "last_kind": getattr(pst, "last_kind", None),
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

    def finalize_bar(self) -> Any:
        self.state.bar_tick += 1
        self._decay_and_cooldowns(1)

        equity = self.eq_provider.get_equity()
        snapshot = self._build_snapshot()
        plan = self.treasury_agent.plan(snapshot, equity)
        self.state.treasury.k_long = plan.k_long
        self.state.treasury.k_short = plan.k_short
        self.state.treasury.theta = plan.theta
        self.state.treasury.final_r = plan.final_r
        self.state.treasury.available = plan.available
        self.state.treasury.bias = plan.bias
        self.state.treasury.volatility = plan.volatility

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
                for pst in self.state.per_pair.values():
                    pst.local_loss = 0.0
                    pst.closs = 0
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        cap_abs = cap_pct * equity
        used_risk = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        cap_used_pct = (used_risk / cap_abs) if cap_abs > 0 else 0.0
        reservations_count = len(self.reservation.reservations)

        tier_summary: Dict[str, Dict[str, Any]] = {}
        for pair, pst in self.state.per_pair.items():
            try:
                tier_pol = self.tier_mgr.get(pst.closs) if self.tier_mgr else None
                tier_name = getattr(tier_pol, "name", None) if tier_pol else None
            except Exception:
                tier_pol = None
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
            k_long=plan.k_long,
            k_short=plan.k_short,
            theta=plan.theta,
            final_r=plan.final_r,
            cap_used_pct=cap_used_pct,
            reservations=reservations_count,
            cycle_cleared=cycle_cleared,
            tier_summary=tier_summary,
        )

        report = self.risk_agent.check_invariants(self.state, equity, cap_pct)
        report_payload = report.to_dict() if hasattr(report, "to_dict") else {"ok": True}
        self.analytics.log_invariant(report_payload)
        if not report_payload.get("ok", True):
            print("[WARN] Risk invariant breach:", report_payload)

        self.persist.save()
        return plan

    def sync_to_time(self, current_time) -> None:
        if current_time is None:
            return
        if not self._is_backtest_like():
            return

        now_ts = current_time.timestamp()
        last_ts = self.state.last_finalized_bar_ts
        if last_ts is None or last_ts > now_ts or last_ts == 0.0:
            self.state.last_finalized_bar_ts = now_ts - self._tf_sec
            last_ts = self.state.last_finalized_bar_ts
            if self.state.treasury.cycle_start_tick == 0:
                self.state.treasury.cycle_start_tick = self.state.bar_tick
                self.state.treasury.cycle_start_equity = self.eq_provider.get_equity()

        delta_seconds = now_ts - last_ts
        if delta_seconds < self._tf_sec:
            return

        bars_passed = int(delta_seconds // self._tf_sec)
        if bars_passed <= 0:
            return
        self.state.bar_tick += bars_passed
        self.state.last_finalized_bar_ts += (bars_passed * self._tf_sec)
        self._decay_and_cooldowns(bars_passed)

        cycle_len = int(self.cfg.cycle_len_bars)
        start_tick = self.state.treasury.cycle_start_tick
        if (self.state.bar_tick - start_tick) >= cycle_len:
            equity = self.eq_provider.get_equity()
            if (equity - self.state.treasury.cycle_start_equity) >= 0 and self.cfg.risk.clear_debt_on_profitable_cycle:
                self.state.debt_pool = 0.0
                for pst in self.state.per_pair.values():
                    pst.local_loss = 0.0
                    pst.closs = 0
            self.state.treasury.cycle_start_tick = self.state.bar_tick
            self.state.treasury.cycle_start_equity = equity

        try:
            snapshot = self._build_snapshot()
            plan = self.treasury_agent.plan(snapshot, self.eq_provider.get_equity())
            self.state.treasury.k_long = plan.k_long
            self.state.treasury.k_short = plan.k_short
            self.state.treasury.theta = plan.theta
            self.state.treasury.final_r = plan.final_r
            self.state.treasury.available = plan.available
            self.state.treasury.bias = plan.bias
            self.state.treasury.volatility = plan.volatility
        except Exception:
            pass

        try:
            equity_now = float(self.eq_provider.get_equity())
        except Exception:
            equity_now = 0.0
        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity_now)
        report = self.risk_agent.check_invariants(self.state, equity_now, cap_pct)
        report_payload = report.to_dict() if hasattr(report, "to_dict") else {"ok": True}
        self.analytics.log_invariant(report_payload)
        if not report_payload.get("ok", True):
            print("[WARN] Risk invariant breach:", report_payload)
