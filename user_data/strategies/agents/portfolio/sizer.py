# -*- coding: utf-8 -*-
"""Sizing agent: combine tier policy, treasury allocation, recovery target and caps."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from ...config.v29_config import SizingAlgoConfig, V29Config, get_exit_profile
from .schemas import SizingContext
from .sizing_algos import ALGO_REGISTRY, Caps, SizingInputs
from .reservation import ReservationAgent
from .tier import TierManager, TierPolicy
from .global_backend import GlobalRiskBackend


class SizerAgent:
    """Combine sizing policies into a final pre-leverage stake and risk."""

    def __init__(
        self,
        state,
        reservation: ReservationAgent,
        eq_provider,
        cfg: V29Config,
        tier_mgr: TierManager,
        backend: Optional[GlobalRiskBackend] = None,
    ) -> None:
        self._log = logging.getLogger(__name__)
        self.state = state
        self.reservation = reservation
        self.eq = eq_provider
        self.cfg = cfg
        self.tier_mgr = tier_mgr
        self.backend = backend
        self._debug_file = (
            Path(getattr(cfg, "user_data_dir", "user_data")) / "logs" / "sizer_debug.log"
        )
        self._validate_sizing_algos()
        self.dp = None

    def set_dataprovider(self, dp) -> None:
        """Inject DataProvider for dynamic min_notional lookups."""
        self.dp = dp

    def _log_debug(self, msg: str) -> None:
        try:
            self._debug_file.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().isoformat()
            with open(self._debug_file, "a", encoding="utf-8") as handle:
                handle.write(f"{ts} {msg}\n")
        except Exception:
            pass

    def _validate_sizing_algos(self) -> None:
        """Fail fast when tiers reference deprecated/unknown algos."""

        valid_algos = set(ALGO_REGISTRY.keys())
        requested = set()
        try:
            requested.add(str(self.cfg.sizing_algos.default_algo))
        except Exception:
            pass

        try:
            tiers = getattr(getattr(self.cfg, "strategy", None), "tiers", getattr(self.cfg, "tiers", {})) or {}
            for tier in tiers.values():
                algo_name = getattr(tier, "sizing_algo", None)
                if algo_name:
                    requested.add(str(algo_name))
        except Exception:
            # Fall back to default only; nothing else to validate
            pass

        unknown = [algo for algo in requested if algo not in valid_algos]
        if unknown:
            raise ValueError(
                f"Unsupported sizing_algos configured: {unknown}; supported options: {sorted(valid_algos)}"
            )

    def compute(
        self,
        pair: str,
        sl_pct: float,
        tp_pct: float,
        direction: str,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float = 1.0,
        proposed_stake: Optional[float] = None,
        plan_atr_pct: Optional[float] = None,
        exit_profile: Optional[str] = None,
        bucket: Optional[str] = None,
        current_rate: float = 0.0,
        score: float = 0.0,
    ) -> Tuple[float, float, str]:
        ctx = SizingContext(
            pair=pair,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            direction=direction,
            min_stake=float(min_stake) if min_stake is not None else 0.0,
            max_stake=float(max_stake or 0.0),
            leverage=float(leverage or 0.0),
            proposed_stake=proposed_stake,
            plan_atr_pct=plan_atr_pct,
            exit_profile=exit_profile,
            bucket=bucket,
            current_rate=current_rate,
            score=float(score or 0.0),
        )
        return self._compute_internal(ctx)

    def _compute_internal(self, ctx: SizingContext) -> Tuple[float, float, str]:
        equity = self.eq.get_equity()
        pst = self.state.get_pair_state(ctx.pair)
        tier_pol = self.tier_mgr.get(pst.closs)
       
        bucket, vector_k = self._resolve_vector_k(ctx.direction)

        if tier_pol and getattr(tier_pol, "single_position_only", False) and pst.active_trades:
            return (0.0, 0.0, bucket)
        if ctx.sl_pct <= 0:
            return (0.0, 0.0, bucket)

        lev, sl_price_pct, sl_roi_pct, atr_pct_used, atr_mul_sl = self._derive_sl_context(ctx, tier_pol, pst)
        if sl_price_pct <= 0:
            return (0.0, 0.0, bucket)

        base_nominal, min_entry_nominal, per_trade_cap_nominal, exchange_cap_nominal = self._compute_base_nominal(
            ctx, tier_pol, equity, lev, ctx.current_rate
        )
        if base_nominal <= 0 or min_entry_nominal <= 0:
            self._log_debug(
                f"skip sizing base_nominal={base_nominal:.6f} min_entry_nominal={min_entry_nominal:.6f} pair={ctx.pair}"
            )
            return (0.0, 0.0, bucket)
        base_risk = base_nominal * sl_price_pct
        baseline_risk = self._baseline_risk(equity, tier_pol)
        caps = self._compute_caps(
            ctx,
            tier_pol,
            equity,
            sl_price_pct,
            vector_k,
            per_trade_cap_nominal,
            exchange_cap_nominal,
            lev,
        )
        if caps.risk_room_nominal <= 0:
            return (0.0, 0.0, bucket)

        # 中央分配债务
        score_val = float(ctx.score or 0.0)
        if score_val <= 0:
            try:
                score_val = float(pst.last_score)
            except Exception:
                score_val = 0.0
        fluid = 0.0
        score_cfg = getattr(self.cfg, "sizing_algos", getattr(self.cfg, "sizing", None))
        score_floor = float(getattr(score_cfg, "score_floor", 0.3) or 0.0) if score_cfg else 0.3
        score_exp = float(getattr(score_cfg, "score_exponent", 2.0) or 2.0) if score_cfg else 2.0
        clamped_score = max(0.0, min(1.0, score_val))
        if vector_k > 0 and clamped_score > score_floor:
            span = max(1e-9, 1.0 - score_floor)
            normalized = max(0.0, clamped_score - score_floor) / span
            try:
                shaped = normalized ** score_exp if score_exp > 0 else normalized
            except Exception:
                shaped = normalized
            fluid = vector_k * shaped

        c_target_risk =0
        if fluid>0:
            c_inputs = SizingInputs(
                ctx=ctx,
                tier=tier_pol,
                pst=pst,
                equity=equity,
                lev=lev,
                sl_price_pct=sl_price_pct,
                sl_roi_pct=sl_roi_pct,
                atr_pct_used=atr_pct_used,
                atr_mul_sl=atr_mul_sl,
                base_nominal=base_nominal,
                debet = fluid,
                min_entry_nominal=min_entry_nominal,
                per_trade_cap_nominal=per_trade_cap_nominal,
                exchange_cap_nominal=exchange_cap_nominal,
                base_risk=base_risk,
                baseline_risk=baseline_risk,
                bucket_label=bucket,
                caps=caps,
                state=self.state,
            )

            c_algo_name = getattr(tier_pol, "center_algo", self.cfg.sizing_algos.default_algo)
            c_algo_fn = ALGO_REGISTRY.get(c_algo_name)
            if not c_algo_fn:
                c_algo_fn = ALGO_REGISTRY["BASE_ONLY"]
            c_result = c_algo_fn(c_inputs, self.cfg)
            c_target_risk = max(c_result.target_risk, 0.0)

        # 地方
        local_inputs = SizingInputs(
            ctx=ctx,
            tier=tier_pol,
            pst=pst,
            equity=equity,
            lev=lev,
            sl_price_pct=sl_price_pct,
            sl_roi_pct=sl_roi_pct,
            atr_pct_used=atr_pct_used,
            atr_mul_sl=atr_mul_sl,
            base_nominal=base_nominal,
            debet = pst.local_loss,
            min_entry_nominal=min_entry_nominal,
            per_trade_cap_nominal=per_trade_cap_nominal,
            exchange_cap_nominal=exchange_cap_nominal,
            base_risk=base_risk,
            baseline_risk=baseline_risk,
            bucket_label=bucket,
            caps=caps,
            state=self.state,
        )
        algo_name = getattr(tier_pol, "sizing_algo", self.cfg.sizing_algos.default_algo)
        algo_fn = ALGO_REGISTRY.get(algo_name)
        if not algo_fn:
            algo_fn = ALGO_REGISTRY["BASE_ONLY"]
        result = algo_fn(local_inputs, self.cfg)
        target_risk = max(result.target_risk, 0.0)
        if target_risk + c_target_risk <= 0:
            return (0.0, 0.0, bucket)
 
        # 汇总
        nominal_target = (target_risk + c_target_risk) / sl_price_pct if sl_price_pct > 0 else 0.0
        stake_nominal = self._apply_caps(nominal_target, caps, min_entry_nominal)
        if stake_nominal <= 0:
            return (0.0, 0.0, bucket)

        stake_margin = stake_nominal / lev
        risk_final = stake_margin * sl_roi_pct

        if self.backend:
            backend_room = self._backend_risk_room(equity)
            if risk_final > (backend_room + 1e-9):
                self._warn_backend_race(ctx.pair, risk_final, backend_room)
                return (0.0, 0.0, bucket)

        self._log_debug(
            f"pair={ctx.pair} closs={pst.closs} score={score_val:.4f} "
            f"bucket={bucket} vector_k={vector_k:.6f} risk_room_nominal={caps.risk_room_nominal:.6f} "
            f"base_nominal={base_nominal:.6f} fluid={fluid:.6f} score_floor={score_floor:.3f} score_exp={score_exp:.2f} sl_price_pct={sl_price_pct:.6f} "
            f"atr_pct_used={atr_pct_used} atr_mul_sl={atr_mul_sl} nominal_target={nominal_target:.6f} "
            f"min_entry_nominal={min_entry_nominal:.6f} stake_nominal={stake_nominal:.6f} "
            f"stake_margin={stake_margin:.6f} risk_final={risk_final:.6f}"
        )

        return (float(stake_margin), float(risk_final), bucket)

    def _resolve_vector_k(self, direction: str) -> Tuple[str, float]:
        """Map signal方向 to极坐标拨款 K 值."""

        bucket = direction or "long"
        treasury = getattr(self.state, "treasury", None)
        if bucket == "short":
            k_val = float(getattr(treasury, "k_short", 0.0) if treasury else 0.0)
        else:
            k_val = float(getattr(treasury, "k_long", 0.0) if treasury else 0.0)
        return bucket, max(0.0, k_val)

    def _derive_sl_context(
        self, ctx: SizingContext, tier_pol: TierPolicy, pst
    ) -> Tuple[float, float, float, Optional[float], float]:
        """Return leverage, price-space SL pct, ROI-space SL pct, atr_pct and atr_mult used."""

        sizing_cfg = getattr(getattr(self.cfg, "trading", None), "sizing", getattr(self.cfg, "sizing", None))
        lev_cfg = float(sizing_cfg.enforce_leverage or 0.0) if sizing_cfg else 0.0
        lev_ctx = float(getattr(ctx, "leverage", 0.0) or 0.0)
        lev = lev_cfg or lev_ctx or 1.0

        plan_atr_pct: Optional[float] = None
        for candidate in (getattr(ctx, "plan_atr_pct", None), getattr(pst, "last_atr_pct", None)):
            if candidate is None:
                continue
            try:
                val = float(candidate)
            except Exception:
                continue
            if val > 0:
                plan_atr_pct = val
                break

        exit_profile_name = (
            ctx.exit_profile
            or getattr(tier_pol, "default_exit_profile", None)
            or getattr(getattr(self.cfg, "strategy", None), "default_exit_profile", getattr(self.cfg, "default_exit_profile", None))
        )
        atr_mul_sl = 1.0
        try:
            profile = get_exit_profile(self.cfg, exit_profile_name) if exit_profile_name else None
            if profile and getattr(profile, "atr_mul_sl", None):
                atr_mul_sl = float(profile.atr_mul_sl)
        except Exception:
            atr_mul_sl = 1.0

        if self.cfg.sizing_algos.target_recovery.use_atr_based and plan_atr_pct:
            sl_price_pct = max(plan_atr_pct * atr_mul_sl, 0.0)
        else:
            sl_price_pct = abs(ctx.sl_pct) / max(lev, 1e-9)
        sl_roi_pct = sl_price_pct * lev
        return lev, sl_price_pct, sl_roi_pct, plan_atr_pct, atr_mul_sl

    def _compute_base_nominal(
        self, ctx: SizingContext, tier_pol: TierPolicy, equity: float, lev: float, current_rate: float
    ) -> Tuple[float, float, float, float]:
        """Compute base nominal size ignoring recovery/buckets/caps."""

        sizing_cfg = getattr(getattr(self.cfg, "trading", None), "sizing", getattr(self.cfg, "sizing", None))
        mult = float(getattr(sizing_cfg, "min_stake_multiplier", 1.0) or 1.0) if sizing_cfg else 1.0
        max_seed_cap = float(getattr(sizing_cfg, "initial_max_nominal_cap", 0.0) or 0.0) if sizing_cfg else 0.0

        exchange_min = 5.0
        rate = float(current_rate or getattr(ctx, "current_rate", 0.0) or 0.0)
        if self.dp and rate > 0:
            try:
                market = None
                if hasattr(self.dp, "market"):
                    market = self.dp.market(ctx.pair)
                elif hasattr(self.dp, "get_market"):
                    market = self.dp.get_market(ctx.pair)
                limits = market.get("limits", {}) if isinstance(market, dict) else {}
                min_cost = (limits.get("cost") or {}).get("min") if limits else None
                min_amount = (limits.get("amount") or {}).get("min") if limits else None
                candidates = []
                if min_cost is not None:
                    try:
                        mc = float(min_cost)
                        if mc > 0:
                            candidates.append(mc)
                    except Exception:
                        pass
                if min_amount is not None:
                    try:
                        ma = float(min_amount)
                        if ma > 0:
                            candidates.append(ma * rate)
                    except Exception:
                        pass
                if candidates:
                    exchange_min = max(exchange_min, max(candidates))
            except Exception:
                pass

        if ctx.min_stake:
            try:
                exchange_min = max(exchange_min, float(ctx.min_stake) * lev)
            except Exception:
                pass

        base_nominal = max(exchange_min * mult, 0.0)
        if max_seed_cap > 0 and base_nominal > max_seed_cap:
            self._log_debug(
                f"FILTERED: {ctx.pair} min_req {base_nominal:.6f} > cap {max_seed_cap}"
            )
            return (0.0, 0.0, 0.0, 0.0)

        per_trade_cap_nominal = float(sizing_cfg.initial_max_nominal_per_trade or 0.0) if sizing_cfg else 0.0
        if per_trade_cap_nominal <= 0:
            per_trade_cap_nominal = float("inf")

        per_pair_cap_static = float(sizing_cfg.per_pair_max_nominal_static or 0.0) if sizing_cfg else 0.0
        if per_pair_cap_static <= 0:
            per_pair_cap_static = float("inf")

        min_entry_nominal = base_nominal if base_nominal > 0 else exchange_min
        base_nominal = min(base_nominal, per_trade_cap_nominal, per_pair_cap_static)

        max_notional_exchange = 0.0

        return (
            max(base_nominal, 0.0),
            max(min_entry_nominal, 0.0),
            per_trade_cap_nominal,
            float("inf") if max_notional_exchange <= 0 else max_notional_exchange,
        )

    def _compute_caps(
        self,
        ctx: SizingContext,
        tier_pol: TierPolicy,
        equity: float,
        sl_price_pct: float,
        vector_k: float,
        per_trade_cap_nominal: float,
        exchange_cap_nominal: float,
        lev: float,
    ) -> Caps:
        risk_room = self._available_risk_room(ctx.pair, equity, tier_pol)
        risk_room_nominal = risk_room / sl_price_pct if (risk_room > 0 and sl_price_pct > 0) else 0.0

        sizing_cfg = getattr(getattr(self.cfg, "trading", None), "sizing", getattr(self.cfg, "sizing", None))
        per_pair_cap_static = float(sizing_cfg.per_pair_max_nominal_static or 0.0) if sizing_cfg else 0.0
        used_stake = self.state.pair_stake_open.get(ctx.pair, 0.0)
        per_pair_nominal_room = float("inf")
        if per_pair_cap_static > 0:
            per_pair_nominal_room = max(0.0, per_pair_cap_static - used_stake)

        portfolio_cap_total = tier_pol.max_stake_notional_pct * equity
        total_open_nominal = sum(self.state.pair_stake_open.values()) if getattr(self.state, "pair_stake_open", None) else 0.0
        portfolio_nominal_room = max(0.0, portfolio_cap_total - total_open_nominal)

        bucket_cap_nominal = float("inf")
        # if vector_k > 0:
        #     bucket_cap_nominal = vector_k

        proposed_nominal_cap = float("inf")
        if ctx.proposed_stake and ctx.proposed_stake > 0:
            proposed_nominal_cap = float(ctx.proposed_stake) * lev

        return Caps(
            risk_room_nominal=risk_room_nominal,
            bucket_cap_nominal=bucket_cap_nominal,
            per_pair_nominal_room=per_pair_nominal_room,
            portfolio_nominal_room=portfolio_nominal_room,
            per_trade_nominal_cap=per_trade_cap_nominal if per_trade_cap_nominal > 0 else float("inf"),
            exchange_max_nominal=exchange_cap_nominal if exchange_cap_nominal > 0 else float("inf"),
            proposed_nominal_cap=proposed_nominal_cap,
        )

    def _compute_recovery_risk(
        self,
        ctx: SizingContext,
        tier_pol: TierPolicy,
        pst,
        bucket_risk: float,
        sizing_cfg: SizingAlgoConfig,
        base_nominal: float,
        sl_price_pct: float,
        equity: float,
        baseline_risk: float,
        algo: str,
    ) -> float:
        """Deprecated: sizing algorithms now live in sizing_algos.py."""

        raise RuntimeError("Use sizing_algos via SizerAgent pipeline instead.")

    def _apply_caps(self, nominal_target: float, caps: Caps, min_entry_nominal: float) -> float:
        """Clamp the nominal target by all caps, respecting exchange minimums."""

        candidates = [
            nominal_target,
            caps.risk_room_nominal,
            caps.bucket_cap_nominal,
            caps.per_pair_nominal_room,
            caps.portfolio_nominal_room,
            caps.per_trade_nominal_cap,
            caps.exchange_max_nominal,
            caps.proposed_nominal_cap,
        ]
        stake_nominal = min([c for c in candidates if c > 0], default=0.0)
        if stake_nominal <= 0:
            return 0.0

        if min_entry_nominal > 0 and stake_nominal < min_entry_nominal:
            self._log_debug(
                f"skip capacity_below_min_entry stake_nominal={stake_nominal:.6f} "
                f"min_entry_nominal={min_entry_nominal:.6f}"
            )
            return 0.0
        return stake_nominal

    def _baseline_risk(self, equity: float, tier_pol: TierPolicy) -> float:
        base = tier_pol.k_mult_base_pct * equity
        risk_cfg = getattr(self.cfg, "risk", self.cfg)
        if not self.cfg.suppress_baseline_when_stressed or equity <= 0:
            return base
        drawdown = self.state.debt_pool / equity
        if drawdown > risk_cfg.drawdown_threshold_pct:
            return 0.0
        return base

    def _backend_risk_room(self, equity: float) -> float:
        """Re-read backend snapshot to surface potential race conditions."""

        try:
            cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
            snapshot = self.backend.get_snapshot() if self.backend else None
            used = float(getattr(snapshot, "risk_used", 0.0)) if snapshot else 0.0
            return max(0.0, cap_pct * equity - used)
        except Exception:
            return 0.0

    def _warn_backend_race(self, pair: str, desired_risk: float, room: float) -> None:
        msg = (
            f"Risk room stale for {pair}: requested={desired_risk:.6f} "
            f"backend_room={room:.6f}; rejecting sizing to avoid over-allocation"
        )
        try:
            self._log.warning(msg)
        except Exception:
            self._log_debug(msg)

    def _available_risk_room(self, pair: str, equity: float, tier_pol: TierPolicy) -> float:
        cap_pct = self.state.get_dynamic_portfolio_cap_pct(equity)
        port_cap = cap_pct * equity
        if self.backend:
            backend_used = self.backend.get_snapshot().risk_used
            used = float(backend_used)
        else:
            used = self.state.get_total_open_risk() + self.reservation.get_total_reserved()
        port_room = max(0.0, port_cap - used)
        pair_reserved = self.reservation.get_pair_reserved(pair)
        pair_room = self.state.per_pair_cap_room(pair, equity, tier_pol, pair_reserved)
        return max(0.0, min(port_room, pair_room))
