# -*- coding: utf-8 -*-
"""Pluggable sizing algorithms for SizerAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, TYPE_CHECKING

from ...config.v29_config import V29Config
from ...schemas import SizingContext
from .tier import TierPolicy

if TYPE_CHECKING:  # pragma: no cover - type only
    from ...TaxBrainV29 import GlobalState as PortfolioState
else:
    PortfolioState = Any


@dataclass
class Caps:
    """Nominal-space caps derived from risk rooms and static ceilings."""

    risk_room_nominal: float
    bucket_cap_nominal: float
    per_pair_nominal_room: float
    portfolio_nominal_room: float
    per_trade_nominal_cap: float
    exchange_max_nominal: float
    proposed_nominal_cap: float


@dataclass
class SizingInputs:
    ctx: SizingContext
    tier: TierPolicy
    pst: object  # pair state type used in SizerAgent
    equity: float

    lev: float
    sl_price_pct: float
    sl_roi_pct: float
    atr_pct_used: float | None
    atr_mul_sl: float

    base_nominal: float
    min_entry_nominal: float
    per_trade_cap_nominal: float
    exchange_cap_nominal: float

    base_risk: float
    baseline_risk: float
    bucket_label: str
    bucket_risk: float
    caps: Caps
    state: PortfolioState


@dataclass
class SizingResult:
    target_risk: float
    reason: str = ""
    recovery_risk: float = 0.0


def algo_base_only(inputs: SizingInputs, cfg: V29Config) -> SizingResult:
    """Return target risk (U) using only base risk and optional bucket allocation.

    Uses base_risk = base_nominal * sl_price_pct; ignores baseline and recovery so
    the returned target_risk stays close to the configured base size (plus bucket_risk).
    """

    del cfg  # currently unused, reserved for future knobs
    if inputs.base_nominal <= 0 or inputs.sl_price_pct <= 0:
        return SizingResult(target_risk=0.0, reason="BASE_ONLY_zero_budget")

    base_risk = inputs.base_risk
    bucket_risk = inputs.bucket_risk or 0.0
    target_risk = max(base_risk, bucket_risk)
    return SizingResult(target_risk=max(target_risk, 0.0), reason="BASE_ONLY")


def algo_baseline(inputs: SizingInputs, cfg: V29Config) -> SizingResult:
    """Return target risk (U) matching legacy BASELINE = base risk plus baseline top-up.

    Risk units are account currency; risk derives from nominal via sl_price_pct.
    Recovery component is simply baseline_risk - base_risk, capped at zero.
    """

    del cfg  # placeholder for future use
    base_risk = inputs.base_risk
    baseline_risk = inputs.baseline_risk
    bucket_risk = inputs.bucket_risk or 0.0

    if baseline_risk <= 0 and bucket_risk <= 0:
        return SizingResult(target_risk=0.0, reason="BASELINE_zero_budget")

    recovery_risk = max(0.0, baseline_risk - base_risk)
    target_risk = max(base_risk + recovery_risk, bucket_risk, baseline_risk)
    return SizingResult(target_risk=max(target_risk, 0.0), reason="BASELINE", recovery_risk=recovery_risk)


def algo_target_recovery(inputs: SizingInputs, cfg: V29Config) -> SizingResult:
    """Return ATR-style TARGET_RECOVERY risk (U), including loss/bucket recovery.

    Units:
    - inputs.sl_price_pct: price-space SL pct; risk = nominal * sl_price_pct.
    - target_risk: risk budget in account currency (U).
    - base_nominal: baseline nominal; base_risk = base_nominal * sl_price_pct; bucket_risk is optional top-up.

    Recovery mirrors ATR sizing: desired_nominal ~ B0 + local_loss / recovery_price_pct + bucket_risk / recovery_price_pct.
    In ATR-based mode, recovery_price_pct uses the TP ATR distance (tp_price_pct ~ atr_pct_used * atr_mul_tp);
    otherwise recovery falls back to sl_price_pct so non-ATR configs stay unchanged while risk/caps still hinge on sl_price_pct.
    """

    tr_cfg = cfg.sizing_algos.target_recovery
    sl_price_pct = inputs.sl_price_pct
    tier = inputs.tier
    pst = inputs.pst
    base_nominal = inputs.base_nominal
    bucket_risk = inputs.bucket_risk or 0.0
    ctx = inputs.ctx
    recovery_price_pct = sl_price_pct  # default to SL distance; ATR mode may swap this to TP distance for recovery sizing
    base_risk = inputs.base_risk
    bucket_for_recovery = inputs.bucket_risk if tr_cfg.include_bucket_in_recovery else 0.0
    debt_pool_component = 0.0
    if tr_cfg.include_debt_pool:
        if cfg.treasury.debt_pool_cap_pct > 0:
            debt_pool_component = min(inputs.state.debt_pool, cfg.treasury.debt_pool_cap_pct * inputs.equity)
        else:
            debt_pool_component = inputs.state.debt_pool

    B0 = max(cfg.sizing.static_initial_nominal, 0.0)
    L_total = pst.local_loss * tier.recovery_factor
    rec_bucket = bucket_for_recovery + debt_pool_component

    if sl_price_pct <= 0:
        return SizingResult(target_risk=0.0, reason="TARGET_RECOVERY_sl_zero")

    if tr_cfg.use_atr_based and inputs.atr_pct_used is not None and inputs.atr_pct_used > 0:
        # Anchor recovery sizing to the TP ATR distance when ATR-based; risk remains SL-anchored.
        exit_profile_name = ctx.exit_profile or getattr(tier, "default_exit_profile", None)
        profile = cfg.exit_profiles.get(exit_profile_name) if exit_profile_name else None
        atr_mul_tp = getattr(profile, "atr_mul_tp", None) if profile else None
        if atr_mul_tp is not None and atr_mul_tp > 0:
            tp_price_pct = inputs.atr_pct_used * atr_mul_tp
            if tp_price_pct and tp_price_pct > 0:
                recovery_price_pct = tp_price_pct

    L_bucket = rec_bucket
    if recovery_price_pct <= 0:
        return SizingResult(target_risk=0.0, reason="TARGET_RECOVERY_no_recovery_price")

    S_recover_loss_raw = L_total / recovery_price_pct
    S_recover_bucket_raw = L_bucket / recovery_price_pct

    alpha = getattr(tr_cfg, "bucket_scaling_factor", 1.0)
    S_recover_loss = S_recover_loss_raw
    S_recover_bucket = alpha * S_recover_bucket_raw
    S_total_target_nominal = max(B0 + S_recover_loss + S_recover_bucket, 0.0)

    if tr_cfg.max_recovery_multiple and tr_cfg.max_recovery_multiple > 0:
        S_total_target_nominal = min(S_total_target_nominal, base_nominal * tr_cfg.max_recovery_multiple)

    desired_nominal = max(base_nominal, S_total_target_nominal)
    desired_risk = desired_nominal * sl_price_pct
    target_risk = max(desired_risk, bucket_risk or 0.0)
    recovery_risk = max(0.0, desired_risk - base_risk)
    return SizingResult(target_risk=max(target_risk, 0.0), reason="TARGET_RECOVERY", recovery_risk=recovery_risk)


AlgoFn = Callable[[SizingInputs, V29Config], SizingResult]

ALGO_REGISTRY: Dict[str, AlgoFn] = {
    "BASE_ONLY": algo_base_only,
    "BASELINE": algo_baseline,
    "TARGET_RECOVERY": algo_target_recovery,
}
