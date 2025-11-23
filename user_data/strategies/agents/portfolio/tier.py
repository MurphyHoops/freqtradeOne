# -*- coding: utf-8 -*-
"""Tier-specific gating logic for candidate selection and sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, TYPE_CHECKING, Tuple

from ...config.v29_config import DEFAULT_TIERS, DEFAULT_TIER_ROUTING_MAP, TierSpec, V29Config

__all__ = [
    "TierPolicy",
    "TierManager",
    "TierAgent",
    "CLOSS_TO_TIER",
    "TIER_DEFAULT_PROFILE",
]

if TYPE_CHECKING:
    from ..signals.schemas import Candidate


@dataclass
class TierPolicy:
    """Runtime representation of a tier and its guard-rails."""

    name: str
    allowed_recipes: set[str]
    allowed_entries: set[str]
    allowed_squads: set[str]
    min_raw_score: float
    min_rr_ratio: float
    min_edge: float
    sizing_algo: Literal["BASE_ONLY", "BASELINE", "TARGET_RECOVERY"]
    k_mult_base_pct: float
    recovery_factor: float
    cooldown_bars: int
    cooldown_bars_after_win: int
    per_pair_risk_cap_pct: float
    max_stake_notional_pct: float
    icu_force_exit_bars: int
    priority: int = 100
    default_exit_profile: Optional[str] = None
    single_position_only: bool = False  # 新增字段

    def permits(
        self,
        *,
        kind: Optional[str] = None,
        squad: Optional[str] = None,
        recipe: Optional[str] = None,
    ) -> bool:
        """Check whether the candidate metadata is allowed by this tier."""

        if recipe:
            if recipe in self.allowed_recipes:
                return True
            # When recipes are explicitly configured, treat mismatches as hard rejects
            if self.allowed_recipes:
                return False
        if kind:
            if kind in self.allowed_entries:
                return True
            if self.allowed_entries:
                return False
        if squad and squad in self.allowed_squads:
            return True
        return not (self.allowed_recipes or self.allowed_entries or self.allowed_squads)


class TierManager:
    """Resolve TierPolicy objects for the current closs / routing state."""

    def __init__(self, cfg: V29Config) -> None:
        self._cfg = cfg
        self._routing = getattr(cfg, "tier_routing", None)
        tiers = getattr(cfg, "tiers", {}) or {}
        if not tiers:
            raise ValueError("Tier configuration is empty; supply at least one TierSpec.")
        self._policies: dict[str, TierPolicy] = {name: self._from_spec(spec) for name, spec in tiers.items()}
        self._ordered_policies = sorted(self._policies.values(), key=lambda pol: (-pol.priority, pol.name))
        self._routing_map = dict(getattr(self._routing, "loss_tier_map", {}) or {}) or dict(CLOSS_TO_TIER)
        self._tier_profile_defaults = {
            name: policy.default_exit_profile
            for name, policy in self._policies.items()
            if policy.default_exit_profile
        }
        if not self._tier_profile_defaults:
            self._tier_profile_defaults = dict(TIER_DEFAULT_PROFILE)

    def get(self, closs: int) -> TierPolicy:
        """Return the TierPolicy selected by the routing rules."""

        tier_name = self._resolve_tier_name(closs)
        if tier_name and tier_name in self._policies:
            return self._policies[tier_name]
        # fallback: deterministic first tier
        return next(iter(self._policies.values()))

    def get_by_name(self, tier_name: str) -> TierPolicy:
        """Fetch a tier policy directly by name."""

        return self._policies[tier_name]

    def policies(self) -> Tuple[TierPolicy, ...]:
        """Return all tier policies ordered by priority."""

        return tuple(self._ordered_policies)

    def default_profile_for_closs(self, closs: int) -> Optional[str]:
        """Return the default exit profile configured for the tier covering this closs."""

        tier_name = self._resolve_tier_name(closs)
        if not tier_name:
            return None
        policy = self._policies.get(tier_name)
        if policy and policy.default_exit_profile:
            return policy.default_exit_profile
        return self._tier_profile_defaults.get(tier_name) or TIER_DEFAULT_PROFILE.get(tier_name)

    def resolve_for_candidate(self, closs: int, candidate: Candidate) -> TierPolicy:
        """Return the tier that best accommodates the candidate."""

        primary = self.get(closs)
        if candidate and primary.permits(kind=candidate.kind, squad=candidate.squad, recipe=candidate.recipe):
            return primary
        for policy in self._ordered_policies:
            if policy.permits(kind=candidate.kind, squad=candidate.squad, recipe=candidate.recipe):
                return policy
        return primary

    @staticmethod
    def _from_spec(spec: TierSpec) -> TierPolicy:
        return TierPolicy(
            name=spec.name,
            allowed_recipes=set(spec.allowed_recipes),
            allowed_entries=set(spec.allowed_entries),
            allowed_squads=set(spec.allowed_squads),
            min_raw_score=spec.min_raw_score,
            min_rr_ratio=spec.min_rr_ratio,
            min_edge=spec.min_edge,
            sizing_algo=spec.sizing_algo,
            k_mult_base_pct=spec.k_mult_base_pct,
            recovery_factor=spec.recovery_factor,
            cooldown_bars=spec.cooldown_bars,
            cooldown_bars_after_win=spec.cooldown_bars_after_win,
            per_pair_risk_cap_pct=spec.per_pair_risk_cap_pct,
            max_stake_notional_pct=spec.max_stake_notional_pct,
            icu_force_exit_bars=spec.icu_force_exit_bars,
            priority=int(getattr(spec, "priority", 100)),
            default_exit_profile=getattr(spec, "default_exit_profile", None),
            single_position_only=getattr(spec, "single_position_only", False),
        )

    def _resolve_tier_name(self, closs: int) -> Optional[str]:
        tier_name = self._routing.resolve(closs) if self._routing else None
        if tier_name:
            return tier_name
        if not self._routing_map:
            return None
        for threshold, name in sorted(self._routing_map.items()):
            if closs <= threshold:
                return name
        return next(reversed(sorted(self._routing_map.items())))[1]


class TierAgent:
    """Filter candidate lists according to tier rules."""

    @staticmethod
    def filter_best(policy: TierPolicy, candidates: Sequence[Candidate]) -> Optional[Candidate]:
        """Select the best candidate that satisfies the tier thresholds."""

        ok: list[Candidate] = []
        for cand in candidates:
            if not policy.permits(kind=cand.kind, squad=cand.squad, recipe=cand.recipe):
                continue
            if cand.raw_score < policy.min_raw_score:
                continue
            if cand.rr_ratio < policy.min_rr_ratio:
                continue
            if cand.expected_edge < policy.min_edge:
                continue
            ok.append(cand)
        if not ok:
            return None
        ok.sort(key=lambda c: (c.expected_edge, c.raw_score), reverse=True)
        return ok[0]
CLOSS_TO_TIER: dict[int, str] = dict(DEFAULT_TIER_ROUTING_MAP)
TIER_DEFAULT_PROFILE: dict[str, str] = {
    name: spec.default_exit_profile
    for name, spec in DEFAULT_TIERS.items()
    if getattr(spec, "default_exit_profile", None)
}
