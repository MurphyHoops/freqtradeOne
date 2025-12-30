from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging


class RejectReason:
    COOLDOWN = "cooldown"
    SINGLE_POSITION = "single_position_only"
    GUARD = "guard"
    TIME_ALIGNMENT = "time_alignment"
    NO_CANDIDATE = "no_candidate"
    TIER_REJECT = "tier_reject"
    GATEKEEP = "gatekeeping"
    PORTFOLIO_CAP = "portfolio_cap"
    SIZER = "sizer"
    RESERVATION = "reservation"


@dataclass
class RejectTracker:
    log_enabled: bool = False
    stats_enabled: bool = True
    _counts: Dict[str, int] = field(default_factory=dict)
    _logger: Optional[logging.Logger] = None

    def record(self, reason: str, pair: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> None:
        if self.stats_enabled:
            self._counts[reason] = self._counts.get(reason, 0) + 1
        if not self.log_enabled:
            return
        logger = self._logger or logging.getLogger(__name__)
        if pair:
            logger.info("reject=%s pair=%s context=%s", reason, pair, context or {})
        else:
            logger.info("reject=%s context=%s", reason, context or {})

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counts)
