from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Optional, List, Tuple

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - handled in tests
    redis = None


@dataclass(frozen=True)
class GlobalSnapshot:
    debt_pool: float
    risk_used: float


class GlobalRiskBackend(Protocol):
    def get_snapshot(self) -> GlobalSnapshot:
        ...

    def add_loss(self, amount: float) -> None:
        ...

    def repay_loss(self, amount: float) -> None:
        ...

    def add_risk_usage(self, amount: float, cap_limit: Optional[float] = None) -> bool:
        ...

    def release_risk_usage(self, amount: float) -> None:
        ...

    def record_signal_score(self, pair: str, score: float) -> None:
        ...

    def get_score_percentile_threshold(self, percentile: int) -> float:
        ...


class LocalGlobalBackend(GlobalRiskBackend):
    """In-memory backend used for single-instance runs."""

    def __init__(self) -> None:
        self._debt_pool: float = 0.0
        self._risk_used: float = 0.0
        self._scores: List[float] = []  # sliding window of recent scores

    def get_snapshot(self) -> GlobalSnapshot:
        return GlobalSnapshot(debt_pool=self._debt_pool, risk_used=self._risk_used)

    def add_loss(self, amount: float) -> None:
        self._debt_pool += float(amount)

    def repay_loss(self, amount: float) -> None:
        self._debt_pool = max(0.0, self._debt_pool - float(amount))

    def add_risk_usage(self, amount: float, cap_limit: Optional[float] = None) -> bool:
        amount = float(amount)
        if cap_limit is not None and (self._risk_used + amount) > float(cap_limit):
            return False
        self._risk_used += amount
        return True

    def release_risk_usage(self, amount: float) -> None:
        self._risk_used = max(0.0, self._risk_used - float(amount))

    def record_signal_score(self, pair: str, score: float) -> None:
        self._scores.append(float(score))
        # keep the window bounded to avoid unbounded growth during backtests
        if len(self._scores) > 1000:
            overflow = len(self._scores) - 1000
            if overflow > 0:
                self._scores = self._scores[overflow:]

    def get_score_percentile_threshold(self, percentile: int) -> float:
        if not self._scores:
            return 0.0
        scores = sorted(self._scores)
        pct = max(0.0, min(100.0, float(percentile)))
        idx = min(len(scores) - 1, math.floor(len(scores) * (pct / 100.0)))
        try:
            return float(scores[idx])
        except Exception:
            return 0.0


class RedisGlobalBackend(GlobalRiskBackend):
    """Redis-backed implementation for sharing global debt and risk across instances."""

    def __init__(
        self,
        host: str,
        port: int,
        db: int,
        password: str | None = None,
        namespace: str = "TB_V29:",
    ) -> None:
        if redis is None:
            raise ImportError("redis package is required for RedisGlobalBackend")

        self._namespace = namespace
        self._key_debt = f"{namespace}GLOBAL_DEBT"
        self._key_risk_used = f"{namespace}GLOBAL_RISK_USED"
        self._key_scores = f"{namespace}SCORES_WINDOW"
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )
        self._repay_script = self._client.register_script(
            """
            local key = KEYS[1]
            local amount = tonumber(ARGV[1])
            local current = tonumber(redis.call('GET', key) or '0')
            local new_value = current - amount
            if new_value < 0 then
                new_value = 0
            end
            redis.call('SET', key, new_value)
            return new_value
            """
        )
        self._risk_cap_script = self._client.register_script(
            """
            local key = KEYS[1]
            local amount = tonumber(ARGV[1])
            local cap = tonumber(ARGV[2])
            local current = tonumber(redis.call('GET', key) or '0')
            local new_value = current + amount
            if new_value > cap then
                return 0
            end
            redis.call('SET', key, new_value)
            return 1
            """
        )

    def _to_float(self, value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def get_snapshot(self) -> GlobalSnapshot:
        debt_val, risk_val = self._client.mget([self._key_debt, self._key_risk_used])
        return GlobalSnapshot(
            debt_pool=self._to_float(debt_val),
            risk_used=self._to_float(risk_val),
        )

    def add_loss(self, amount: float) -> None:
        self._client.incrbyfloat(self._key_debt, float(amount))

    def repay_loss(self, amount: float) -> None:
        self._repay_script(keys=[self._key_debt], args=[float(amount)])

    def add_risk_usage(self, amount: float, cap_limit: Optional[float] = None) -> bool:
        amount = float(amount)
        if cap_limit is None:
            self._client.incrbyfloat(self._key_risk_used, amount)
            return True
        result = self._risk_cap_script(keys=[self._key_risk_used], args=[amount, float(cap_limit)])
        return bool(result)

    def release_risk_usage(self, amount: float) -> None:
        self._client.incrbyfloat(self._key_risk_used, -float(amount))

    def record_signal_score(self, pair: str, score: float) -> None:
        member = f"{pair}:{int(time.time())}"
        score_val = float(score)
        self._client.zadd(self._key_scores, {member: score_val})
        # keep a rolling 1h window
        self._client.expire(self._key_scores, 3600)

    def get_score_percentile_threshold(self, percentile: int) -> float:
        total = int(self._client.zcard(self._key_scores) or 0)
        if total < 10:
            return 0.0
        idx = math.floor(total * (float(percentile) / 100.0))
        idx = min(max(idx, 0), total - 1)
        values = self._client.zrange(self._key_scores, idx, idx, withscores=True)
        if not values:
            return 0.0
        _, score = values[0]
        try:
            return float(score)
        except Exception:
            return 0.0
