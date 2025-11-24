from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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

    def add_risk_usage(self, amount: float) -> bool:
        ...

    def release_risk_usage(self, amount: float) -> None:
        ...


class LocalGlobalBackend(GlobalRiskBackend):
    """In-memory backend used for single-instance runs."""

    def __init__(self) -> None:
        self._debt_pool: float = 0.0
        self._risk_used: float = 0.0

    def get_snapshot(self) -> GlobalSnapshot:
        return GlobalSnapshot(debt_pool=self._debt_pool, risk_used=self._risk_used)

    def add_loss(self, amount: float) -> None:
        self._debt_pool += float(amount)

    def repay_loss(self, amount: float) -> None:
        self._debt_pool = max(0.0, self._debt_pool - float(amount))

    def add_risk_usage(self, amount: float) -> bool:
        self._risk_used += float(amount)
        return True

    def release_risk_usage(self, amount: float) -> None:
        self._risk_used = max(0.0, self._risk_used - float(amount))


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

    def add_risk_usage(self, amount: float) -> bool:
        self._client.incrbyfloat(self._key_risk_used, float(amount))
        return True

    def release_risk_usage(self, amount: float) -> None:
        self._client.incrbyfloat(self._key_risk_used, -float(amount))
