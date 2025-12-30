from __future__ import annotations

import logging
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
    market_bias: float = 0.0
    market_volatility: float = 1.0


class GlobalRiskBackend(Protocol):
    def get_snapshot(self) -> GlobalSnapshot:
        ...

    def atomic_update_debt(self, delta: float) -> float:
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

    def set_market_metrics(self, bias: float, volatility: float) -> None:
        ...


class LocalGlobalBackend(GlobalRiskBackend):
    """In-memory backend used for single-instance runs."""

    def __init__(self) -> None:
        self._debt_pool: float = 0.0
        self._risk_used: float = 0.0
        self._market_bias: float = 0.0
        self._market_volatility: float = 1.0
        self._scores: List[float] = []  # sliding window of recent scores

    def get_snapshot(self) -> GlobalSnapshot:
        return GlobalSnapshot(
            debt_pool=self._debt_pool,
            risk_used=self._risk_used,
            market_bias=self._market_bias,
            market_volatility=self._market_volatility,
        )

    def atomic_update_debt(self, delta: float) -> float:
        self._debt_pool = max(0.0, self._debt_pool + float(delta))
        return self._debt_pool

    def add_loss(self, amount: float) -> None:
        self.atomic_update_debt(amount)

    def repay_loss(self, amount: float) -> None:
        self.atomic_update_debt(-float(amount))

    def set_market_metrics(self, bias: float, volatility: float) -> None:
        self._market_bias = float(max(-1.0, min(1.0, bias)))
        self._market_volatility = float(max(0.0, volatility if volatility is not None else 1.0) or 1.0)

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
        namespace: str = "TB_V30:",
    ) -> None:
        if redis is None:
            raise ImportError("redis package is required for RedisGlobalBackend")

        self._namespace = namespace
        self._log = logging.getLogger(__name__)
        self._key_debt = f"{namespace}GLOBAL_DEBT"
        self._key_risk_used = f"{namespace}GLOBAL_RISK_USED"
        self._key_scores = f"{namespace}SCORES_WINDOW"
        self._key_bias = f"{namespace}GLOBAL_MARKET_BIAS"
        self._key_vol = f"{namespace}GLOBAL_MARKET_VOL"
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_timeout=0.05,
            socket_connect_timeout=0.05,
        )
        self._debt_update_script = self._register_script(
            """
            local key = KEYS[1]
            local delta = tonumber(ARGV[1])
            local current = tonumber(redis.call('GET', key) or '0')
            local new_value = current + delta
            if new_value < 0 then
                new_value = 0
            end
            redis.call('SET', key, new_value)
            return new_value
            """
        )
        self._risk_cap_script = self._register_script(
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
        self._local_cache_snapshot: Optional[GlobalSnapshot] = None
        self._last_snapshot_time: float = 0.0
        self._cache_ttl: float = 1.0

    def _register_script(self, script: str):
        try:
            return self._client.register_script(script)
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error("Failed to register redis script", exc)
            return None

    def _safe_log_error(self, msg: str, exc: Exception) -> None:
        try:
            self._log.error(msg, exc_info=exc)
        except Exception:
            try:
                print(f"[RedisGlobalBackend] {msg}: {exc}")
            except Exception:
                pass

    def _to_float(self, value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _invalidate_snapshot_cache(self) -> None:
        self._local_cache_snapshot = None
        self._last_snapshot_time = 0.0

    def get_snapshot(self) -> GlobalSnapshot:
        now = time.time()
        if (
            self._local_cache_snapshot is not None
            and (now - self._last_snapshot_time) < self._cache_ttl
        ):
            return self._local_cache_snapshot

        try:
            pipe = self._client.pipeline()
            pipe.mget([self._key_debt, self._key_risk_used, self._key_bias, self._key_vol])
            (debt_val, risk_val, bias_val, vol_val) = pipe.execute()[0]
            snapshot = GlobalSnapshot(
                debt_pool=self._to_float(debt_val),
                risk_used=self._to_float(risk_val),
                market_bias=max(-1.0, min(1.0, self._to_float(bias_val))),
                market_volatility=self._to_float(vol_val) or 1.0,
            )
            self._local_cache_snapshot = snapshot
            self._last_snapshot_time = now
            return snapshot
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error("Redis get_snapshot failed", exc)
            return GlobalSnapshot(debt_pool=0.0, risk_used=0.0, market_bias=0.0, market_volatility=1.0)

    def add_loss(self, amount: float) -> None:
        self.atomic_update_debt(float(amount))

    def repay_loss(self, amount: float) -> None:
        self.atomic_update_debt(-float(amount))

    def atomic_update_debt(self, delta: float) -> float:
        try:
            if self._debt_update_script:
                value = self._debt_update_script(keys=[self._key_debt], args=[float(delta)])
            else:
                value = self._client.incrbyfloat(self._key_debt, float(delta))
                if value < 0:
                    self._client.set(self._key_debt, 0.0)
                    value = 0.0
            self._invalidate_snapshot_cache()
            return float(value)
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(f"Redis atomic_update_debt failed for delta={delta}", exc)
            return 0.0

    def add_risk_usage(self, amount: float, cap_limit: Optional[float] = None) -> bool:
        amount = float(amount)
        try:
            if cap_limit is None:
                self._client.incrbyfloat(self._key_risk_used, amount)
                self._invalidate_snapshot_cache()
                return True
            if self._risk_cap_script is None:
                return False
            result = self._risk_cap_script(keys=[self._key_risk_used], args=[amount, float(cap_limit)])
            self._invalidate_snapshot_cache()
            return bool(result)
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(
                f"Redis add_risk_usage failed for amount={amount} cap_limit={cap_limit}", exc
            )
            self._invalidate_snapshot_cache()
            return False

    def release_risk_usage(self, amount: float) -> None:
        try:
            self._client.incrbyfloat(self._key_risk_used, -float(amount))
            self._invalidate_snapshot_cache()
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(f"Redis release_risk_usage failed for amount={amount}", exc)

    def set_market_metrics(self, bias: float, volatility: float) -> None:
        """Atomically store latest market bias/volatility computed by MarketSensor."""

        bias = max(-1.0, min(1.0, float(bias)))
        volatility = float(volatility if volatility is not None else 1.0) or 1.0
        try:
            pipe = self._client.pipeline()
            pipe.set(self._key_bias, bias)
            pipe.set(self._key_vol, volatility)
            pipe.execute()
            self._invalidate_snapshot_cache()
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(
                f"Redis set_market_metrics failed for bias={bias} volatility={volatility}", exc
            )

    def record_signal_score(self, pair: str, score: float) -> None:
        try:
            member = f"{pair}:{int(time.time())}"
            score_val = float(score)
            self._client.zadd(self._key_scores, {member: score_val})
            # keep a rolling 1h window
            self._client.expire(self._key_scores, 3600)
            # trim to last N samples to cap memory
            self._client.zremrangebyrank(self._key_scores, 0, -2001)
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(f"Redis record_signal_score failed for pair={pair}", exc)

    def get_score_percentile_threshold(self, percentile: int) -> float:
        try:
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
        except (redis.RedisError, TimeoutError, Exception) as exc:  # type: ignore
            self._safe_log_error(
                f"Redis get_score_percentile_threshold failed for percentile={percentile}", exc
            )
            return 0.0
