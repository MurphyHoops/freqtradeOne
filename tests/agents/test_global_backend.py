"""Tests for GlobalRiskBackend implementations."""

import uuid

import pytest

from user_data.strategies.agents.portfolio.global_backend import (
    LocalGlobalBackend,
    RedisGlobalBackend,
)

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None


def test_local_backend_debt_and_repay_clamps_to_zero():
    backend = LocalGlobalBackend()
    backend.add_loss(5.5)
    backend.add_loss(4.5)
    backend.repay_loss(3.0)
    assert backend.get_snapshot().debt_pool == pytest.approx(7.0)

    backend.repay_loss(10.0)
    assert backend.get_snapshot().debt_pool == 0.0


def test_local_backend_risk_usage_tracks_and_clamps():
    backend = LocalGlobalBackend()
    backend.add_risk_usage(2.0)
    backend.add_risk_usage(1.5)
    backend.release_risk_usage(0.75)
    assert backend.get_snapshot().risk_used == pytest.approx(2.75)

    backend.release_risk_usage(10.0)
    assert backend.get_snapshot().risk_used == 0.0


def test_local_backend_percentile_and_window_bound():
    backend = LocalGlobalBackend()
    for i in range(1, 6):
        backend.record_signal_score("ETH/USDT", float(i) / 10.0)
    assert backend.get_score_percentile_threshold(50) == pytest.approx(0.3)

    for i in range(2000):
        backend.record_signal_score("BTC/USDT", float(i))
    assert len(getattr(backend, "_scores")) == 1000
    assert backend.get_score_percentile_threshold(90) == pytest.approx(1900.0)


@pytest.mark.skipif(redis is None, reason="redis package not installed")
def test_redis_backend_repay_clamps_to_zero_and_tracks_risk():
    client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    try:
        client.ping()
    except Exception:
        pytest.skip("redis server not available")

    namespace = f"TB_V30_TEST:{uuid.uuid4()}:"
    debt_key = f"{namespace}GLOBAL_DEBT"
    risk_key = f"{namespace}GLOBAL_RISK_USED"
    backend = RedisGlobalBackend(
        host="localhost", port=6379, db=0, password=None, namespace=namespace
    )

    client.delete(debt_key, risk_key)
    try:
        backend.add_loss(10.0)
        backend.repay_loss(25.0)
        assert backend.get_snapshot().debt_pool == 0.0

        backend.add_risk_usage(3.0)
        backend.add_risk_usage(1.0)
        backend.release_risk_usage(0.5)
        assert backend.get_snapshot().risk_used == pytest.approx(3.5)
    finally:
        client.delete(debt_key, risk_key)
