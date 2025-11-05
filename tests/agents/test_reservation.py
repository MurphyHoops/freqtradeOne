"""ReservationAgent 生命周期相关的单元测试。"""

import pytest

from user_data.strategies.agents.reservation import ReservationAgent
from user_data.strategies.config.v29_config import V29Config


class AnalyticsStub:
    """记录预约日志事件的分析桩。"""

    def __init__(self):
        self.events = []

    def log_reservation(self, event, reservation_id, pair, bucket, risk):
        self.events.append((event, reservation_id, pair, bucket, risk))

    def log_finalize(self, *args, **kwargs):
        pass

    def log_exit(self, *args, **kwargs):
        pass

    def log_invariant(self, *args, **kwargs):
        pass

    def log_debug(self, *args, **kwargs):
        pass


def test_reservation_lifecycle_and_metrics_reset():
    """验证预约创建、释放、TTL 过期以及统计归零逻辑。"""

    cfg = V29Config()
    cfg.reservation_ttl_bars = 2
    analytics = AnalyticsStub()
    agent = ReservationAgent(cfg, analytics=analytics)

    rid = "PAIR:fast:123"
    agent.reserve("PAIR", rid, risk=10.0, bucket="fast")
    assert agent.get_total_reserved() == pytest.approx(10.0)
    assert agent.get_pair_reserved("PAIR") == pytest.approx(10.0)
    assert analytics.events[-1][0] == "create"

    agent.release(rid)
    assert agent.get_total_reserved() == pytest.approx(0.0)
    assert agent.get_pair_reserved("PAIR") == pytest.approx(0.0)
    assert analytics.events[-1][0] == "release"

    agent.reserve("PAIR", rid, risk=5.0, bucket="slow")
    agent.tick_ttl()
    agent.tick_ttl()
    assert rid not in agent.reservations
    assert analytics.events[-1][0] == "expire"
    assert agent.get_total_reserved() == pytest.approx(0.0)
