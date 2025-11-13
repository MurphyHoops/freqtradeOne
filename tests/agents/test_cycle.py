"""CycleAgent 行为的单元测试。

包含盈利周期清债、预约 TTL 推进以及风险日志写入等路径的验证。
"""

import time
from types import SimpleNamespace

import pytest

from user_data.strategies.agents.cycle import CycleAgent
from user_data.strategies.config.v29_config import V29Config
from user_data.strategies.agents.treasury import AllocationPlan


class DummyPairState:
    """最小化的交易对状态对象，用于模拟 CycleAgent 所需字段。"""

    def __init__(self, local_loss: float = 0.0):
        self.cooldown_bars_left = 0
        self.active_trades = {}
        self.last_score = 0.0
        self.last_kind = None
        self.last_dir = None
        self.last_squad = None
        self.last_sl_pct = 0.02
        self.last_tp_pct = 0.04
        self.local_loss = local_loss
        self.closs = 0


class DummyTreasuryState:
    """跟踪 fast/slow 拨款及财政周期起点。"""

    def __init__(self):
        self.fast_alloc_risk = {}
        self.slow_alloc_risk = {}
        self.cycle_start_tick = 0
        self.cycle_start_equity = 0.0


class DummyState:
    """半仿真的 GlobalState，保留 CycleAgent 访问的核心接口。"""

    def __init__(self, debt_pool: float, cap_pct: float, total_open: float, pair_state: DummyPairState):
        self.debt_pool = debt_pool
        self._cap_pct = cap_pct
        self._total_open = total_open
        self.pair_risk_open = {"PAIR/USDT": 0.0}
        self.per_pair = {"PAIR/USDT": pair_state}
        self.treasury = DummyTreasuryState()
        self.bar_tick = 0
        self.current_cycle_ts = None
        self.last_finalized_bar_ts = None
        self.reported_pairs_for_current_cycle: set[str] = set()
        self.last_finalize_walltime = time.time()

    def get_pair_state(self, pair: str) -> DummyPairState:
        """返回（或懒初始化）指定交易对状态。"""

        if pair not in self.per_pair:
            self.per_pair[pair] = DummyPairState()
        return self.per_pair[pair]

    def get_total_open_risk(self) -> float:
        """提供组合在市风险，用于 CAP 计算。"""

        return self._total_open

    def get_dynamic_portfolio_cap_pct(self, _equity: float) -> float:
        """模拟组合 CAP 百分比。"""

        return self._cap_pct


class ReservationStub:
    """仅实现 CycleAgent 所需接口的预约桩对象。"""

    def __init__(self):
        self.total_reserved = 0.0
        self.pair_reserved = {"PAIR/USDT": 0.0}
        self.reservations = {}
        self.tick_calls = 0
        self._released = []

    def tick_ttl(self):
        self.tick_calls += 1

    def get_total_reserved(self) -> float:
        return self.total_reserved

    def get_pair_reserved(self, pair: str) -> float:
        return self.pair_reserved.get(pair, 0.0)

    def drain_recent_releases(self):
        rel = tuple(self._released)
        self._released.clear()
        return rel


class TreasuryStub:
    """返回固定拨款结果的财政桩。"""

    def __init__(self, fast: dict[str, float], slow: dict[str, float]):
        self.fast = fast
        self.slow = slow

    def plan(self, _snapshot: dict, _equity: float) -> AllocationPlan:
        return AllocationPlan(fast_alloc_risk=dict(self.fast), slow_alloc_risk=dict(self.slow))


class RiskStub:
    """记录被调用参数的风险桩。"""

    def __init__(self):
        self.calls = []

    def check_invariants(self, state, equity: float, cap_pct: float):
        self.calls.append((state.bar_tick, equity, cap_pct))
        return {"ok": True, "violations": []}


class AnalyticsStub:
    """收集 finalize 与 invariant 日志的桩实现。"""

    def __init__(self):
        self.finalize_calls = []
        self.invariant_calls = []

    def log_finalize(self, **kwargs):
        self.finalize_calls.append(kwargs)

    def log_invariant(self, report):
        self.invariant_calls.append(report)

    def log_reservation(self, *args, **kwargs):
        pass

    def log_exit(self, *args, **kwargs):
        pass

    def log_debug(self, *args, **kwargs):
        pass


class PersistStub:
    """统计保存次数的持久化桩。"""

    def __init__(self):
        self.saves = 0

    def save(self):
        self.saves += 1


class EquityStub:
    """简单的权益提供器，返回固定 equity。"""

    def __init__(self, equity: float):
        self.equity = equity

    def get_equity(self) -> float:
        return self.equity


def latest_finalize_entry(analytics: AnalyticsStub):
    """方便断言时获取最近一次 finalize 日志。"""

    if not analytics.finalize_calls:
        return None
    return analytics.finalize_calls[-1]


def test_cycle_finalize_clears_debt_on_profitable_cycle():
    """验证盈利周期触发清债，同时负盈利不应清空债务。"""

    cfg = V29Config()
    cfg.cycle_len_bars = 3
    cfg.clear_debt_on_profitable_cycle = True
    cfg.pain_decay_per_bar = 1.0
    state = DummyState(debt_pool=100.0, cap_pct=0.5, total_open=0.0, pair_state=DummyPairState(local_loss=50.0))
    reservation = ReservationStub()
    treasury = TreasuryStub({"PAIR/USDT": 20.0}, {"PAIR/USDT": 10.0})
    risk = RiskStub()
    analytics = AnalyticsStub()
    persist = PersistStub()
    eq = EquityStub(1000.0)

    agent = CycleAgent(cfg, state, reservation, treasury, risk, analytics, persist, tier_mgr=None)

    # 先积累多个周期，最后一次带正收益，触发清债。
    agent.finalize(eq)
    agent.finalize(eq)
    agent.finalize(eq)
    eq.equity = 1200.0
    agent.finalize(eq)

    assert state.bar_tick == 4
    assert reservation.tick_calls == 4
    assert persist.saves == 4
    assert state.debt_pool == pytest.approx(0.0)
    assert state.get_pair_state("PAIR/USDT").local_loss == pytest.approx(0.0)
    assert any(call.get("cycle_cleared", False) for call in analytics.finalize_calls)

    # 引入新债务并制造亏损周期，确认不会再次清债。
    state.debt_pool = 50.0
    state.get_pair_state("PAIR/USDT").local_loss = 25.0
    eq.equity = 1100.0  # 重设周期起点
    agent.finalize(eq)
    eq.equity = 900.0
    agent.finalize(eq)
    agent.finalize(eq)
    agent.finalize(eq)

    final_entries = [call for call in analytics.finalize_calls if "cycle_cleared" in call]
    assert final_entries[-1]["cycle_cleared"] is False
    last_entry = latest_finalize_entry(analytics)
    assert "tier_summary" in last_entry
    assert isinstance(last_entry["tier_summary"], dict)
