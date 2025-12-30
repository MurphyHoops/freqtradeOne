from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pytest

from user_data.strategies.core.engine import PairState
from user_data.strategies.TaxBrainV30 import TaxBrainV30
from user_data.strategies.agents.portfolio import global_backend as gb
from user_data.strategies.config.v30_config import V30Config

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None


class _DummyState:
    def __init__(self) -> None:
        self.per_pair: dict[str, PairState] = {}

    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]

    def get_dynamic_portfolio_cap_pct(self, equity: float) -> float:
        return 0.1


def _make_stub_strategy(gate_payload: dict | None = None) -> TaxBrainV30:
    strat = TaxBrainV30.__new__(TaxBrainV30)
    strat.state = _DummyState()
    strat.cfg = V30Config()
    strat.treasury_agent = SimpleNamespace(
        evaluate_signal_quality=lambda *args, **kwargs: gate_payload or {}
    )
    strat._pending_entry_meta = {}
    strat._last_signal = {}
    strat.reservation = mock.Mock()
    strat.sizer = mock.Mock()
    strat.eq_provider = SimpleNamespace(get_equity=lambda: 1000.0)
    strat.global_backend = None
    strat.logger = mock.Mock()
    strat._is_backtest_like_runmode = lambda: True
    strat.hub = SimpleNamespace(
        meta_for_id=lambda *args, **kwargs: None,
    )
    strat.bridge = SimpleNamespace(get_side_meta=lambda *args, **kwargs: None, get_row_meta=lambda *args, **kwargs: None)
    strat.engine = SimpleNamespace(is_permitted=lambda *args, **kwargs: True, sync_to_time=lambda *args, **kwargs: None)
    strat.rejections = SimpleNamespace(record=lambda *args, **kwargs: None)
    strat._reserve_risk_resources = lambda **kwargs: True
    return strat


@pytest.mark.skipif(redis is None, reason="redis package not installed")
def test_redis_backend_add_risk_usage_timeout(monkeypatch):
    mock_client = mock.Mock()
    mock_client.register_script.return_value = lambda *args, **kwargs: 1
    mock_client.mget.return_value = (0, 0)
    mock_client.incrbyfloat.side_effect = TimeoutError("timeout")
    monkeypatch.setattr(gb.redis, "Redis", lambda *args, **kwargs: mock_client)

    backend = gb.RedisGlobalBackend(host="localhost", port=6379, db=0, password=None, namespace="TEST:")
    assert backend.add_risk_usage(1.0) is False


def test_custom_stake_amount_rejects_when_gate_disallows():
    strat = _make_stub_strategy(gate_payload={"allowed": False, "reason": "blocked"})
    strat.sizer = mock.Mock()
    strat.sizer.compute.side_effect = AssertionError("sizer should not run when gate blocks")
    strat.engine = SimpleNamespace(
        is_permitted=lambda *args, **kwargs: False,
        sync_to_time=lambda *args, **kwargs: None,
    )
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)
    stake = strat.custom_stake_amount(
        pair="BTC/USDT",
        current_time=datetime(2024, 1, 1),
        current_rate=100.0,
        proposed_stake=10.0,
        min_stake=None,
        max_stake=1000.0,
        leverage=1.0,
        entry_tag=None,
        side="buy",
    )
    assert stake == 0.0


def test_confirm_trade_entry_no_cooldown_decrement_in_backtest():
    strat = _make_stub_strategy()
    strat.engine = SimpleNamespace(
        is_permitted=lambda *args, **kwargs: True,
        sync_to_time=mock.Mock(),
    )
    meta = {
        "signal_id": 7,
        "raw_score": 0.5,
        "rr_ratio": 2.0,
        "expected_edge": 0.6,
        "sl_pct": 0.01,
        "tp_pct": 0.02,
        "plan_atr_pct": 0.03,
    }
    meta_info = SimpleNamespace(
        name="mean_rev_long",
        direction="long",
        squad="MRL",
        exit_profile=None,
        recipe=None,
        timeframe="5m",
        plan_timeframe=None,
    )
    strat.bridge.get_side_meta = lambda *args, **kwargs: (meta, meta_info)

    pair = "BTC/USDT"
    pst = strat.state.get_pair_state(pair)
    pst.cooldown_bars_left = 5
    current_time = datetime(2024, 1, 1, 0, 0, 0)
    pst._cooldown_last_ts = float(current_time.timestamp()) - 600.0

    strat.confirm_trade_entry(
        pair=pair,
        order_type="limit",
        amount=1.0,
        rate=100.0,
        time_in_force="gtc",
        current_time=current_time,
        entry_tag=None,
        side="buy",
    )

    strat.engine.sync_to_time.assert_called_once()
