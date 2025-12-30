"""ExitPolicyV30 and TaxBrainV30 exit wiring unit tests."""

from __future__ import annotations

import pytest

import user_data.strategies.TaxBrainV30 as strat_mod
from user_data.strategies.core.engine import ActiveTradeMeta, PairState
from user_data.strategies.TaxBrainV30 import TaxBrainV30
from user_data.strategies.agents.exits.exit import ExitPolicyV30, ExitTags
from user_data.strategies.config.v30_config import V30Config


class DummyState:
    """Provide minimal state needed by ExitPolicyV30."""

    def __init__(self, cfg: V30Config):
        self.cfg = cfg
        self.per_pair = {}
        self.debt_pool = 0.0

    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]


class DummyEquity:
    """Fixed equity provider."""

    def __init__(self, value: float):
        self.value = value

    def get_equity(self) -> float:
        return self.value

    def set_equity(self, value: float) -> None:
        self.value = value


class DummyTrade:
    """Lightweight trade stub used by custom_stoploss/exit tests."""

    def __init__(self, trade_id: int = 1, is_long: bool = True):
        self.trade_id = trade_id
        self.open_rate = 100.0
        self.is_long = is_long
        self.user_data = {}
        self._custom = {}
        self.exit_reason = None

    def get_custom_data(self, key):
        return self._custom.get(key)

    def set_custom_data(self, key, value):
        self._custom[key] = value


def test_exit_policy_paths():
    """Validate exit tags emitted by ExitPolicyV30."""

    cfg = V30Config()
    state = DummyState(cfg)
    equity = DummyEquity(1000.0)
    exit_policy = ExitPolicyV30(state, equity, cfg)

    pair = "TEST/USDT"
    pst = state.get_pair_state(pair)
    pst.active_trades["1"] = ActiveTradeMeta(
        sl_pct=0.02,
        tp_pct=0.04,
        direction="long",
        entry_bar_tick=0,
        entry_price=100.0,
        bucket="fast",
        icu_bars_left=3,
    )

    assert exit_policy.decide(pair, "1", current_profit_pct=0.05) == ExitTags.TP_HIT

    pst.active_trades["1"].icu_bars_left = 0
    assert exit_policy.decide(pair, "1", current_profit_pct=0.0) == ExitTags.ICU_TIMEOUT

    pst.active_trades["1"].icu_bars_left = 5
    pst.last_dir = "short"
    pst.last_score = 0.05
    assert exit_policy.decide(pair, "1", current_profit_pct=0.0) == ExitTags.flip("short")

    pst.last_dir = None
    pst.last_score = 0.0
    state.debt_pool = 500.0
    assert exit_policy.decide(pair, "1", current_profit_pct=-0.01) == ExitTags.RISK_OFF


@pytest.mark.parametrize("router_value", [0.05, None])
def test_custom_stoploss_router_behavior(monkeypatch, tmp_path, router_value):
    """Custom stoploss should rely solely on ExitRouter outputs."""

    config = {
        "strategy_params": {"timeframe": "5m", "startup_candle_count": 50},
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV30(config)
    trade = DummyTrade(trade_id=42)
    pair = "BTC/USDT"

    monkeypatch.setattr(
        strat_mod.EXIT_ROUTER,
        "_sl_rules",
        [(0, "test_stub", lambda ctx: router_value)],
    )

    result = strategy.custom_stoploss(pair, trade, None, 100.0, 0.02, False)
    if router_value is None:
        assert result == strategy.stoploss
    else:
        assert result == pytest.approx(-router_value)


def test_custom_exit_records_reason(monkeypatch, tmp_path):
    """ExitPolicy decisions should be recorded and returned."""

    config = {
        "strategy_params": {"timeframe": "5m", "startup_candle_count": 50},
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV30(config)
    trade = DummyTrade()
    monkeypatch.setattr(strategy.exit_policy, "decide", lambda *args, **kwargs: ExitTags.ICU_TIMEOUT)

    reason = strategy.custom_exit("BTC/USDT", trade, None, 100.0, 0.05)
    assert reason == ExitTags.ICU_TIMEOUT
    assert trade.exit_reason == ExitTags.ICU_TIMEOUT
    assert trade.get_custom_data("exit_tag") == ExitTags.ICU_TIMEOUT
