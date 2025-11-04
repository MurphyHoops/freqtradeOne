import pandas as pd
import pytest

from user_data.strategies.TaxBrainV29 import ActiveTradeMeta, PairState, TaxBrainV29
from user_data.strategies.agents.exit import ExitPolicyV29
from user_data.strategies.config.v29_config import V29Config


class DummyState:
    def __init__(self, cfg: V29Config):
        self.cfg = cfg
        self.per_pair = {}
        self.debt_pool = 0.0

    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]


class DummyEquity:
    def __init__(self, value: float):
        self.value = value

    def get_equity(self) -> float:
        return self.value

    def set_equity(self, value: float) -> None:
        self.value = value


class DummyTrade:
    def __init__(self, trade_id: int = 1, is_long: bool = True):
        self.trade_id = trade_id
        self.open_rate = 100.0
        self.is_long = is_long
        self.user_data = {}
        self._custom = {}

    def get_custom_data(self, key):
        return self._custom.get(key)

    def set_custom_data(self, key, value):
        self._custom[key] = value


def test_exit_policy_paths():
    cfg = V29Config()
    state = DummyState(cfg)
    equity = DummyEquity(1000.0)
    exit_policy = ExitPolicyV29(state, equity, cfg)

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

    # tp hit
    assert exit_policy.decide(pair, "1", current_profit_pct=0.05) == "tp_hit"

    # icu timeout
    pst.active_trades["1"].icu_bars_left = 0
    assert exit_policy.decide(pair, "1", current_profit_pct=0.0) == "icu_timeout"

    # flip direction
    pst.active_trades["1"].icu_bars_left = 5
    pst.last_dir = "short"
    pst.last_score = 0.05
    assert exit_policy.decide(pair, "1", current_profit_pct=0.0) == "flip_short"

    # risk off when stressed and losing
    pst.last_dir = None
    pst.last_score = 0.0
    state.debt_pool = 500.0
    assert exit_policy.decide(pair, "1", current_profit_pct=-0.01) == "risk_off"


def test_custom_stoploss_fallbacks_and_breakeven(tmp_path):
    config = {
        "strategy_params": {
            "timeframe": "5m",
            "startup_candle_count": 50,
        },
        "dry_run_wallet": 1000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV29(config)
    pair = "BTC/USDT"
    trade = DummyTrade(trade_id=77)

    pst = strategy.state.get_pair_state(pair)
    pst.active_trades[str(trade.trade_id)] = ActiveTradeMeta(
        sl_pct=0.03,
        tp_pct=0.06,
        direction="long",
        entry_bar_tick=0,
        entry_price=100.0,
        bucket="fast",
        icu_bars_left=None,
    )
    pst.last_atr_pct = 0.01

    # No custom data or user data -> fallback to meta
    sl = strategy.custom_stoploss(pair, trade, None, 100.0, 0.0, False)
    assert sl == pytest.approx(-0.03)

    # User data fallback
    trade.user_data["sl_pct"] = 0.05
    trade.user_data["tp_pct"] = 0.08
    sl = strategy.custom_stoploss(pair, trade, None, 100.0, 0.0, False)
    assert sl == pytest.approx(-0.05)

    # Custom data takes precedence and triggers breakeven lock
    trade.set_custom_data("sl_pct", 0.02)
    trade.set_custom_data("tp_pct", 0.04)

    df = pd.DataFrame({"atr": [1.0], "close": [100.0]})
    strategy.dp.get_analyzed_dataframe = lambda *args, **kwargs: (df, None)

    lock = strategy.custom_stoploss(pair, trade, None, 110.0, 0.04, False)
    assert lock > -0.02  # raised towards breakeven
