"""ExitPolicyV29 及 TaxBrainV29 自定义止损逻辑的单元测试。"""

import pandas as pd
import pytest

from user_data.strategies.TaxBrainV29 import ActiveTradeMeta, PairState, TaxBrainV29
from user_data.strategies.agents.exit import ExitPolicyV29
from user_data.strategies.agents.exits.exit_tags import ExitTags
from user_data.strategies.config.v29_config import ExitProfile, V29Config


class DummyState:
    """提供最少属性以支撑退出策略测试。"""

    def __init__(self, cfg: V29Config):
        self.cfg = cfg
        self.per_pair = {}
        self.debt_pool = 0.0

    def get_pair_state(self, pair: str) -> PairState:
        if pair not in self.per_pair:
            self.per_pair[pair] = PairState()
        return self.per_pair[pair]


class DummyEquity:
    """返回固定数值的权益桩对象。"""

    def __init__(self, value: float):
        self.value = value

    def get_equity(self) -> float:
        return self.value

    def set_equity(self, value: float) -> None:
        self.value = value


class DummyTrade:
    """模拟 Freqtrade Trade，用于 custom_stoploss 测试。"""

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
    """覆盖 tp_hit、icu_timeout、flip 和 risk_off 等路径。"""

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


def test_custom_stoploss_fallbacks_and_breakeven(tmp_path):
    """验证三重 tp/sl 兜底及早锁盈触发逻辑。"""

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

    sl = strategy.custom_stoploss(pair, trade, None, 100.0, 0.0, False)
    assert sl == pytest.approx(-0.03)

    trade.set_custom_data("sl_pct", 0.05)
    trade.set_custom_data("tp_pct", 0.08)
    sl = strategy.custom_stoploss(pair, trade, None, 100.0, 0.0, False)
    assert sl == pytest.approx(-0.05)

    trade.set_custom_data("sl_pct", 0.02)
    trade.set_custom_data("tp_pct", 0.04)

    df = pd.DataFrame({"atr": [1.0], "close": [100.0]})
    strategy.dp.get_analyzed_dataframe = lambda *args, **kwargs: (df, None)

    lock = strategy.custom_stoploss(pair, trade, None, 110.0, 0.04, False)
    assert lock > -0.02

    # Exit profile should override baseline plan when sl/tp missing
    profile_name = "unit_profile"
    strategy.cfg.exit_profiles[profile_name] = ExitProfile(
        atr_mul_sl=1.5,
        floor_sl_pct=0.01,
        atr_mul_tp=3.0,
    )
    trade2 = DummyTrade(trade_id=99)
    trade2.set_custom_data("exit_profile", profile_name)
    trade2.set_custom_data("sl_pct", 0.0)
    trade2.set_custom_data("tp_pct", 0.0)
    trade2.set_custom_data("exit_profile", profile_name)
    pst.active_trades[str(trade2.trade_id)] = ActiveTradeMeta(
        sl_pct=0.0,
        tp_pct=0.0,
        direction="long",
        entry_bar_tick=0,
        entry_price=100.0,
        bucket="fast",
        icu_bars_left=None,
        exit_profile=profile_name,
    )

    df_profile = pd.DataFrame({"atr": [2.0], "close": [100.0]})
    strategy.dp.get_analyzed_dataframe = lambda *args, **kwargs: (df_profile, None)
    sl_profile = strategy.custom_stoploss(pair, trade2, None, 100.0, 0.0, False)
    # ATR% = 0.02, *1.5 => 0.03 (> floor 0.01)
    assert sl_profile == pytest.approx(-0.03)
