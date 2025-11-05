"""验证 TaxBrainV29 对 timeframe/startup_candle_count 的同步行为。"""

from user_data.strategies.TaxBrainV29 import TaxBrainV29


def test_timeframe_startup_sync(tmp_path):
    """实例化后应同步实例与类属性，符合 V29.1 修订 #3。"""

    params = {
        "strategy_params": {
            "timeframe": "1h",
            "startup_candle_count": 123,
            "adx_len": 21,
        },
        "dry_run_wallet": 2000,
        "user_data_dir": str(tmp_path),
    }
    strategy = TaxBrainV29(params)

    assert strategy.timeframe == "1h"
    assert strategy.startup_candle_count == 123
    assert strategy.__class__.timeframe == "1h"
    assert strategy.__class__.startup_candle_count == 123
