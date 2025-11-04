from user_data.strategies.TaxBrainV29 import TaxBrainV29


def test_timeframe_startup_sync(tmp_path):
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
