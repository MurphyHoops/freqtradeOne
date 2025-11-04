import sys
import types
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Provide a lightweight stub for freqtrade.strategy.IStrategy so strategy modules can import.
if 'freqtrade.strategy' not in sys.modules:
    freqtrade_module = types.ModuleType('freqtrade')
    strategy_module = types.ModuleType('freqtrade.strategy')

    class _IStrategy:
        def __init__(self, config):
            self.config = config
            # Data provider stub used by custom_stoploss.
            self.dp = types.SimpleNamespace(
                current_whitelist=lambda: [],
                get_analyzed_dataframe=lambda *args, **kwargs: ([], None),
            )

    strategy_module.IStrategy = _IStrategy
    sys.modules['freqtrade'] = freqtrade_module
    sys.modules['freqtrade.strategy'] = strategy_module
    freqtrade_module.strategy = strategy_module

import pandas as pd
import numpy as np

# Stub pandas_ta if the real library is unavailable
if 'pandas_ta' not in sys.modules:
    ta_module = types.ModuleType('pandas_ta')

    def _ensure_series(obj):
        return obj if isinstance(obj, pd.Series) else pd.Series(obj)

    def ema(series, length=10):
        s = _ensure_series(series)
        return s.ewm(span=length, adjust=False).mean()

    def rsi(series, length=14):
        s = _ensure_series(series).astype(float)
        delta = s.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(length, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(length, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val.fillna(0.0)

    def atr(high, low, close, length=14):
        high_s = _ensure_series(high).astype(float)
        low_s = _ensure_series(low).astype(float)
        close_s = _ensure_series(close).astype(float)
        prev_close = close_s.shift(1)
        tr = pd.concat([high_s - low_s, (high_s - prev_close).abs(), (low_s - prev_close).abs()], axis=1).max(axis=1)
        return tr.rolling(length, min_periods=1).mean().fillna(0.0)

    def adx(high, low, close, length=14):
        close_s = _ensure_series(close)
        values = pd.Series(np.full(len(close_s), 50.0), index=close_s.index)
        return pd.DataFrame({f'ADX_{length}': values})

    ta_module.ema = ema
    ta_module.rsi = rsi
    ta_module.atr = atr
    ta_module.adx = adx

    sys.modules['pandas_ta'] = ta_module

