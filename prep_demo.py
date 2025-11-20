import pandas as pd
from user_data.strategies.TaxBrainV29 import TaxBrainV29
cfg = {'stake_currency':'USDT','timeframe':'5m','dry_run':True}
strat = TaxBrainV29(cfg)
df = pd.DataFrame({'date':['2024-01-01'], 'high':[1.0], 'newbars_high':[2.0], 'atr':[3.0]})
prepared = strat._prepare_informative_frame(df.copy(), '30m')
print('columns:', prepared.columns.tolist())
cache = {'30m': prepared.copy()}
strat._informative_cache['BTC/USDT'] = cache
res = strat._get_informative_dataframe('BTC/USDT','30m')
print('cached columns:', res.columns.tolist())
