import pandas as pd
import os
path='user_data/backtest_results/backtest-result-2025-11-22_15-33-52_signals.pkl'
print('exists', os.path.exists(path))
df=pd.read_pickle(path)
print(df.head())
print('rows', len(df))
print('tail signals\n', df[['date','enter_long','enter_short']].tail())
cut=pd.to_datetime('2025-01-05')
after=df[pd.to_datetime(df['date'])>=cut]
print('after rows', len(after))
print('after sums', after[['enter_long','enter_short']].sum())
print('first signals after cutoff', after.loc[(after.enter_long==1)|(after.enter_short==1), 'date'].head())
