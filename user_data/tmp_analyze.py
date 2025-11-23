import zipfile, joblib
from io import BytesIO
zpath='user_data/backtest_results/backtest-result-2025-11-22_15-50-33.zip'
member='backtest-result-2025-11-22_15-50-33_signals.pkl'
with zipfile.ZipFile(zpath) as z:
    data=z.read(member)
obj=joblib.load(BytesIO(data))
for k,v in obj.items():
    for pair, df in v.items():
        print('pair', pair, 'len', len(df))
        print(df['date'])
