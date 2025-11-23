import pandas as pd
from pathlib import Path
for tf in ['5m-futures','30m-futures','8h-funding_rate','8h-mark']:
    path=Path(f'user_data/data/binanceusdm/futures/TRB_USDT_USDT-{tf}.feather')
    df=pd.read_feather(path)
    print(tf, len(df), df['date'].min(), df['date'].max())
