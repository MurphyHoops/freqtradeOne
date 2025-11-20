import pandas as pd

def strip_double(df, timeframe):
    suffix = timeframe.replace('/', '_')
    if not suffix:
        return df
    double = f'_{suffix}_{suffix}'
    drop=[]; rename={}
    for col in list(df.columns):
        if not col.endswith(double):
            continue
        base = col[:-len(f'_{suffix}')]
        if base in df.columns:
            drop.append(col)
        else:
            rename[col]=base
    if drop:
        df.drop(columns=drop, inplace=True, errors='ignore')
    if rename:
        df.rename(columns=rename, inplace=True)
    return df

df = pd.DataFrame({'high':[1.0],'high_30m':[2.0],'high_30m_30m':[3.0],'date':['2024-01-01']})
print('before', df.columns.tolist())
clean = strip_double(df.copy(), '30m')
print('after', clean.columns.tolist())
