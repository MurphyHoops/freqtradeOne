import pandas as pd


def rank_pct_change(df: pd.DataFrame, n: int, out_col: str) -> pd.DataFrame:
    if "candle_begin_time" not in df.columns:
        return df
    source = f"PctChange_{n}"
    if source not in df.columns:
        return df
    df[out_col] = df.groupby("candle_begin_time")[source].rank(ascending=True, method="min")
    return df
