import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from user_data.strategies.config.v29_config import V29Config
from user_data.strategies.agents.signals import indicators
from user_data.strategies.agents.signals.builder import build_candidates, collect_indicator_requirements
from user_data.strategies.agents.portfolio.tier import TierManager
cfg=V29Config(); tier_mgr=TierManager(cfg)
path5=ROOT/'user_data/data/binanceusdm/futures/TRB_USDT_USDT-5m-futures.feather'
path30=ROOT/'user_data/data/binanceusdm/futures/TRB_USDT_USDT-30m-futures.feather'
df5=pd.read_feather(path5); df30=pd.read_feather(path30)
start=pd.Timestamp('2025-01-01', tz='UTC'); end=pd.Timestamp('2025-01-25', tz='UTC')
df5=df5[(df5['date']>=start)&(df5['date']<end)].copy(); df30=df30[(df30['date']>=start)&(df30['date']<end)].copy()
reqs=collect_indicator_requirements(cfg=cfg)
df5=indicators.compute_indicators(df5, cfg, required=reqs.get(None))
df30=indicators.compute_indicators(df30, cfg, required=reqs.get('30m'), suffix='30m', duplicate_ohlc=True)
left=pd.DataFrame({'_t':pd.to_datetime(df5['date'])}); right=df30.copy(); right['_tinfo']=pd.to_datetime(right['date'])
merged=pd.merge_asof(left.sort_values('_t'), right.sort_values('_tinfo'), left_on='_t', right_on='_tinfo', direction='backward'); merged.index=df5.index
def allowed_any(c):
    for policy in tier_mgr.policies():
        if not policy.permits(kind=c.kind, squad=c.squad, recipe=c.recipe):
            continue
        if c.raw_score < policy.min_raw_score or c.rr_ratio < policy.min_rr_ratio or c.expected_edge < policy.min_edge:
            continue
        return True
    return False
signals=0; long_short=0
for idx,row in df5.iterrows():
    cands=build_candidates(row, cfg, informative={'30m': merged.loc[idx]})
    cands=[c for c in cands if allowed_any(c)]
    grouped={'long':[], 'short':[]}
    for c in cands:
        grouped.setdefault(c.direction, []).append(c)
    for d in grouped:
        grouped[d]=sorted(grouped[d], key=lambda c:(c.expected_edge, c.raw_score), reverse=True)[:4]
    if grouped['long'] or grouped['short']:
        signals+=1
        if grouped['long'] and grouped['short']:
            long_short+=1
print('bars with entries', signals)
print('bars with both directions', long_short)
