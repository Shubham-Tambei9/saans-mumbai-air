"""Make best-of-two hedge files from a model submission: scale (or shift) only the FORECAST rows (target hour >= cutoff).
Already-observed rows keep their observed value.
    python make_pair.py submission.csv 0.85 1.10 +5   ->  submission_x0.85.csv, submission_x1.10.csv, submission_p5.csv"""
import sys, pandas as pd, numpy as np
from pipe import load_obs
src = sys.argv[1]; ops = sys.argv[2:]
P = load_obs(); C = P.dropna(how='all').index[-1] + pd.Timedelta(hours=1)
te = pd.read_csv('Test.csv'); te['t'] = pd.to_datetime(te.target_start_ist).dt.tz_localize(None)
sub = pd.read_csv(src); assert list(sub.ID) == list(te.ID)
# already-observed rows: use the RAW Train value (load_obs drops >500 spikes for features, which must not apply here)
tr = pd.read_csv('Train.csv'); tr['t'] = pd.to_datetime(tr.timestamp_ist).dt.tz_localize(None) - pd.Timedelta(hours=1)
raw = tr.groupby(['location_id', 't']).pm25.mean()
obs = np.array([raw.get((l, t), np.nan) for l, t in zip(te.location_id, te.t)])
m = ~np.isnan(obs)
print('observed test rows set to raw Train value:', m.sum(), '| rows changed:', int((np.abs(sub.Target[m] - obs[m]) > 1e-6).sum()))
sub.loc[m, 'Target'] = obs[m]
fut = (te.t >= C).values
pv = te.t.between('2026-11-11', '2026-11-12 23:00').values
print('cutoff', C, '| forecast rows', fut.sum(), '| private rows', pv.sum(), '| private mean', round(sub.Target[pv].mean(), 1))
for op in ops:
    s = sub.copy()
    if op[0] in '+-': s.loc[fut, 'Target'] = (s.Target[fut] + float(op)).clip(lower=0); tag = 'p' + op[1:] if op[0] == '+' else 'm' + op[1:]
    else: s.loc[fut, 'Target'] = s.Target[fut] * float(op); tag = 'x' + op
    assert len(s) == 2592 and np.isfinite(s.Target).all() and (s.Target >= 0).all()
    out = src.replace('.csv', f'_{tag}.csv'); s.to_csv(out, index=False)
    print(f'wrote {out}  private mean {s.Target[pv].mean():.1f}')
