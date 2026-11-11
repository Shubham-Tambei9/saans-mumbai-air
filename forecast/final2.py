"""Final submission. Re-run after downloading the newest Train.csv:
    python fetch_ext.py      # refresh weather + CAMS forecasts
    python final2.py         # -> submission.csv
Rows already observed in Train.csv get the observation; everything after the last observation is forecast."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings, sys
from pipe import *
warnings.filterwarnings('ignore')
SEEDS = (0, 1, 2, 3, 4)
P0 = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, reg_lambda=5, verbose=-1)
F = [f for f in FEATS if f not in ('doy', 'dwl')]
NOCAMS = '--nocams' in sys.argv   # use only Train.csv + Open-Meteo weather (no CAMS air-quality forecast)
CAMSF = ('pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'dust', 'aerosol_optical_depth', 'cams24', 'cams6',
         'bias7', 'cams_adj', 'cams24_adj', 'cams_anom')
if NOCAMS: F = [f for f in F if f not in CAMSF]
HL = 180   # recency half-life (days)
OUT = 'submission_nocams.csv' if NOCAMS else 'submission.csv'
P1 = dict(P0, num_leaves=15, min_child_samples=300, n_estimators=800, colsample_bytree=0.5)

P = load_obs(); E = load_ext()
last_obs = P.dropna(how='all').index[-1]
C = last_obs + pd.Timedelta(hours=1)
KEEPDW = '--keep-diwali' in sys.argv
MASKALL = '--mask-all-nights' in sys.argv   # also hide the 9->10 Nov firework night (everything after 8 Nov 12:00)
P_true = P.copy()          # untouched observations (used to fill already-observed test rows)
if not KEEPDW:             # hide Diwali-night firework spikes from the 'recent level' features
    mend = C - pd.Timedelta(hours=1) if MASKALL else pd.Timestamp('2026-11-09 12:00')
    P.loc['2026-11-08 12:00':mend] = np.nan
    print('masked festival hours 2026-11-08 12:00 ..', mend, 'from features')
if KEEPDW: OUT = OUT.replace('.csv', '_keepdiwali.csv')
if MASKALL: OUT = OUT.replace('.csv', '_maskall.csv')
print('last observation (hour start):', last_obs, '-> cutoff', C)

origins = list(pd.date_range('2025-03-10', C, freq='12h'))
maxg = int((pd.Timestamp('2026-11-13') - C).total_seconds() // 3600)
GAPS = list(range(1, 97)) + (list(range(97, maxg + 1, 3)) if maxg > 96 else [])
print('max horizon needed (h):', maxg)
d = build(P, E, origins, GAPS).dropna(subset=['y'])
for X in (d,):
    X['base'] = X.l168.fillna(X.l336).fillna(X.l672); X['cb'] = X.cams24_adj.fillna(X.base)
w = 0.5 ** ((d.t.max() - d.t).dt.days / HL) if HL else None
models = []
for mode, prm in ((('log', P0), ('log', P1)) if NOCAMS else (('log', P0), ('log', P1), ('resc', P1))):
    yt = np.log1p(d.y) if mode == 'log' else np.log1p(d.y) - np.log1p(d.cb)
    for s in SEEDS:
        models.append((mode, lgb.LGBMRegressor(random_state=s, **prm).fit(d[F], yt, sample_weight=w)))
print('trained on', len(d), 'rows')

te = pd.read_csv('Test.csv'); sm = pd.read_csv('SampleSubmission.csv')
te['t'] = pd.to_datetime(te.target_start_ist).dt.tz_localize(None)

def forecast(C, ts):
    gaps = sorted({int((t - C).total_seconds() // 3600) for t in ts})
    f = build(P, E, [C], gaps, need_y=False)
    f['base'] = f.l168.fillna(f.l336).fillna(f.l672); f['cb'] = f.cams24_adj.fillna(f.base)
    ps = {'log': [], 'resc': []}
    for mode, m in models:
        p = m.predict(f[F]); ps[mode].append(np.expm1(p) if mode == 'log' else np.expm1(p + np.log1p(f.cb)))
    f['p'] = np.clip(np.mean(ps['log'], 0) if NOCAMS else (2 * np.mean(ps['log'], 0) + np.mean(ps['resc'], 0)) / 3, 0, None)
    return f.set_index(['location_id', 't'])

fut = te[te.t >= C]
pf = forecast(C, fut.t.unique())
te['p'] = [pf.p.get((l, t), np.nan) for l, t in zip(te.location_id, te.t)]
# past rows: observation if known, else a 1-day-ahead forecast from the previous midnight
for i in te.index[te.t < C]:
    l, t = te.at[i, 'location_id'], te.at[i, 't']
    o = P_true.at[t, l] if t in P_true.index else np.nan
    if not np.isnan(o): te.at[i, 'p'] = o
miss = te[te.p.isna()].copy(); miss['c0'] = miss.t.dt.normalize() - pd.Timedelta(days=1)
for c0, g in miss.groupby('c0'):
    f = forecast(c0, g.t.unique())
    for i in g.index: te.at[i, 'p'] = f.p.get((te.at[i, 'location_id'], te.at[i, 't']), np.nan)
te['p'] = te.p.fillna(te.groupby('location_id').p.transform('median'))

sub = sm[['ID']].merge(te[['ID', 'p']], on='ID', how='left').rename(columns={'p': 'Target'})
assert list(sub.ID) == list(sm.ID) and len(sub) == 2592 and np.isfinite(sub.Target).all() and (sub.Target >= 0).all()
sub.to_csv(OUT, index=False); print('wrote', OUT)
# archive the exact inputs used (forecasts change every few hours) for code review / reproducibility
import shutil, os
snap = os.path.join('snapshots', pd.Timestamp.now().strftime('%Y%m%d_%H%M') + '_' + OUT.replace('.csv', ''))
os.makedirs(snap, exist_ok=True)
for fpath in ('ext/weather.csv', 'ext/aq.csv', 'Train.csv', OUT): shutil.copy(fpath, snap)
print('archived inputs to', snap)
pv = te[(te.t >= '2026-11-11') & (te.t < '2026-11-13')]
print('PRIVATE WINDOW 11-12 Nov forecast (daily mean by station):')
print(pv.groupby([pv.t.dt.date, 'location_id']).p.mean().unstack().round(1).to_string())
print(pv.groupby(pv.t.dt.hour).p.mean().round(0).to_string())
