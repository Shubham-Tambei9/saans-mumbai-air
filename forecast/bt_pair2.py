"""Which PAIR of submissions to select? Private score = the better of the 2 selected submissions' RMSE on one
2-day window (11-12 Nov). Backtest: for every midnight cutoff in each fold, score the next 2 calendar days
(gap 24..71 h) for many candidate forecasts, then rank pairs by mean over windows of min(RMSE_a, RMSE_b)."""
import pandas as pd, numpy as np, lightgbm as lgb, warnings, itertools
from pipe import FEATS
warnings.filterwarnings('ignore')
d = pd.read_pickle('ext/ds.pkl').dropna(subset=['y'])
d['base'] = d.l168.fillna(d.l336).fillna(d.l672); d['cb'] = d.cams24_adj.fillna(d.base)
FOLDS = [('2025-11-20', '2026-01-15'), ('2026-02-01', '2026-04-01'), ('2026-04-01', '2026-06-01'),
         ('2026-09-01', '2026-10-01'), ('2026-10-01', '2026-11-03')]
F = [f for f in FEATS if f not in ('doy', 'dwl')]
CAMSF = ('pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'dust', 'aerosol_optical_depth', 'cams24', 'cams6',
         'bias7', 'cams_adj', 'cams24_adj', 'cams_anom')
FN = [f for f in F if f not in CAMSF]
P0 = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, reg_lambda=5, verbose=-1)
P1 = dict(P0, num_leaves=15, min_child_samples=300, n_estimators=800, colsample_bytree=0.5)
SEEDS = (0, 1, 2); HL = 180

def fit_pred(trn, ev, Fv, mode, prm):
    yt = np.log1p(trn.y) if mode == 'log' else np.log1p(trn.y) - np.log1p(trn.cb)
    w = 0.5 ** ((trn.t.max() - trn.t).dt.days / HL); out = []
    for s in SEEDS:
        p = lgb.LGBMRegressor(random_state=s, **prm).fit(trn[Fv], yt, sample_weight=w).predict(ev[Fv])
        out.append(np.expm1(p) if mode == 'log' else np.expm1(p + np.log1p(ev.cb)))
    return np.mean(out, 0)

evs = []
for a, b in FOLDS:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    trn = d[d.t < a]
    ev = d[(d.C >= a) & (d.C < b) & (((d.C.dt.hour == 0) & d.gap.between(24, 71)) | ((d.C.dt.hour == 12) & d.gap.between(12, 59)))].copy()
    lg = (fit_pred(trn, ev, F, 'log', P0) + fit_pred(trn, ev, F, 'log', P1)) / 2
    ev['main'] = np.clip((2 * lg + fit_pred(trn, ev, F, 'resc', P1)) / 3, 0, None)
    ev['nocams'] = np.clip((fit_pred(trn, ev, FN, 'log', P0) + fit_pred(trn, ev, FN, 'log', P1)) / 2, 0, None)
    ev['fold'] = str(a.date()); evs.append(ev); print('fold', a.date(), 'windows', ev.C.nunique(), flush=True)
ev = pd.concat(evs)

cand = {f'main*{k:.2f}': ev.main * k for k in (0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4)}
cand.update({f'main{s:+d}': (ev.main + s).clip(lower=0) for s in (-10, -5, 5, 10, 15)})
cand.update({'nocams': ev.nocams, 'cams_adj': ev.cb, 'last_week': ev.base,
             'main+nocams/2': (ev.main + ev.nocams) / 2, 'main+cams_adj/2': (ev.main + ev.cb) / 2})
se = pd.DataFrame({k: (v - ev.y) ** 2 for k, v in cand.items()})
se['C'] = ev.C.values; se['fold'] = ev.fold.values
W = np.sqrt(se.groupby(['fold', 'C']).mean())          # RMSE per 2-day window, per candidate
print('\nwindows:', len(W))
print('\nSINGLE submission, mean window RMSE:'); print(W.mean().sort_values().round(2).head(12).to_string())
pairs = []
for x, y in itertools.combinations(W.columns, 2):
    m = np.minimum(W[x], W[y]); pairs.append((x, y, m.mean(), m.groupby(level=0).mean().round(2).to_dict()))
pr = pd.DataFrame(pairs, columns=['a', 'b', 'best_of_two', 'by_fold']).sort_values('best_of_two')
print('\nBEST PAIRS (mean over windows of the better RMSE):')
print(pr.head(15).to_string(index=False))
for x in ('main*1.00',):
    print(f'\npairs that include {x}:'); print(pr[(pr.a == x) | (pr.b == x)].head(10)[['a', 'b', 'best_of_two']].to_string(index=False))
print('\nhow often truth is above main (window mean y / mean main):')
g = ev.groupby(['fold', 'C']).agg(y=('y', 'mean'), p=('main', 'mean')); print((g.y / g.p).describe().round(3).to_string())

# ---- robustness: per-fold numbers for the shortlisted pairs, both cutoff hours, saved for later
for k in (0.85, 0.9, 1.0, 1.1):
    cand[f'blend*{k:.2f}'] = (ev.main + ev.cb) / 2 * k
se = pd.DataFrame({k: (v - ev.y) ** 2 for k, v in cand.items()}); se['C'] = ev.C.values; se['fold'] = ev.fold.values
W = np.sqrt(se.groupby(['fold', 'C']).mean()); W.to_pickle('ext/bt_pair_W.pkl')
SHORT = [('main*1.00', 'main*1.00'), ('main*0.85', 'main*1.10'), ('main*0.85', 'main+5'), ('main*0.80', 'main*1.05'),
         ('main*0.85', 'main*1.05'), ('main*0.90', 'main*1.10'), ('main*0.85', 'main*1.00'), ('main*0.90', 'main+5'),
         ('blend*0.85', 'blend*1.10'), ('blend*0.85', 'main*1.10'), ('main*0.85', 'blend*1.10')]
rows = {}
for x, y in SHORT:
    m = np.minimum(W[x], W[y]); r = m.groupby(level=0).mean(); r['ALL'] = m.mean()
    r['win%_vs_main'] = 100 * (m < W['main*1.00'] - 1e-9).mean(); rows[f'{x} | {y}'] = r
print('\nPER-FOLD best-of-two (both cutoff hours, 3 seeds):'); print(pd.DataFrame(rows).T.round(2).to_string())
for h in (0, 12):
    Wh = W[W.index.get_level_values(1).hour == h]
    print(f'\ncutoff hour {h}: windows {len(Wh)} | main {Wh["main*1.00"].mean():.2f} | 0.85+1.10 {np.minimum(Wh["main*0.85"], Wh["main*1.10"]).mean():.2f}'
          f' | 0.85+(+5) {np.minimum(Wh["main*0.85"], Wh["main+5"]).mean():.2f}')
