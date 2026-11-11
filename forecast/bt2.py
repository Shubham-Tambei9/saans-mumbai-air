"""Rolling-origin backtest at the real horizon (targets 24-72h after cutoff)."""
import pandas as pd, numpy as np, lightgbm as lgb, sys, os
from pipe import *

P = load_obs(); E = load_ext()
cache = 'ext/ds.pkl'
if os.path.exists(cache) and '--rebuild' not in sys.argv:
    d = pd.read_pickle(cache)
else:
    origins = [o for o in pd.date_range('2025-03-10', P.index[-1], freq='12h')]
    d = build(P, E, origins, range(1, 73)); d.to_pickle(cache)
d = d.dropna(subset=['y'])
print('rows', len(d))

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

FOLDS = [('2025-10-01', '2025-11-20'), ('2025-11-20', '2026-01-15'), ('2026-09-01', '2026-10-01'), ('2026-10-01', '2026-11-03')]
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.7, reg_lambda=5, verbose=-1)

def fit_pred(trn, ev, F, obj='regression', log=False, seed=0):
    y = np.log1p(trn.y) if log else trn.y
    m = lgb.LGBMRegressor(objective=obj, random_state=seed, **PARAMS).fit(trn[F], y)
    p = m.predict(ev[F]); p = np.expm1(p) if log else p
    return np.clip(p, 0, None), m


NOCAL=[f for f in FEATS if f not in ('doy',)]
NODW=[f for f in NOCAL if f!='dwl']
OBS=['loc','gap','hour','dow','last','age','l6','l24','l72','l168','l336','l672','prof','prof7','sd168','a24','a168']
res=[]; imp=None
for a, b in FOLDS:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    trn = d[d.t < a].copy()
    ev = d[(d.C >= a) & (d.C < b) & (d.gap >= 24) & (d.gap <= 72) & (d.C.dt.hour == 0)].copy()
    for X in (trn, ev):
        X['base'] = X.l168.fillna(X.l336).fillna(X.l672)
        X['cb'] = X.cams24_adj.fillna(X.base)
    r = {'fold': a.date(), 'n': len(ev)}
    r['l168'] = rmse(ev.y, ev.base); r['prof'] = rmse(ev.y, ev.prof.fillna(ev.base)); r['cams24_adj'] = rmse(ev.y, ev.cb)
    r['l72'] = rmse(ev.y, ev.l72.fillna(ev.base))
    P_ = {}
    for name, F, mode in [('obs_log', OBS, 'log'), ('nocal_log', NOCAL, 'log'), ('nodw_log', NODW, 'log'),
                          ('nocal_raw', NOCAL, 'raw'), ('nocal_res', NOCAL, 'res'), ('nocal_resc', NOCAL, 'resc')]:
        if mode == 'log': yt = np.log1p(trn.y)
        elif mode == 'raw': yt = trn.y
        elif mode == 'res': yt = np.log1p(trn.y) - np.log1p(trn.base)
        else: yt = np.log1p(trn.y) - np.log1p(trn.cb)
        m = lgb.LGBMRegressor(random_state=0, **PARAMS).fit(trn[F], yt); p = m.predict(ev[F])
        if mode == 'log': p = np.expm1(p)
        elif mode == 'res': p = np.expm1(p + np.log1p(ev.base))
        elif mode == 'resc': p = np.expm1(p + np.log1p(ev.cb))
        p = np.clip(p, 0, None); P_[name] = p; r[name] = rmse(ev.y, p)
        if name == 'nocal_log':
            fi = pd.Series(m.booster_.feature_importance('gain'), F); imp = fi if imp is None else imp + fi
    r['ens'] = rmse(ev.y, (P_['nocal_log'] + P_['nocal_res'] + P_['nocal_resc']) / 3)
    res.append(r)
R = pd.DataFrame(res).set_index('fold'); print(R.round(2).to_string()); print(R.drop(columns='n').mean().round(2).to_string())
print((imp / imp.sum()).sort_values(ascending=False).head(20).round(3).to_string())
