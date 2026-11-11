"""Replay Diwali 2025 as an analog of the private window (11-12 Nov 2026).
2026: main firework night 8->9 Nov, public window 7-8 Nov hidden from Train, final Train ends ~10 Nov 06:00,
      private targets 11-12 Nov.
2025 (shift -384 days = 1 year + 19 days): main night 20->21 Oct, hide 19-20 Oct, cutoff 22 Oct 07:00, targets 23-24 Oct.
Same model as final2.py (2 log LGBMs + 1 CAMS-residual LGBM, HL=180). Only observations before the cutoff are used."""
import sys, pandas as pd, numpy as np, lightgbm as lgb, warnings
from pipe import *
warnings.filterwarnings('ignore')
SH = pd.Timedelta(days=int(sys.argv[1]) if len(sys.argv) > 1 else 384)   # 2026-11-08 -> 2025-10-20 (main firework night)
a = lambda s: pd.Timestamp(s) - SH          # 2026 time -> 2025 analog time
SEEDS = (0, 1, 2)
P0 = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.7, reg_lambda=5, verbose=-1)
P1 = dict(P0, num_leaves=15, min_child_samples=300, n_estimators=800, colsample_bytree=0.5)
F = [f for f in FEATS if f not in ('doy', 'dwl')]
CAMSF = ('pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'dust', 'aerosol_optical_depth', 'cams24', 'cams6',
         'bias7', 'cams_adj', 'cams24_adj', 'cams_anom')
FN = [f for f in F if f not in CAMSF]
HL = 180

Pall = load_obs(); E = load_ext()
C = a('2026-11-10 07:00'); T0, T1 = a('2026-11-11'), a('2026-11-13')
truth = Pall.loc[T0:T1 - pd.Timedelta(hours=1)]
print('analog cutoff', C, '| targets', T0, '..', T1, '| truth daily means:')
print(truth.resample('D').mean().round(1).to_string())
gap_hide = (a('2026-11-07 00:00'), a('2026-11-08 23:00'))          # public-window gap (not in Train)
VARIANTS = {   # name: (extra hidden spans, use CAMS)
    'A_mask (=submission.csv)':      ([(a('2026-11-08 12:00'), a('2026-11-09 12:00'))], True),
    'A2_mask_all_nights':            ([(a('2026-11-08 12:00'), C - pd.Timedelta(hours=1))], True),
    'B_keep (=keepdiwali)':          ([], True),
    'C_mask_nocams (=nocams)':       ([(a('2026-11-08 12:00'), a('2026-11-09 12:00'))], False),
    'D_mask_nights_keep_days':       ([(a('2026-11-08 12:00'), a('2026-11-09 08:00')),
                                       (a('2026-11-09 18:00'), C - pd.Timedelta(hours=1))], True),
}
origins = list(pd.date_range('2025-03-10', C, freq='12h'))
gaps_f = sorted({int((t - C).total_seconds() // 3600) for t in truth.index})
def rmse(x, y): m = ~(np.isnan(x) | np.isnan(y)); return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))
res, daily = {}, {}
for name, (spans, cams) in VARIANTS.items():
    P = Pall.loc[:C - pd.Timedelta(hours=1)].copy()
    for s, e in [gap_hide] + spans: P.loc[s:e] = np.nan
    d = build(P, E, origins, list(range(1, 97))).dropna(subset=['y'])
    d = d[d.t < C]
    d['base'] = d.l168.fillna(d.l336).fillna(d.l672); d['cb'] = d.cams24_adj.fillna(d.base)
    f = build(P, E, [C], gaps_f, need_y=False)
    f['base'] = f.l168.fillna(f.l336).fillna(f.l672); f['cb'] = f.cams24_adj.fillna(f.base)
    w = 0.5 ** ((d.t.max() - d.t).dt.days / HL)
    Fv = F if cams else FN
    ps = {'log': [], 'resc': []}
    for mode, prm in ((('log', P0), ('log', P1), ('resc', P1)) if cams else (('log', P0), ('log', P1))):
        yt = np.log1p(d.y) if mode == 'log' else np.log1p(d.y) - np.log1p(d.cb)
        for s in SEEDS:
            p = lgb.LGBMRegressor(random_state=s, **prm).fit(d[Fv], yt, sample_weight=w).predict(f[Fv])
            ps[mode].append(np.expm1(p) if mode == 'log' else np.expm1(p + np.log1p(f.cb)))
    f['p'] = np.clip(np.mean(ps['log'], 0) if not cams else (2 * np.mean(ps['log'], 0) + np.mean(ps['resc'], 0)) / 3, 0, None)
    pred = f.pivot_table(index='t', columns='location_id', values='p').reindex(truth.index)[truth.columns]
    res[name] = rmse(pred.values.ravel(), truth.values.ravel())
    daily[name] = pred.resample('D').mean().round(1)
    print(f'\n{name}: RMSE {res[name]:.2f}  (train rows {len(d)})'); print(daily[name].to_string(), flush=True)
    if name.startswith('A_mask'):
        base = f.pivot_table(index='t', columns='location_id', values='base').reindex(truth.index)[truth.columns]
        cb = f.pivot_table(index='t', columns='location_id', values='cb').reindex(truth.index)[truth.columns]
        res['ref: last-week mean (masked)'] = rmse(base.values.ravel(), truth.values.ravel())
        res['ref: CAMS bias-adjusted'] = rmse(cb.values.ravel(), truth.values.ravel())
print('\nRMSE on analog of 11-12 Nov:'); print(pd.Series(res).round(2).to_string())
