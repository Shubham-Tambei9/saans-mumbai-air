"""Shared feature pipeline. Rows = (station, cutoff C, target hour-start t). Only obs with start < C are used."""
import pandas as pd, numpy as np

CAP = 500.0
WX = ['temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'wind_speed_10m', 'wind_gusts_10m',
      'boundary_layer_height', 'surface_pressure', 'cloud_cover', 'shortwave_radiation']
AQV = ['pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'dust', 'aerosol_optical_depth']
# Lakshmi Puja dates (Diwali main night)
DIWALI = pd.to_datetime(['2018-11-07', '2019-10-27', '2020-11-14', '2021-11-04', '2022-10-24', '2023-11-12',
                         '2024-11-01', '2025-10-21', '2026-11-08'])


def load_obs(path='Train.csv'):
    tr = pd.read_csv(path)
    tr['t'] = pd.to_datetime(tr.timestamp_ist).dt.tz_localize(None) - pd.Timedelta(hours=1)  # hour START
    tr.loc[tr.pm25 > CAP, 'pm25'] = np.nan
    P = tr.pivot_table(index='t', columns='location_id', values='pm25')
    P = P.reindex(pd.date_range(P.index.min().floor('D'), P.index.max(), freq='h'))
    return P


def load_ext():
    w = pd.read_csv('ext/weather.csv', parse_dates=['time'])
    a = pd.read_csv('ext/aq.csv', parse_dates=['time'])
    m = w.merge(a, on=['location_id', 'time'], how='outer').sort_values(['location_id', 'time'])
    rad = np.deg2rad(m.wind_direction_10m)
    m['u'] = -m.wind_speed_10m * np.sin(rad); m['v'] = -m.wind_speed_10m * np.cos(rad)
    out = []
    for loc, g in m.groupby('location_id'):
        g = g.set_index('time').asfreq('h'); g['location_id'] = loc
        g['rain24'] = g.precipitation.rolling(24, min_periods=1).sum()
        g['rain72'] = g.precipitation.rolling(72, min_periods=1).sum()
        g['ws12'] = g.wind_speed_10m.rolling(12, min_periods=1).mean()
        g['u24'] = g.u.rolling(24, min_periods=1).mean(); g['v24'] = g.v.rolling(24, min_periods=1).mean()
        g['blh6'] = g.boundary_layer_height.rolling(6, min_periods=1).mean()
        g['blh24'] = g.boundary_layer_height.rolling(24, min_periods=1, center=True).mean()
        g['cams24'] = g.pm2_5.rolling(24, min_periods=1, center=True).mean()
        g['cams6'] = g.pm2_5.rolling(6, min_periods=1, center=True).mean()
        out.append(g.reset_index())
    return pd.concat(out).rename(columns={'time': 't'})


def diwali_day(t):
    t = pd.DatetimeIndex(t).normalize()
    d = np.full(len(t), 99.0)
    for D in DIWALI:
        dd = (t - D).days.values.astype(float)
        d = np.where(np.abs(dd) < np.abs(d), dd, d)
    return np.clip(d, -15, 15)


def station_stats(P, C):
    """Per-station stats from obs with start < C."""
    h = P.loc[:C - pd.Timedelta(hours=1)].iloc[-24 * 28:]
    r = {}
    for l in P.columns:
        s = h[l]
        nn = s.dropna()
        r[l] = dict(last=nn.iloc[-1] if len(nn) else np.nan, age=(C - nn.index[-1]).total_seconds() / 3600 if len(nn) else 999,
                    l6=s.iloc[-6:].mean(), l24=s.iloc[-24:].mean(), l72=s.iloc[-72:].mean(), l168=s.iloc[-168:].mean(),
                    l336=s.iloc[-336:].mean(), l672=s.mean(),
                    prof=s.iloc[-336:].groupby(s.iloc[-336:].index.hour).mean(),
                    prof7=s.iloc[-168:].groupby(s.iloc[-168:].index.hour).mean(),
                    sd168=s.iloc[-168:].std())
    return r


def build(P, E, origins, gaps, need_y=True):
    Ei = E.set_index(['location_id', 't'])
    rows = []
    for C in origins:
        S = station_stats(P, C)
        a24 = np.nanmean([S[l]['l24'] for l in P.columns]); a168 = np.nanmean([S[l]['l168'] for l in P.columns])
        for l in P.columns:
            r = S[l]
            if np.isnan(r['l168']): continue
            for g in gaps:
                t = C + pd.Timedelta(hours=int(g))
                rows.append((l, C, t, g, r['last'], r['age'], r['l6'], r['l24'], r['l72'], r['l168'], r['l336'], r['l672'],
                             r['prof'].get(t.hour, np.nan), r['prof7'].get(t.hour, np.nan), r['sd168'], a24, a168))
    d = pd.DataFrame(rows, columns=['location_id', 'C', 't', 'gap', 'last', 'age', 'l6', 'l24', 'l72', 'l168', 'l336', 'l672',
                                    'prof', 'prof7', 'sd168', 'a24', 'a168'])
    # external features at target time
    d = d.join(Ei, on=['location_id', 't'])
    # external context at cutoff time (mean of 72h before C) -> anomalies
    ctx = E.copy()
    for c in ['pm2_5', 'boundary_layer_height', 'wind_speed_10m']:
        ctx[c + '_p72'] = ctx.groupby('location_id')[c].transform(lambda s: s.rolling(72, min_periods=12).mean())
    ctx = ctx.set_index(['location_id', 't'])[['pm2_5_p72', 'boundary_layer_height_p72', 'wind_speed_10m_p72']]
    d['Cm1'] = d.C - pd.Timedelta(hours=1)
    d = d.join(ctx, on=['location_id', 'Cm1'])
    # obs/CAMS bias over the last 7 days before C
    camsP = E.pivot_table(index='t', columns='location_id', values='pm2_5').reindex(P.index)
    ratio = {}
    for C in origins:
        o = P.loc[C - pd.Timedelta(hours=168):C - pd.Timedelta(hours=1)]; c = camsP.loc[o.index]
        for l in P.columns:
            m = o[l].notna() & c[l].notna()
            ratio[(l, C)] = o[l][m].mean() / max(c[l][m].mean(), 1) if m.sum() > 24 else np.nan
    d['bias7'] = [ratio[(l, C)] for l, C in zip(d.location_id, d.C)]
    d['cams_adj'] = d.pm2_5 * d.bias7
    d['cams24_adj'] = d.cams24 * d.bias7
    d['cams_anom'] = d.cams24 / d.pm2_5_p72
    d['blh_anom'] = d.blh24 / d.boundary_layer_height_p72
    d['ws_anom'] = d.ws12 - d.wind_speed_10m_p72
    d['hour'] = d.t.dt.hour; d['dow'] = d.t.dt.dayofweek; d['doy'] = d.t.dt.dayofyear
    d['dwl'] = diwali_day(d.t); d['dwlC'] = diwali_day(d.C)
    d['loc'] = pd.Categorical(d.location_id, categories=sorted(P.columns))
    if need_y:
        Pl = P.stack().rename('y').reset_index().rename(columns={'level_0': 't', 'level_1': 'location_id'})
        Pl.columns = ['t', 'location_id', 'y']
        d = d.merge(Pl, on=['t', 'location_id'], how='left')
    return d.drop(columns=['Cm1'])


FEATS = ['loc', 'gap', 'hour', 'dow', 'doy', 'last', 'age', 'l6', 'l24', 'l72', 'l168', 'l336', 'l672', 'prof', 'prof7', 'sd168',
         'a24', 'a168', 'dwl',
         'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'precipitation', 'wind_speed_10m', 'wind_gusts_10m',
         'boundary_layer_height', 'surface_pressure', 'cloud_cover', 'shortwave_radiation', 'u', 'v', 'u24', 'v24',
         'rain24', 'rain72', 'ws12', 'blh6', 'blh24',
         'pm2_5', 'pm10', 'carbon_monoxide', 'nitrogen_dioxide', 'dust', 'aerosol_optical_depth', 'cams24', 'cams6',
         'bias7', 'cams_adj', 'cams24_adj', 'cams_anom', 'blh_anom', 'ws_anom']
