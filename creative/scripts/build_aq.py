"""Bundle everything the Saans web app needs into creative/site/data.js (window.SAANS = {...}).
Inputs (project root): Train.csv (CPCB/OpenAQ hourly PM2.5, hour-END timestamps), Stations.csv, Test.csv,
submission.csv (our LightGBM forecast, final2.py), creative/data/*.csv|geojson (ClimateVerse layers)."""
import os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(H, '..', '..'); D = os.path.join(H, '..', 'data')
SITE = os.environ.get('SAANS_SITE', os.path.join(H, '..', 'site')); os.makedirs(SITE, exist_ok=True)
r1 = lambda x: None if pd.isna(x) else round(float(x), 1)

st = pd.read_csv(os.path.join(ROOT, 'Stations.csv'))
SHORT = {11611: 'Malad West', 3409511: 'Shivaji Nagar', 11606: 'Borivali East', 3409510: 'Byculla'}
stations = [dict(id=int(r.location_id), name=r.station_name, short=SHORT[int(r.location_id)], lat=r.latitude, lon=r.longitude)
            for r in st.itertuples()]
ids = [s['id'] for s in stations]

tr = pd.read_csv(os.path.join(ROOT, 'Train.csv')); tr = tr[~tr.flag_high_1000.astype(bool)]   # drop flagged sensor errors (>1000)
tr['t'] = pd.to_datetime(tr.timestamp_ist).dt.tz_localize(None) - pd.Timedelta(hours=1)      # hour START
P = tr.pivot_table(index='t', columns='location_id', values='pm25')
P = P.reindex(pd.date_range(P.index.min(), P.index.max(), freq='h'))[ids]
last = P.dropna(how='all').index[-1]; C = last + pd.Timedelta(hours=1)

te = pd.read_csv(os.path.join(ROOT, 'Test.csv')); te['t'] = pd.to_datetime(te.target_start_ist).dt.tz_localize(None)
sub = pd.read_csv(os.path.join(ROOT, 'submission.csv')); te = te.merge(sub, on='ID')
fc = te[(te.t >= C) & (te.t < '2026-11-13')].pivot_table(index='t', columns='location_id', values='Target')[ids]
hours = [t.strftime('%Y-%m-%dT%H:00') for t in fc.index]
forecast = {str(l): [r1(v) for v in fc[l]] for l in ids}

rec = P.loc[C - pd.Timedelta(days=14):]
recent = dict(hours=[t.strftime('%Y-%m-%dT%H:00') for t in rec.index], values={str(l): [r1(v) for v in rec[l]] for l in ids})

# Diwali: main firework night 20 Oct 2025 vs 8 Nov 2026 (Lakshmi Puja evenings). Hourly city mean, -3 .. +4 days.
def window(day0, before=3, after=4):
    a = pd.Timestamp(day0) - pd.Timedelta(days=before); b = pd.Timestamp(day0) + pd.Timedelta(days=after)
    w = P.loc[a:b - pd.Timedelta(hours=1)]
    city = w.mean(axis=1).rolling(3, center=True, min_periods=1).mean()     # 3-h smoothing (stations drop in and out)
    return dict(start=a.strftime('%Y-%m-%dT%H:00'), city=[r1(v) for v in city],
                daily=[r1(v) for v in w.mean(axis=1).resample('D').mean().reindex(pd.date_range(a, b - pd.Timedelta(days=1)))])
diwali = {'2025': dict(night='2025-10-20', **window('2025-10-20')), '2026': dict(night='2026-11-08', **window('2026-11-08'))}

# typical daily cycle in Oct-Jan (pollution season), by station
season = P[(P.index.month.isin([10, 11, 12, 1]))]
cycle = {str(l): [r1(v) for v in season[l].groupby(season.index.hour).mean()] for l in ids}
monthly = P.mean(axis=1).resample('MS').mean()
monthly = dict(months=[t.strftime('%Y-%m') for t in monthly.index], city=[r1(v) for v in monthly])

pop = pd.read_csv(os.path.join(D, 'pop_grid.csv')) if os.path.exists(os.path.join(D, 'pop_grid.csv')) else None
if pop is not None:   # tag each 1 km population cell with the pincode it falls in ('' = outside the BMC pincode map)
    from shapely.geometry import shape, Point
    from shapely.strtree import STRtree
    pc = json.load(open(os.path.join(D, 'pincodes.geojson')))['features']
    geoms = [shape(f['geometry']) for f in pc]; tree = STRtree(geoms)
    def pin_of(lat, lon):
        pt = Point(lon, lat)
        for i in tree.query(pt):
            if geoms[i].contains(pt): return pc[i]['properties']['pincode']
        return ''
    pop['pin'] = [pin_of(a, b) for a, b in zip(pop.lat, pop.lon)]
    pop = pop[pop.pin != '']     # keep cells inside Greater Mumbai's pincode map (drops sea, Thane, Navi Mumbai)
    print('pop cells in Mumbai pincodes', len(pop), '| people', int(pop['pop'].sum()), '| kids<5', int(pop.kids_u5.sum()),
          '| 65+', int(pop.elderly_65p.sum()))
parks = pd.read_csv(os.path.join(D, 'parks.csv')); bus = pd.read_csv(os.path.join(D, 'bus_stops.csv'))
clean = lambda s: ' '.join(str(s).split()).strip(' "') if pd.notna(s) else ''
# weather forecast at the forecast hours (Open-Meteo): wind and boundary-layer (mixing) height, for the 3D city view
wx = pd.read_csv(os.path.join(ROOT, 'ext', 'weather.csv'), parse_dates=['time'])
wx = wx[wx.time.isin(fc.index)]
def wcol(c, nd=1):
    p = wx.pivot_table(index='time', columns='location_id', values=c).reindex(fc.index)[ids].interpolate(limit_direction='both')
    return {str(l): [None if pd.isna(v) else round(float(v), nd) for v in p[l]] for l in ids}
met = dict(ws=wcol('wind_speed_10m'), wd=wcol('wind_direction_10m', 0), blh=wcol('boundary_layer_height', 0), temp=wcol('temperature_2m'))
out = dict(
    met=met,
    generated=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST'), cutoff=C.strftime('%Y-%m-%dT%H:00'),
    stations=stations, forecast=dict(hours=hours, values=forecast, low=0.85, high=1.10), recent=recent,
    diwali=diwali, cycle=cycle, monthly=monthly,
    backtest=dict(folds=['Nov 2025', 'Feb 2026', 'Apr 2026', 'Sep 2026', 'Oct 2026'],
                  persistence=[30.94, 22.35, 17.65, 10.55, 15.68], cams=[30.25, 17.58, 15.93, 10.02, 13.05],
                  model=[29.97, 15.62, 15.79, 8.07, 11.11]),
    pop=None if pop is None else dict(lat=pop.lat.tolist(), lon=pop.lon.tolist(), pop=pop['pop'].astype(int).tolist(),
                                       kids=pop.kids_u5.astype(int).tolist(), eld=pop.elderly_65p.astype(int).tolist(),
                                       pin=pop.pin.tolist()),
    parks=[[clean(r.name), r.type, clean(r.ward), r.lat, r.lon] for r in parks.itertuples()],
    bus=dict(lat=bus.lat.tolist(), lon=bus.lon.tolist(), name=[clean(x) for x in bus.name], kind=bus.kind.tolist()),
    pincodes=json.load(open(os.path.join(D, 'pincodes.geojson'))),
    landfills=json.load(open(os.path.join(D, 'landfills.geojson'))) if os.path.exists(os.path.join(D, 'landfills.geojson')) else None,
)
js = 'window.SAANS = ' + json.dumps(out, ensure_ascii=False, separators=(',', ':')) + ';\n'
open(os.path.join(SITE, 'data.js'), 'w', encoding='utf-8').write(js)
print('cutoff', C, '| forecast hours', len(hours), '| recent', len(recent['hours']), '| pop cells', 0 if pop is None else len(pop),
      '| data.js KB', len(js.encode()) // 1024)
print('diwali daily 2025', diwali['2025']['daily']); print('diwali daily 2026', diwali['2026']['daily'])
