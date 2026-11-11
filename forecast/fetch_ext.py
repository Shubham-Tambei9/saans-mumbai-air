"""Fetch Open-Meteo weather (archived forecasts + live forecast) and CAMS air-quality for the 4 stations.
Writes ext/weather.csv and ext/aq.csv (hourly, IST, keyed by location_id and hour START)."""
import pandas as pd, requests, os, time
os.makedirs('ext', exist_ok=True)
st = pd.read_csv('Stations.csv')
W = 'temperature_2m,relative_humidity_2m,dew_point_2m,precipitation,wind_speed_10m,wind_direction_10m,wind_gusts_10m,boundary_layer_height,surface_pressure,cloud_cover,shortwave_radiation'
A = 'pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth'
START, END = '2025-02-01', (pd.Timestamp.now() + pd.Timedelta(days=6)).strftime('%Y-%m-%d')

def get(url, params):
    for k in range(5):
        r = requests.get(url, params=params, timeout=120)
        if r.ok: return r.json()
        print('retry', r.status_code, r.text[:200]); time.sleep(5 * (k + 1))
    r.raise_for_status()

def frame(js, loc):
    d = pd.DataFrame(js['hourly']); d['time'] = pd.to_datetime(d['time']); d['location_id'] = loc; return d

wx, aq = [], []
today = pd.Timestamp.now().normalize()
for _, s in st.iterrows():
    base = dict(latitude=s.latitude, longitude=s.longitude, timezone='Asia/Kolkata')
    # archived forecasts (consistent with what a live forecast looks like)
    h = frame(get('https://historical-forecast-api.open-meteo.com/v1/forecast',
                  {**base, 'hourly': W, 'start_date': START, 'end_date': (today - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}), s.location_id)
    f = frame(get('https://api.open-meteo.com/v1/forecast',
                  {**base, 'hourly': W, 'past_days': 3, 'forecast_days': 7}), s.location_id)
    w = pd.concat([h, f]).drop_duplicates(['location_id', 'time'], keep='last'); wx.append(w)
    a = frame(get('https://air-quality-api.open-meteo.com/v1/air-quality',
                  {**base, 'hourly': A, 'start_date': START, 'end_date': END}), s.location_id)
    aq.append(a)
    print(s.location_id, len(w), w.time.min(), w.time.max(), '| aq', len(a), a.dropna(subset=['pm2_5']).time.max())
pd.concat(wx).to_csv('ext/weather.csv', index=False)
pd.concat(aq).to_csv('ext/aq.csv', index=False)
