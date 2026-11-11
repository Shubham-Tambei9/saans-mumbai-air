"""WorldPop Global2 R2025A 2026 age structures (ClimateVerse doi:10.71646/KHOTYE) -> Mumbai 1 km grid.
Source files (1 km, both sexes, one per 5-year age group) downloaded to creative/raw/worldpop/ from
https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/R2025A/2026/IND/v1/1km_ua/constrained/
Output: creative/data/pop_grid.csv  lat,lon,pop,kids_u5,elderly_65p"""
import os, numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raw', 'worldpop') + os.sep
W, S, E, N = 72.77, 18.89, 73.00, 19.28
AGES = ['00', '01', '05', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90']
layers = {}
for a in AGES:
    with rasterio.open(BASE + f'ind_t_{a}_2026_CN_1km_R2025A_UA_v1.tif') as src:
        win = from_bounds(W, S, E, N, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win).astype('float64'); nod = src.nodata
        if nod is not None: arr[arr == nod] = 0
        arr[~np.isfinite(arr) | (arr < 0)] = 0
        layers[a] = arr; tr = src.window_transform(win)
        print(a, arr.shape, round(arr.sum()), flush=True)
pop = sum(layers.values()); kids = layers['00'] + layers['01']
eld = sum(layers[a] for a in ['65', '70', '75', '80', '85', '90'])
rows, cols = np.nonzero(pop > 0)
xs, ys = rasterio.transform.xy(tr, rows, cols)
df = pd.DataFrame({'lat': np.round(ys, 5), 'lon': np.round(xs, 5), 'pop': np.round(pop[rows, cols]),
                   'kids_u5': np.round(kids[rows, cols]), 'elderly_65p': np.round(eld[rows, cols])})
out = os.path.join(os.path.dirname(__file__), '..', 'data', 'pop_grid.csv'); df.to_csv(out, index=False)
print('cells', len(df), '| total pop', int(df['pop'].sum()), '| kids<5', int(df.kids_u5.sum()), '| 65+', int(df.elderly_65p.sum()))
