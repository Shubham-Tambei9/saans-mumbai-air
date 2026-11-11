"""3D-GloBFP building footprints with height (ClimateVerse doi:10.71646/IEBMQY, grid 1724) -> creative/site/buildings.js
Each kept building becomes an oriented box: centre, width, depth, rotation, height. Packed as base64 (8 bytes each):
  uint16 x, uint16 z (0..65535 over the box below), uint8 w, uint8 d (2 m steps), uint8 angle (0..180 deg), uint8 height (m, capped 255)"""
import os, sys, glob, json, base64, numpy as np, duckdb
from shapely import wkb
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
H = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(H, '..', 'raw', 'globfp'); D = os.path.join(H, '..', 'data')
KEEP = int(sys.argv[1]) if len(sys.argv) > 1 else 110000
W, S, E, N = 72.77, 18.89, 73.00, 19.28
LAT0, LON0 = 19.085, 72.885; KX = 111.32 * np.cos(np.radians(LAT0)); KZ = 110.6     # km per degree (same as city.html)
shp = glob.glob(os.path.join(RAW, '**', '*.shp'), recursive=True); shp = [s for s in shp if 'world_grid' not in s][0]
con = duckdb.connect(); con.execute('INSTALL spatial; LOAD spatial; SET geometry_always_xy = true;')
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM ST_Read('{shp}') LIMIT 1").fetchall()]
hcol = [c for c in cols if 'eight' in c.lower()][0]; print('file', os.path.basename(shp), '| columns', cols, '| height column', hcol)
rows = con.execute(f"""SELECT ST_AsWKB(geom), "{hcol}" FROM ST_Read('{shp}')
                       WHERE "{hcol}" IS NOT NULL AND ST_XMin(geom) < {E} AND ST_XMax(geom) > {W} AND ST_YMin(geom) < {N} AND ST_YMax(geom) > {S}""").fetchall()
print('buildings in Mumbai box:', len(rows))
city = prep(unary_union([shape(f['geometry']) for f in json.load(open(os.path.join(D, 'pincodes.geojson')))['features']]).buffer(0.002))
out = []
for g, h in rows:
    p = wkb.loads(bytes(g)); c = p.centroid
    if not (W <= c.x <= E and S <= c.y <= N) or not city.contains(c): continue
    area = p.area * KX * KZ * 1e6
    if area < 40 or h <= 0: continue
    out.append((area * h, p, float(h), area))
print('inside the pincode map, >= 40 m2:', len(out), '| height m: median', np.median([o[2] for o in out]), 'p99', np.percentile([o[2] for o in out], 99), 'max', max(o[2] for o in out))
out.sort(key=lambda o: -o[0]); out = out[:KEEP]
buf = bytearray()
for _, p, h, area in out:
    r = p.minimum_rotated_rectangle
    xs, ys = r.exterior.coords.xy
    pts = [((x - LON0) * KX, (y - LAT0) * KZ) for x, y in zip(xs, ys)][:4]          # km east, km north
    e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]); e2 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
    w = np.hypot(*e1) * 1000; d = np.hypot(*e2) * 1000
    k = area / max(w * d, 1e-6); w *= np.sqrt(k); d *= np.sqrt(k)                    # keep the true footprint area
    ang = np.degrees(np.arctan2(e1[1], e1[0])) % 180
    cx = sum(q[0] for q in pts) / 4; cy = sum(q[1] for q in pts) / 4
    X = int(round((cx + 14) / 28 * 65535)); Z = int(round((cy + 24) / 48 * 65535))
    if not (0 <= X <= 65535 and 0 <= Z <= 65535): continue
    buf += np.array([X, Z], dtype='<u2').tobytes() + bytes([min(255, max(1, round(w / 2))), min(255, max(1, round(d / 2))),
                                                            int(round(ang / 180 * 255)), min(255, max(1, round(h)))])
n = len(buf) // 8
js = 'window.SAANS_B = {n:%d, src:"3D-GloBFP v2 (doi:10.71646/IEBMQY)", total:%d, d:"%s"};\n' % (n, len(rows), base64.b64encode(bytes(buf)).decode())
p = os.path.join(H, '..', 'site', 'buildings.js'); open(p, 'w').write(js)
print('kept', n, 'buildings | smallest kept volume m3', round(out[-1][0]), '| buildings.js KB', os.path.getsize(p) // 1024)
