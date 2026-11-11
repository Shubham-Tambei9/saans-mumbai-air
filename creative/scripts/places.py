"""ClimateVerse/OpenCity KML layers -> small web files.
parks  (doi:10.71646/8EKNG8)  -> data/parks.csv      name,type,ward,lat,lon
BEST   (doi:10.71646/MEEUPH)  -> data/bus_stops.csv  name,kind,lat,lon
pincode(doi:10.71646/A6VENV)  -> data/pincodes.geojson (simplified)"""
import os, sys, json, re
import xml.etree.ElementTree as ET
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
H = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(H, '..', 'raw'); OUT = os.path.join(H, '..', 'data')
NS = '{http://www.opengis.net/kml/2.2}'
W, S, E, N = 72.77, 18.89, 73.00, 19.28
inbox = lambda lon, lat: W <= lon <= E and S <= lat <= N
txt = lambda e: (e.text or '').strip() if e is not None else ''

def placemarks(path):
    root = ET.parse(path).getroot()
    for doc in root.iter(NS + 'Document'):
        layer = txt(doc.find(NS + 'name'))
        for pm in doc.iter(NS + 'Placemark'):
            f = {sd.get('name'): txt(sd) for sd in pm.iter(NS + 'SimpleData')}
            f['_name'] = txt(pm.find(NS + 'name'))
            pt = pm.find('.//' + NS + 'Point/' + NS + 'coordinates')
            polys = [[tuple(map(float, c.split(',')[:2])) for c in txt(r).split()]
                     for r in pm.iter(NS + 'outerBoundaryIs') for r in r.iter(NS + 'coordinates')]
            yield layer, f, (tuple(map(float, txt(pt).split(',')[:2])) if pt is not None and txt(pt) else None), polys

def pick(f, *keys):
    for k in f:
        if any(x in k.lower() for x in keys) and f[k]: return f[k]
    return ''

# ---- parks
rows = []
for layer, f, pt, _ in placemarks(os.path.join(RAW, 'parks', 'parks.kml')):
    if not pt or not inbox(*pt): continue
    L = layer.lower()
    typ = 'miyawaki' if 'miyawaki' in L else 'zoo' if 'zoo' in L else 'park'
    rows.append(dict(name=pick(f, 'name') or f['_name'], type=typ, ward=pick(f, 'ward'), lat=round(pt[1], 5), lon=round(pt[0], 5), layer=layer))
parks = pd.DataFrame(rows); print('parks', len(parks)); print(parks.layer.value_counts().to_string()); print(parks.head(3).to_string())
parks.drop(columns='layer').to_csv(os.path.join(OUT, 'parks.csv'), index=False)

# ---- BEST stops & depots
rows = []
for layer, f, pt, _ in placemarks(os.path.join(RAW, 'bus', 'bus.kml')):
    if not pt or not inbox(*pt): continue
    kind = 'depot' if 'depot' in layer.lower() else 'stop'
    rows.append(dict(name=f.get('BEST_BUS_S') or pick(f, 'name', 'stop', 'depot') or f['_name'], kind=kind, lat=round(pt[1], 5), lon=round(pt[0], 5), layer=layer))
bus = pd.DataFrame(rows); print('\nbus', len(bus)); print(bus.layer.value_counts().to_string()); print(bus.head(3).to_string())
bus.drop(columns='layer').to_csv(os.path.join(OUT, 'bus_stops.csv'), index=False)

# ---- pincodes
feats = {}
for layer, f, _, polys in placemarks(os.path.join(RAW, 'pincode', 'mumbai_pincodes.kml')):
    ps = [Polygon(p) for p in polys if len(p) >= 4]
    if not ps: continue
    g = unary_union([p.buffer(0) for p in ps])
    c = g.representative_point()
    if not inbox(c.x, c.y): continue
    pin = f.get('Pincode', '').strip()
    feats.setdefault(pin, {'name': f.get('Office_Name', '').strip(), 'geoms': []})['geoms'].append(g)
out = []
for pin, v in feats.items():
    g = unary_union(v['geoms']).simplify(0.0004, preserve_topology=True)
    area = g.area * (111.32 ** 2) * 0.945   # deg^2 -> km^2 at ~19 N
    gj = json.loads(json.dumps(mapping(g)), parse_float=lambda x: round(float(x), 5))
    out.append({'type': 'Feature', 'properties': {'pincode': pin, 'name': v['name'], 'area_km2': round(area, 2)}, 'geometry': gj})
fc = {'type': 'FeatureCollection', 'features': out}
p = os.path.join(OUT, 'pincodes.geojson'); json.dump(fc, open(p, 'w'), separators=(',', ':'))
print('\npincodes', len(out), 'size KB', os.path.getsize(p) // 1024)
