"""Overture Maps base theme, land_use class=landfill, Mumbai bbox (ClimateVerse doi:10.71646/MWQNR8, release 2026-10-21.0)
-> creative/data/landfills.geojson"""
import os, json, duckdb
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'landfills.geojson')
REL = os.environ.get('OVERTURE_RELEASE', '2026-10-21.0')
W, S, E, N = 72.77, 18.89, 73.00, 19.28
con = duckdb.connect()
for ext in ('httpfs', 'spatial'):
    con.execute(f'INSTALL {ext}; LOAD {ext};')
con.execute("SET s3_region='us-west-2'; SET geometry_always_xy = true;")
q = f"""
SELECT id, names.primary AS name, class, subtype, ST_AsGeoJSON(ST_SimplifyPreserveTopology(geometry, 0.00005)) AS gj,
       ST_Area_Spheroid(geometry) / 1e4 AS area_ha
FROM read_parquet('s3://overturemaps-us-west-2/release/{REL}/theme=base/type=land_use/*', hive_partitioning=1)
WHERE bbox.xmin < {E} AND bbox.xmax > {W} AND bbox.ymin < {N} AND bbox.ymax > {S} AND class = 'landfill'
"""
rows = con.execute(q).fetchall()
feats = [{'type': 'Feature', 'properties': {'name': r[1] or '', 'class': r[2], 'subtype': r[3], 'area_ha': round(r[5] or 0, 1)},
          'geometry': json.loads(r[4], parse_float=lambda x: round(float(x), 5))} for r in rows]
json.dump({'type': 'FeatureCollection', 'features': feats}, open(OUT, 'w'), separators=(',', ':'))
print(len(feats), 'landfill polygons;', sorted(((f['properties']['name'], f['properties']['area_ha']) for f in feats), key=lambda x: -x[1])[:12])
