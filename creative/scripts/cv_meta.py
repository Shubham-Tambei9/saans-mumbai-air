"""Print the external source URLs and field names of ClimateVerse records (given DOI suffixes)."""
import sys, json, re, requests
for doi in sys.argv[1:]:
    r = requests.get('https://india.climateverse.net/api/datasets/:persistentId/',
                     params={'persistentId': 'doi:10.71646/' + doi}, timeout=60).json()
    s = json.dumps(r, ensure_ascii=False)
    urls = sorted(set(re.findall(r'https?://[^"\s\\]+', s)))
    print('=====', doi)
    print([u for u in urls if 'climateverse' not in u and 'doi.org/10.71646' not in u])
    v = r['data']['latestVersion']
    for b in v['metadataBlocks'].values():
        for x in b['fields']:
            if x['typeName'] in ('alternativeURL', 'otherReferences', 'dataSources', 'distributor', 'producer', 'publication',
                                 'relatedMaterial', 'originOfSources', 'characteristicOfSources'):
                print('  ', x['typeName'], ':', json.dumps(x['value'], ensure_ascii=False)[:600])
    print('  fields:', [x['typeName'] for b in v['metadataBlocks'].values() for x in b['fields']])
