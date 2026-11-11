import sys, requests
for pid in sys.argv[1:]:
    r = requests.get('https://data.opencity.in/api/3/action/package_show', params={'id': pid}, timeout=60).json()
    print('=====', pid, r.get('success'))
    for res in r['result']['resources']:
        print('  ', res.get('format'), '|', res.get('name'), '|', res.get('url'), '|', res.get('size'))
