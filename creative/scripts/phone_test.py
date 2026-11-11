"""Real phone emulation (Pixel 7 / iPhone-size) of the Saans pages with installed Chrome; checks layout width,
JS errors, chart tooltips, map tap, language switch and the 3D page. Usage: python phone_test.py [base_url]"""
import sys, os
from playwright.sync_api import sync_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8765'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pdf')
with sync_playwright() as p:
    br = p.chromium.launch(channel='chrome', headless=True, args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    for dev in ['Pixel 7', 'iPhone 13']:
        d = dict(p.devices[dev]); d.pop('default_browser_type', None)
        ctx = br.new_context(**d); pg = ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
        pg.goto(BASE + '/index.html?theme=light', wait_until='networkidle'); pg.wait_for_timeout(1500)
        vw = pg.evaluate('innerWidth'); sw = pg.evaluate('document.documentElement.scrollWidth'); sy = pg.evaluate('scrollY')
        wide = pg.evaluate("""[...document.querySelectorAll('body *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1 && !e.closest('.leaflet-container') && !e.closest('.tablewrap'))
                              .slice(0, 6).map(e => e.tagName + '.' + e.className + ' ' + Math.round(e.getBoundingClientRect().right))""")
        print(f'\n== {dev}: viewport {vw}px, page width {sw}px, opens at scrollY={sy}, overflowing elements: {wide}')
        pg.screenshot(path=os.path.join(OUT, f'_m_{dev[:5]}_top.png'))
        # chart tooltip on the place chart (tap) and the Diwali chart (hover)
        for sel in ['#placeChart', '#dwChart']:
            box = pg.locator(sel + ' svg').bounding_box(); pg.mouse.move(box['x'] + box['width'] * 0.7, box['y'] + box['height'] * 0.5)
            pg.wait_for_timeout(200); tip = pg.locator(sel + ' .tip')
            print(f'   {sel} tooltip visible: {tip.is_visible()} | text: {tip.inner_text()[:80]!r}' if tip.is_visible() else f'   {sel} tooltip visible: False')
        pg.locator('#placeChart').scroll_into_view_if_needed(); pg.screenshot(path=os.path.join(OUT, f'_m_{dev[:5]}_place.png'))
        # tap the map, check the place panel updates
        before = pg.inner_text('#placeTitle'); m = pg.locator('#map').bounding_box()
        pg.mouse.click(m['x'] + m['width'] * 0.55, m['y'] + m['height'] * 0.55); pg.wait_for_timeout(600)
        print(f'   map tap: place panel "{before}" -> "{pg.inner_text("#placeTitle")}" | best hours rows: {pg.locator("#bestHours > div").count()}')
        # day switch in exposure, language switch
        pg.locator('#daySeg button').nth(1).click(); pg.wait_for_timeout(300); print('   exposure 12 Nov KPI:', pg.locator('#kpis .val').first.inner_text())
        pg.locator('[data-lang=mr]').click(); pg.wait_for_timeout(500); print('   Marathi heading:', pg.inner_text('h1')[:40], '| page width', pg.evaluate('document.documentElement.scrollWidth'))
        pg.locator('#expH' if pg.locator('#expH').count() else '#pinTable').first.scroll_into_view_if_needed(); pg.screenshot(path=os.path.join(OUT, f'_m_{dev[:5]}_exposure.png'))
        # 3D page
        c = ctx.new_page(); c.on('pageerror', lambda e: errs.append('city: ' + str(e)))
        c.goto(BASE + '/city.html?still=1&h=22', wait_until='networkidle'); c.wait_for_timeout(4000)
        over = c.evaluate("""[...document.querySelectorAll('.panel, .panel *')].filter(e => e.getBoundingClientRect().right > innerWidth + 1).slice(0, 5).map(e => (e.id || e.tagName) + ' ' + Math.round(e.getBoundingClientRect().right))""")
        print(f'   3D page width {c.evaluate("document.documentElement.scrollWidth")}px, panels overflowing: {over}, HUD PM: {c.inner_text("#pmNow")}')
        c.screenshot(path=os.path.join(OUT, f'_m_{dev[:5]}_city.png'))
        print('   JS errors:', errs or 'none'); ctx.close()
    br.close()
