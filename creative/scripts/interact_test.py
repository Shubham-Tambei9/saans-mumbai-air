"""Interaction check: chart tooltips and map taps, on desktop (mouse) and phone (touch)."""
import sys
from playwright.sync_api import sync_playwright
BASE = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8765'
with sync_playwright() as p:
    br = p.chromium.launch(channel='chrome', headless=True)
    for name, opts in [('desktop', dict(viewport={'width': 1280, 'height': 900})), ('iPhone 13', {k: v for k, v in p.devices['iPhone 13'].items() if k != 'default_browser_type'})]:
        ctx = br.new_context(**opts); pg = ctx.new_page(); errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('response', lambda r: errs.append(f'{r.status} {r.url}') if r.status >= 400 else None)
        pg.goto(BASE + '/index.html?theme=light', wait_until='networkidle'); pg.wait_for_timeout(1200)
        touch = name != 'desktop'
        for sel in ['#placeChart', '#dwChart']:
            pg.locator(sel).scroll_into_view_if_needed(); pg.wait_for_timeout(200)
            b = pg.locator(sel + ' svg').bounding_box(); x, y = b['x'] + b['width'] * 0.7, b['y'] + b['height'] * 0.5
            top = pg.evaluate(f'(() => {{ const e = document.elementFromPoint({x}, {y}); return e ? e.tagName + "." + (e.getAttribute("class") || "") : null; }})()')
            if touch: pg.touchscreen.tap(x, y)
            else: pg.mouse.move(x, y); pg.mouse.move(x + 3, y)
            pg.wait_for_timeout(250); tip = pg.locator(sel + ' .tip')
            print(f'{name:9s} {sel:12s} element under finger: {top:25s} tooltip: {tip.is_visible()} {tip.inner_text()[:60]!r}' if tip.is_visible() else f'{name:9s} {sel:12s} element under finger: {top:25s} tooltip: False')
        pg.locator('#map').scroll_into_view_if_needed(); pg.wait_for_timeout(300)
        m = pg.locator('#map').bounding_box(); x, y = m['x'] + m['width'] * 0.35, m['y'] + m['height'] * 0.3
        top = pg.evaluate(f'(() => {{ const e = document.elementFromPoint({x}, {y}); return e ? e.tagName + "." + (e.getAttribute("class") || "").slice(0, 40) : null; }})()')
        before = pg.inner_text('#placeTitle')
        if touch: pg.touchscreen.tap(x, y)
        else: pg.mouse.click(x, y)
        pg.wait_for_timeout(700)
        print(f'{name:9s} map tap on {top}: "{before}" -> "{pg.inner_text("#placeTitle")}"')
        print(f'{name:9s} errors/404s: {errs or "none"}'); ctx.close()
    br.close()
