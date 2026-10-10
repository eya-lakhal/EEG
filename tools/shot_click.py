"""Charge une URL, clique successivement sur des boutons (par texte), attend, capture. Usage: shot_click.py URL out.png W H 'bouton1' 'bouton2' ..."""
import os, sys, time
from playwright.sync_api import sync_playwright
url, out, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); clicks = sys.argv[5:]
def settle(pg):
    calm = 0; t0 = time.time()
    while time.time() - t0 < 900:
        running = pg.evaluate("() => !!document.querySelector('[data-testid=\"stStatusWidget\"]')")
        calm = calm + 1 if not running else 0
        if calm >= 3: return
        time.sleep(1)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True); pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(url); pg.wait_for_selector('[data-testid="stApp"]'); time.sleep(3); settle(pg)
    for c in clicks:
        (pg.get_by_role("tab", name=c[4:]) if c.startswith("tab:") else pg.get_by_role("button", name=c)).first.click(); time.sleep(1.5); settle(pg); time.sleep(2)
    pg.screenshot(path=out, full_page=False); b.close()
print("ok", out)
