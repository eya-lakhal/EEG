"""Capture d'écran de l'app Streamlit avec le vrai Edge (Playwright). Usage : python cache/shot.py URL sortie.png [largeur] [hauteur] [attente_s] [fullpage]"""
import os, sys, time
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
w = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
h = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
wait = float(sys.argv[5]) if len(sys.argv) > 5 else 8
full = (sys.argv[6] == "1") if len(sys.argv) > 6 else True
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(url)
    pg.wait_for_selector('[data-testid="stApp"]', timeout=60000)
    t0 = time.time(); calm = 0
    while time.time() - t0 < 900:       # attend que le script ait fini (pas de widget "RUNNING") 3 contrôles de suite
        running = pg.evaluate("() => !!document.querySelector('[data-testid=\"stStatusWidget\"]')")
        ready = pg.evaluate("() => (document.querySelector('section.main')||document.body).innerText.length > 300")
        calm = calm + 1 if (not running and ready) else 0
        if calm >= 3: break
        time.sleep(1)
    time.sleep(wait)
    pg.screenshot(path=out, full_page=full)
    b.close()
print("ok", out)
