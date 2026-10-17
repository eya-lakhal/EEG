"""Prépare l'image de fond de l'application : static/brain_source.jpg -> static/bg_brain.png

    python tools/make_background.py

L'image d'origine (cerveau lumineux, violet et cyan sur fond sombre) est trop saturée pour un outil médical. On la convertit en
camaïeu : les zones lumineuses (les sillons, le contour du cerveau) deviennent un bleu marine, le fond sombre devient le rose de la page.
Les bords sont fondus dans le rose (masque elliptique) pour qu'il n'y ait ni cadre ni effet de collage.
Le résultat est posé en fond par ui/theme.py, sous un voile rose supplémentaire : il reste reconnaissable mais discret.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

PINK = np.array([251, 228, 234], float)     # fond de page
NAVY = np.array([22, 48, 95], float)        # encre des sillons
WIDTH = 1800                                  # l'original fait 800 px : on agrandit en lissant (le voile masque le grain)

root = Path(__file__).resolve().parent.parent / "static"
im = Image.open(root / "brain_source.jpg").convert("RGB")
h = round(im.height * WIDTH / im.width)
im = im.resize((WIDTH, h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.6))

lum = np.asarray(im.convert("L"), float) / 255.0
smooth = lambda e0, e1, x: np.clip((x - e0) / (e1 - e0), 0, 1) ** 2 * (3 - 2 * np.clip((x - e0) / (e1 - e0), 0, 1))
ink = smooth(0.34, 0.95, lum)                  # seules les zones lumineuses laissent de l'encre

yy, xx = np.mgrid[0:h, 0:WIDTH]
r = np.sqrt(((xx - WIDTH / 2) / (WIDTH / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
mask = 1 - smooth(0.62, 1.0, r)                # plein au centre, fondu vers les bords

a = (ink * mask)[..., None]
out = PINK * (1 - a) + NAVY * a
Image.fromarray(out.astype(np.uint8)).save(root / "bg_brain.png", optimize=True)
print("écrit", root / "bg_brain.png", (root / "bg_brain.png").stat().st_size // 1024, "Ko")
