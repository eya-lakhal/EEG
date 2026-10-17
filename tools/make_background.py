"""Génère static/bg_eeg.png : fond d'application en « papier EEG » (quadrillage rose, tracés bleus, une salve de pointes-ondes à 3 Hz).

    python tools/make_background.py

L'image est servie par Streamlit (`enableStaticServing`, voir .streamlit/config.toml) et posée en fond de page par ui/theme.py,
sous une couche rose semi-transparente : elle reste discrète et ne gêne jamais la lecture.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PINK, GRID, GRID_BOLD, BLUE = "#FBE4EA", "#F3CAD6", "#EDB4C5", "#1D4E96"
W, H, DPI = 19.2, 10.8, 100
rng = np.random.default_rng(7)

fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=PINK)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

# quadrillage du papier (petits carreaux tous les 0,2, gros carreaux tous les 1,0)
for x in np.arange(0, W + 0.01, 0.2):
    ax.plot([x, x], [0, H], color=GRID_BOLD if round(x * 5) % 5 == 0 else GRID, lw=0.9 if round(x * 5) % 5 == 0 else 0.5, zorder=1)
for y in np.arange(0, H + 0.01, 0.2):
    ax.plot([0, W], [y, y], color=GRID_BOLD if round(y * 5) % 5 == 0 else GRID, lw=0.9 if round(y * 5) % 5 == 0 else 0.5, zorder=1)

# 10 canaux EEG : bruit de fond + salve de pointes-ondes à 3 Hz au milieu
n_ch, x = 10, np.linspace(0, W, 3200)
burst = np.exp(-((x - 10.2) / 3.0) ** 8)                    # enveloppe de la crise
for i in range(n_ch):
    y0 = H * (0.93 - i * 0.092)
    bg = 0.10 * np.sin(x * 7.5 + i) + 0.07 * np.sin(x * 15.3 + 2 * i) + 0.05 * rng.standard_normal(x.size).cumsum() / 40
    ph = (x * 3.0 + 0.04 * i) % 1.0                           # 3 cycles par unité de temps
    sw = np.where(ph < 0.10, 1.0 - ph * 6, -0.55 * np.sin(np.pi * (ph - 0.10) / 0.90) ** 2)   # pointe puis onde lente
    y = y0 + bg + burst * sw * (0.34 + 0.05 * np.sin(i))
    ax.plot(x, y, color=BLUE, lw=1.5, alpha=0.9, solid_capstyle="round", zorder=3)

out = Path(__file__).resolve().parent.parent / "static" / "bg_eeg.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=DPI, facecolor=PINK)
print("écrit", out, out.stat().st_size // 1024, "Ko")
