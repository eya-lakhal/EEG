"""Reconstruction temporelle des crises à partir des prédictions par fenêtre (Partie X)."""
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def predictions_to_intervals(t_start, t_end, pred, smooth=3, max_gap=2.0, min_dur=2.0):
    """Fenêtres prédites '1' consécutives -> intervalles [début, fin] en secondes.

    smooth : filtre médian (nb de fenêtres) pour supprimer les détections isolées
    max_gap : fusion de deux intervalles séparés de moins de max_gap secondes
    min_dur : durée minimale d'une crise retenue
    """
    p = median_filter(np.asarray(pred).astype(int), size=smooth, mode="nearest") if smooth > 1 else np.asarray(pred)
    iv, cur = [], None
    for ts, te, v in zip(t_start, t_end, p):
        if v:
            if cur and ts - cur[1] <= max_gap:
                cur[1] = te
            else:
                if cur:
                    iv.append(cur)
                cur = [ts, te]
    if cur:
        iv.append(cur)
    iv = [(a, b) for a, b in iv if b - a >= min_dur]
    return pd.DataFrame(iv, columns=["debut_s", "fin_s"]).assign(duree_s=lambda d: d.fin_s - d.debut_s)


def compare_intervals(true_iv, pred_iv, tol=2.0):
    """Une crise réelle est détectée si un intervalle prédit la chevauche (tolérance tol s).
    Retourne (n_detectées, n_réelles, n_fausses_alarmes)."""
    T = list(zip(true_iv.debut_s, true_iv.fin_s))
    P = list(zip(pred_iv.debut_s, pred_iv.fin_s))
    hit = sum(any(p0 <= t1 + tol and p1 >= t0 - tol for p0, p1 in P) for t0, t1 in T)
    fa = sum(not any(p0 <= t1 + tol and p1 >= t0 - tol for t0, t1 in T) for p0, p1 in P)
    return hit, len(T), fa
