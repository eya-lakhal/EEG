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


def _iou(a, b):
    """Intersection sur union de deux intervalles (début, fin)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_intervals(true_iv, pred_iv, iou_min=0.3):
    """Appariement STRICT, un pour un : chaque intervalle prédit ne compte que pour UNE crise réelle.

    On associe les paires par recouvrement (IoU) décroissant, tant que l'IoU >= iou_min.
    Retourne un dict : tp (crises retrouvées), fn (manquées), fp (fausses alarmes) et `pairs`, un tableau
    avec les erreurs de bornes (prédit - réel, en secondes) des paires appariées.
    """
    T = list(zip(true_iv.debut_s, true_iv.fin_s))
    P = list(zip(pred_iv.debut_s, pred_iv.fin_s))
    cand = sorted(((_iou(p, t), i, j) for i, p in enumerate(P) for j, t in enumerate(T)), reverse=True)
    used_p, used_t, pairs = set(), set(), []
    for v, i, j in cand:
        if v < iou_min:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        pairs.append({"vrai": j, "predit": i, "iou": v, "err_debut": P[i][0] - T[j][0], "err_fin": P[i][1] - T[j][1]})
    return {"tp": len(pairs), "fn": len(T) - len(pairs), "fp": len(P) - len(pairs),
            "pairs": pd.DataFrame(pairs, columns=["vrai", "predit", "iou", "err_debut", "err_fin"])}


# Mesure LARGE (optimiste), gardée pour comparaison : un intervalle prédit qui couvre plusieurs crises
# rapprochées compte pour toutes. Préférer `match_intervals`.
def compare_intervals(true_iv, pred_iv, tol=2.0):
    """Une crise réelle est détectée si un intervalle prédit la chevauche (tolérance tol s).
    Retourne (n_detectées, n_réelles, n_fausses_alarmes)."""
    T = list(zip(true_iv.debut_s, true_iv.fin_s))
    P = list(zip(pred_iv.debut_s, pred_iv.fin_s))
    hit = sum(any(p0 <= t1 + tol and p1 >= t0 - tol for p0, p1 in P) for t0, t1 in T)
    fa = sum(not any(p0 <= t1 + tol and p1 >= t0 - tol for t0, t1 in T) for p0, p1 in P)
    return hit, len(T), fa
