"""Évaluation de référence : source UNIQUE des chiffres de l'application et des documents.

    python -m src.evaluate [--rf-leaf 2]

Fait, avec exactement les modèles de `src/models.py` :
  1. leave-one-recording-out pour les 4 modèles (15 meilleures features) ;
  2. métriques sur les fenêtres (AUC, PR-AUC, Recall, Précision, F1) + intervalles de confiance bootstrap par patient ;
  3. détection temporelle avec appariement STRICT (IoU >= 0,3) et mesure large pour comparaison.
Écrit dans results/ : loo_scores.csv, summary_models.csv, summary_detection.csv, run_info.txt
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .data_loader import load_annotations
from .dataset import feature_cols, load_or_build_dataset
from .detection import compare_intervals, match_intervals, predictions_to_intervals
from .models import MODEL_NAMES, loo_proba
from .selection import rank_features, select_uncorrelated

warnings.filterwarnings("ignore")
RESULTS = Path(__file__).resolve().parent.parent / "results"
THR, SMOOTH, MIN_DUR, IOU_MIN = 0.5, 3, 2.0, 0.3


def detection_table(ds, scores, annots):
    rows, hours = [], len(ds) / 3600
    for name in scores.columns:
        tp = fn = fp = hit = ntrue = fa_large = 0
        e_s, e_e = [], []
        for rec in ds.rec_id.unique():
            m = (ds.rec_id == rec).to_numpy()
            d = ds[m]
            iv = predictions_to_intervals(d.t_start, d.t_end, (scores[name].to_numpy()[m] > THR).astype(int), smooth=SMOOTH, min_dur=MIN_DUR)
            true_iv = annots[annots.rec_id == rec]
            r = match_intervals(true_iv, iv, IOU_MIN)
            tp, fn, fp = tp + r["tp"], fn + r["fn"], fp + r["fp"]
            e_s += list(r["pairs"].err_debut)
            e_e += list(r["pairs"].err_fin)
            h, n, f = compare_intervals(true_iv, iv)
            hit, ntrue, fa_large = hit + h, ntrue + n, fa_large + f
        rows.append({"Modèle": name, "TP": tp, "FN": fn, "FP": fp, "Crises retrouvées (strict)": tp / (tp + fn),
                     "Précision événements": tp / max(tp + fp, 1), "Fausses alarmes / h": fp / hours,
                     "Erreur début médiane (s)": np.median(e_s), "Erreur début moy. abs (s)": np.mean(np.abs(e_s)),
                     "Erreur fin médiane (s)": np.median(e_e), "Erreur fin moy. abs (s)": np.mean(np.abs(e_e)),
                     "Crises retrouvées (large)": hit / ntrue, "Fausses alarmes (large)": fa_large})
    return pd.DataFrame(rows).set_index("Modèle")


def model_table(ds, scores, n_boot=300, seed=0):
    y = ds.Class.to_numpy()
    pat = [r for r in ds.rec_id.unique() if ds[ds.rec_id == r].Class.sum() > 0]
    idx = {r: np.where(ds.rec_id == r)[0] for r in pat}
    rng = np.random.default_rng(seed)
    boots = [np.concatenate([idx[r] for r in rng.choice(pat, len(pat))]) for _ in range(n_boot)]
    rows, ap_b = [], {}
    for n in scores.columns:
        s = scores[n].to_numpy()
        p = (s > THR).astype(int)
        ap_b[n] = np.array([average_precision_score(y[b], s[b]) for b in boots])
        f1_b = np.array([f1_score(y[b], (s[b] > THR).astype(int)) for b in boots])
        f1_pat = np.mean([f1_score(y[idx[r]], p[idx[r]]) for r in pat])
        rows.append({"Modèle": n, "AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s),
                     "PR-AUC IC95 bas": np.percentile(ap_b[n], 2.5), "PR-AUC IC95 haut": np.percentile(ap_b[n], 97.5),
                     "Recall": recall_score(y, p), "Précision": precision_score(y, p), "F1": f1_score(y, p),
                     "F1 IC95 bas": np.percentile(f1_b, 2.5), "F1 IC95 haut": np.percentile(f1_b, 97.5), "F1 moyen/patient": f1_pat})
    out = pd.DataFrame(rows).set_index("Modèle")
    best = out["PR-AUC"].idxmax()
    out["P(meilleur PR-AUC > ce modèle)"] = [np.nan if n == best else float((ap_b[best] > ap_b[n]).mean()) for n in out.index]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rf-leaf", type=int, default=2, help="min_samples_leaf du Random Forest")
    ap.add_argument("--models", nargs="*", default=MODEL_NAMES)
    args = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    ds = load_or_build_dataset()
    annots = load_annotations()
    F = [c for c in feature_cols(ds) if c != "Class3"]
    feats = select_uncorrelated(rank_features(ds, F), ds, 15, 0.9)
    print("features:", feats, flush=True)
    scores = pd.DataFrame(index=ds.index)
    for name in args.models:
        t = time.time()
        scores[name] = loo_proba(ds, feats, name, rf_min_leaf=args.rf_leaf)
        print(f"{name}: {time.time() - t:.0f} s", flush=True)
    pd.concat([ds[["rec_id", "t_start"]], scores], axis=1).to_csv(RESULTS / "loo_scores.csv", index=False)
    mt, dt = model_table(ds, scores), detection_table(ds, scores, annots)
    mt.round(4).to_csv(RESULTS / "summary_models.csv")
    dt.round(4).to_csv(RESULTS / "summary_detection.csv")
    (RESULTS / "run_info.txt").write_text(
        f"features ({len(feats)}): {', '.join(feats)}\nseuil={THR}, lissage={SMOOTH}, duree_min={MIN_DUR}s, IoU_min={IOU_MIN}\n"
        f"RF min_samples_leaf={args.rf_leaf}, 200 arbres ; KNN k=31 + sous-echantillonnage x5 ; SVM RBF C=1 (probabilite Platt)\n"
        f"fenetres={len(ds)}, patients={ds.rec_id.nunique()}, heures={len(ds) / 3600:.2f}\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(mt.round(3).to_string())
    print(dt.round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
