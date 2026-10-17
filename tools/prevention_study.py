"""Étude de faisabilité de la prévention : peut-on distinguer les fenêtres PRÉ-CRISE des fenêtres inter-critiques ?

    python -m tools.prevention_study      (depuis la racine du projet)

Cible : fenêtre située dans les `HORIZON` s avant le début d'une crise (et qui ne chevauche pas la crise) = 1,
        fenêtre calme (>= QUIET s avant la prochaine crise et >= POST s après la précédente) = 0.
        Les fenêtres de crise, de post-crise et de la zone intermédiaire sont exclues.
Validation : un enregistrement laissé de côté (aucune fuite entre fenêtres voisines), 3 modèles (DT, RF, KNN), toutes les features.
Écrit results/prevention_study.csv (par modèle) et results/prevention_lead.csv (par délai avant la crise).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_annotations
from src.dataset import feature_cols, load_or_build_dataset
from src.models import loo_proba

warnings.filterwarnings("ignore")
HORIZON, QUIET, POST = 30.0, 60.0, 15.0
BINS = [(0, 5), (5, 10), (10, 20), (20, 30)]


def build_target(ds, annots):
    """Ajoute `t_to_onset` (s avant la prochaine crise, +inf sinon), `t_since_end` et la cible `pre` (NaN = exclu)."""
    tto, tse = np.full(len(ds), np.inf), np.full(len(ds), np.inf)
    for rec, sub in annots.groupby("rec_id"):
        m = (ds.rec_id == rec).to_numpy()
        te, ts = ds.t_end.to_numpy()[m], ds.t_start.to_numpy()[m]
        a, b = np.full(m.sum(), np.inf), np.full(m.sum(), np.inf)
        for d, f in zip(sub.debut_s, sub.fin_s):
            a = np.where((te <= d) & (d - te < a), d - te, a)
            b = np.where((ts >= f) & (ts - f < b), ts - f, b)
        tto[m], tse[m] = a, b
    pre = np.full(len(ds), np.nan)
    quiet = (ds.Class.to_numpy() == 0) & (tto >= QUIET) & (tse >= POST)
    pre[quiet] = 0
    pre[(ds.Class.to_numpy() == 0) & (tto < HORIZON) & (tse >= POST)] = 1
    return ds.assign(t_to_onset=tto, pre=pre)


def main():
    ds = load_or_build_dataset()
    d = build_target(ds, load_annotations())
    d = d[d.pre.notna()].reset_index(drop=True)
    d["pre"] = d.pre.astype(int)
    F = [c for c in feature_cols(d) if c not in ("Class3", "t_to_onset", "pre")]   # jamais le délai avant la crise : ce serait tricher
    # leave-one-recording-out sur la cible `pre` : loo_proba lit la colonne "Class"
    work = d.assign(Class=d.pre)
    print(f"{len(d)} fenêtres, dont {int(d.pre.sum())} pré-crise ({d.pre.mean():.1%}) ; {d.rec_id.nunique()} enregistrements", flush=True)
    rows, scores = [], pd.DataFrame(index=d.index)
    y = d.pre.to_numpy()
    for name in ["Decision Tree", "Random Forest", "KNN"]:
        s = loo_proba(work, F, name)
        scores[name] = s
        thr = np.quantile(s, 1 - y.mean())            # seuil = autant d'alarmes que de pré-crises (compare l'ordre des scores)
        per = [roc_auc_score(y[g], s[g]) for r in d.rec_id.unique() if len(set(y[(g := (d.rec_id == r).to_numpy())])) == 2]
        rows.append({"Modèle": name, "AUC": roc_auc_score(y, s), "PR-AUC": average_precision_score(y, s), "PR-AUC aléatoire": y.mean(),
                     "Balanced Accuracy": balanced_accuracy_score(y, (s >= thr).astype(int)),
                     "AUC médiane / patient": np.median(per), "Patients avec AUC > 0,5": f"{sum(a > .5 for a in per)}/{len(per)}"})
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows).set_index("Modèle")
    best = res.AUC.idxmax()
    lead = []
    for lo, hi in BINS:
        m = ((d.pre == 0) | ((d.t_to_onset >= lo) & (d.t_to_onset < hi))).to_numpy()
        yy = (d.pre.to_numpy()[m] == 1).astype(int)
        lead.append({"Délai avant la crise (s)": f"{lo}-{hi}", "Fenêtres pré-crise": int(yy.sum()), "AUC": roc_auc_score(yy, scores[best].to_numpy()[m])})
    lead = pd.DataFrame(lead).set_index("Délai avant la crise (s)")
    out = Path(__file__).resolve().parent.parent / "results"
    res.round(4).to_csv(out / "prevention_study.csv")
    lead.round(4).to_csv(out / "prevention_lead.csv")
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(), f"\nmeilleur modèle : {best}\n", lead.round(3).to_string())


if __name__ == "__main__":
    main()
