"""Évaluation du protocole du sujet : split STRATIFIÉ 80/20 sur les fenêtres, répété sur plusieurs graines.

    python -m tools.eval_stratified [--seeds 5]      (depuis la racine du projet)

Pour chaque graine : 80/20 stratifié ; dans le train, un sous-ensemble de validation stratifié (20 %) sert à APPRENDRE le seuil
(maximise le F1) ; le test n'est jamais utilisé pour choisir le seuil. On compare le seuil 0,5 et le seuil appris, avec et sans
data augmentation des crises. Écrit results/stratified_80_20.csv (moyenne et écart-type sur les graines).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import feature_cols, load_or_build_dataset
from src.models import MODEL_NAMES, build_model, learn_threshold
from src.selection import rank_features, select_uncorrelated

warnings.filterwarnings("ignore")


def scores(y, p):
    return {"Balanced Accuracy": balanced_accuracy_score(y, p), "Recall": recall_score(y, p, zero_division=0),
            "Précision": precision_score(y, p, zero_division=0), "F1": f1_score(y, p, zero_division=0)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args(argv)
    ds = load_or_build_dataset()
    F = [c for c in feature_cols(ds) if c != "Class3"]
    feats = select_uncorrelated(rank_features(ds, F), ds, 15, 0.9)
    X, y = ds[feats].to_numpy(), ds.Class.to_numpy()
    rows = []
    for seed in range(args.seeds):
        itr, ite = train_test_split(np.arange(len(ds)), test_size=0.2, stratify=y, random_state=seed)
        iin, iva = train_test_split(itr, test_size=0.2, stratify=y[itr], random_state=seed)
        for aug in (False, True):
            for name in MODEL_NAMES:
                m = build_model(name, augment=aug).fit(X[iin], y[iin])
                thr = learn_threshold(y[iva], m.predict_proba(X[iva])[:, 1])
                s = m.predict_proba(X[ite])[:, 1]
                for mode, t in (("seuil 0,5", 0.5), ("seuil appris", thr)):
                    rows.append({"seed": seed, "Augmentation": "oui" if aug else "non", "Modèle": name, "Seuil": mode,
                                 "valeur seuil": t, **scores(y[ite], (s > t).astype(int))})
        print(f"graine {seed} terminée", flush=True)
    d = pd.DataFrame(rows)
    keys = ["Augmentation", "Modèle", "Seuil"]
    out = d.groupby(keys).agg(["mean", "std"]).drop(columns="seed")
    out.columns = [f"{a} {'moy' if b == 'mean' else 'std'}" for a, b in out.columns]
    out.round(4).to_csv(Path(__file__).resolve().parent.parent / "results" / "stratified_80_20.csv")
    pd.set_option("display.width", 250)
    print(d.groupby(keys)[["Balanced Accuracy", "Recall", "Précision", "F1", "valeur seuil"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
