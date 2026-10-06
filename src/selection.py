"""Classement et sélection des meilleures features."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif


def rank_features(ds, F, target="Class", seed=0):
    """Score combiné de 4 critères (rang moyen) : information mutuelle, importance RF, |d| de Cohen, F-ANOVA."""
    pos = ds[ds[target] == 1]
    sample = pd.concat([pos, ds[ds[target] == 0].sample(min(12000, (ds[target] == 0).sum()), random_state=seed)])
    X, y = sample[F], sample[target]
    mi = pd.Series(mutual_info_classif(X, y, random_state=seed), index=F)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=F)
    g1, g0 = ds[ds[target] == 1][F], ds[ds[target] == 0][F]
    d = ((g1.mean() - g0.mean()) / np.sqrt((g1.var() + g0.var()) / 2 + 1e-12)).abs()
    fstat = pd.Series(np.nan_to_num(f_classif(X, y)[0]), index=F)
    out = pd.DataFrame({"Info mutuelle": mi, "Importance RF": imp, "|d| Cohen": d, "F-ANOVA": fstat})
    out["Score"] = out.rank(ascending=False).mean(axis=1).rsub(len(F) + 1) / len(F)   # 1 = meilleur
    return out.sort_values("Score", ascending=False)


def select_uncorrelated(ranked, ds, k=15, thr=0.9):
    """Parcourt les features par score décroissant et écarte celles trop corrélées (|r| > thr) à une déjà choisie."""
    corr = ds[list(ranked.index)].corr().abs()
    kept = []
    for f in ranked.index:
        if all(corr.loc[f, g] <= thr for g in kept):
            kept.append(f)
        if len(kept) == k:
            break
    return kept
