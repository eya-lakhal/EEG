"""Conversion SANS PERTE de tous les fichiers sources en CSV.

Produit dans data_csv/ :
  <rec>.csv          Time (s) + tous les canaux (noms d'origine, ordre d'origine, float64 pleine précision)
  <rec>_events.csv   marqueurs de l'enregistreur (fichier *_0000h.mat) : Onset_s, Duration_s, Annotations
  Annotations_MAJ.csv  copie de l'Excel (seules les lignes entièrement vides sont omises)
  manifest.csv       récapitulatif (lignes, fs, canaux)

Usage :  python -m src.convert_to_csv            (convertit + vérifie l'égalité exacte avec les .mat)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = ROOT / "data_csv"
FS = 256


def read_mat(rec):
    """Lit un .mat d'origine -> (matrice float64 [n, canaux], noms d'origine)."""
    import matio
    d = matio.load_from_mat(str(ROOT / f"{rec}_0000d.mat"))["output_d"]
    cols = list(d.columns)
    blocks = [np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in d[c]]) for c in cols]
    n = min(len(b) for b in blocks)
    assert all(len(b) == n for b in blocks), f"{rec}: canaux de longueurs différentes"
    return np.stack([b[:n] for b in blocks], axis=1), cols


def write_signal_csv(rec):
    x, cols = read_mat(rec)
    t = np.arange(len(x), dtype=np.float64) / FS
    table = pa.table({"Time": t, **{c: x[:, i] for i, c in enumerate(cols)}})
    pacsv.write_csv(table, CSV_DIR / f"{rec}.csv")
    return x, cols


def write_events_csv(rec):
    import matio
    h = matio.load_from_mat(str(ROOT / f"{rec}_0000h.mat"))["output_h"]
    out = pd.DataFrame({"Onset_s": h.index.total_seconds(), "Duration_s": h["Duration"].dt.total_seconds().to_numpy(),
                        "Annotations": h["Annotations"].to_numpy()})
    out.to_csv(CSV_DIR / f"{rec}_events.csv", index=False, encoding="utf-8")


def write_annotations_csv():
    df = pd.read_excel(ROOT / "Annotations MAJ.xlsx").dropna(how="all")
    df = df.rename(columns={df.columns[0]: "ID"})
    df.to_csv(CSV_DIR / "Annotations_MAJ.csv", index=False, encoding="utf-8")
    return df


def load_signal_csv(rec):
    """Relit un CSV de signal en float64 exact -> (matrice, noms de colonnes sans Time, vecteur Time)."""
    t = pacsv.read_csv(CSV_DIR / f"{rec}.csv", convert_options=pacsv.ConvertOptions(
        column_types=None)).to_pandas()
    time = t.pop("Time").to_numpy()
    return t.to_numpy(dtype=np.float64), list(t.columns), time


def main():
    CSV_DIR.mkdir(exist_ok=True)
    recs = sorted(p.name.replace("_0000d.mat", "") for p in ROOT.glob("*_0000d.mat"))
    manifest, bad = [], []
    for i, rec in enumerate(recs, 1):
        x, cols = write_signal_csv(rec)
        write_events_csv(rec)
        y, cols2, time = load_signal_csv(rec)            # vérification : relecture du CSV
        exact = bool(np.array_equal(x, y)) and cols == cols2 and np.array_equal(time, np.arange(len(x)) / FS)
        if not exact:
            bad.append(rec)
        manifest.append({"rec_id": rec, "n_rows": len(x), "fs_hz": FS, "duree_s": len(x) / FS,
                         "n_columns": len(cols), "columns": "|".join(cols), "identique_au_mat": exact,
                         "taille_Mo": round((CSV_DIR / f"{rec}.csv").stat().st_size / 1e6, 1)})
        print(f"[{i}/{len(recs)}] {rec}: {x.shape} exact={exact} {manifest[-1]['taille_Mo']} Mo", flush=True)
    write_annotations_csv()
    pd.DataFrame(manifest).to_csv(CSV_DIR / "manifest.csv", index=False)
    print("ÉCARTS:", bad if bad else "aucun - conversion identique bit à bit")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
