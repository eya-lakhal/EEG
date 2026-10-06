"""Chargement des enregistrements EEG et des annotations depuis les CSV de data_csv/."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
FS = 256  # Hz (256 échantillons par ligne d'une seconde)
EEG_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
OTHER_CHANNELS = ["EMG1", "EMG2", "SLI", "ECG"]
DAY = 86400.0


CSV_DIR = ROOT / "data_csv"   # données converties sans perte par src/convert_to_csv.py


def list_recordings():
    """IDs des enregistrements (ex: '190304A_E') ayant un CSV de signal dans data_csv/."""
    return sorted(p.stem for p in CSV_DIR.glob("*.csv")
                  if p.stem[:6].isdigit() and not p.stem.endswith("_events"))


def load_recording(rec_id, use_cache=True):
    """Lit data_csv/<rec>.csv -> (signal[n_samples, n_canaux] float64, noms_canaux, fs).

    Lecture exacte (float64). Un cache .npz float64 dérivé du CSV accélère les ouvertures suivantes ;
    il est supprimable et se régénère à partir du CSV.
    """
    cf = CACHE / f"{rec_id}_f64.npz"
    if use_cache and cf.exists() and cf.stat().st_mtime >= (CSV_DIR / f"{rec_id}.csv").stat().st_mtime:
        z = np.load(cf, allow_pickle=True)
        return z["x"], [str(c) for c in z["ch"]], int(z["fs"])
    import pyarrow.csv as pacsv
    t = pacsv.read_csv(CSV_DIR / f"{rec_id}.csv").to_pandas()
    time = t.pop("Time").to_numpy()
    fs = int(round(1.0 / (time[1] - time[0])))
    x = t.to_numpy(dtype=np.float64)
    ch = [c.replace("EEG", "", 1) if c.startswith("EEG") else c for c in t.columns]
    CACHE.mkdir(exist_ok=True)
    np.savez(cf, x=x, ch=np.array(ch), fs=fs)
    return x, ch, fs


def load_header(rec_id):
    """Marqueurs de l'enregistreur (data_csv/<rec>_events.csv) : Onset_s, Annotations."""
    h = pd.read_csv(CSV_DIR / f"{rec_id}_events.csv", encoding="utf-8")
    return h[["Onset_s", "Annotations"]]


def load_annotations(path=None, max_dur=120.0):
    """Lit data_csv/Annotations_MAJ.csv (copie exacte de 'Annotations MAJ.xlsx') -> DataFrame (rec_id, crise, debut_s, fin_s, duree_s).

    Les temps Excel sont des fractions de jour (heure de la journée). Le temps relatif
    à l'enregistrement = (DebCE - DebutTrace) * 86400.
    - Doublons d'un même enregistrement : la dernière ligne (mise à jour) est conservée.
    - Crises invalides (durée <= 0 ou > max_dur s, ex. FinCE1 erroné de 210504B-C) écartées.
    """
    path = path or CSV_DIR / "Annotations_MAJ.csv"
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={df.columns[0]: "ID"})
    df = df[df["ID"].astype(str).str.match(r"^\d{6}[A-Z]-[A-Z]$")]
    df = df.drop_duplicates("ID", keep="last")
    rows, dropped = [], []
    for _, r in df.iterrows():
        rec = r["ID"].replace("-", "_")
        t0 = float(r["DebutTrace"])
        k = 0
        for i in range(1, 13):
            dcol = [c for c in df.columns if c.lower().replace("t", "", 0) in (f"debce{i}", f"debutce{i}", f"debtce{i}")]
            fcol = f"FinCE{i}"
            if not dcol or pd.isna(r[dcol[0]]) or pd.isna(r.get(fcol)):
                continue
            deb, fin = (float(r[dcol[0]]) - t0) * DAY, (float(r[fcol]) - t0) * DAY
            dur = fin - deb
            if dur <= 0 or dur > max_dur or deb < 0:
                dropped.append((rec, i, deb, fin))
                continue
            k += 1
            rows.append((rec, k, deb, fin, dur))
    out = pd.DataFrame(rows, columns=["rec_id", "crise", "debut_s", "fin_s", "duree_s"])
    out.attrs["dropped"] = dropped
    return out
