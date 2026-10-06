"""Construction du dataset de classification (1 ligne = 1 fenêtre EEG)."""
import numpy as np
import pandas as pd

from .data_loader import load_annotations, load_recording, list_recordings
from .features import window_features
from .preprocessing import (COMMON_EEG, bandpass, make_windows, robust_scale,
                            sample_labels, select_channels)


def build_recording(rec_id, annots, win_s=2.0, overlap=0.5, pre_s=10.0, min_frac=0.5):
    x, ch, fs = load_recording(rec_id)
    x, _ = select_channels(x, ch, COMMON_EEG)
    x = robust_scale(bandpass(x, fs))
    sub = annots[annots.rec_id == rec_id]
    crises = list(zip(sub.debut_s, sub.fin_s))
    y = sample_labels(len(x), fs, crises)
    W, frac, t0 = make_windows(x, y, fs, win_s, overlap)
    df = window_features(W, fs)
    df.insert(0, "rec_id", rec_id)
    df.insert(1, "t_start", t0)
    df.insert(2, "t_end", t0 + win_s)
    df.insert(3, "frac_crise", frac)
    df["Class"] = (frac >= min_frac).astype(int)
    # Classe 3 états : 0 = inter-critique, 1 = pré-critique (pre_s avant le début), 2 = crise
    c3 = df["Class"].to_numpy() * 2
    for d, _ in crises:
        c3[(df.t_end > d - pre_s) & (df.t_end <= d) & (c3 == 0)] = 1
    df["Class3"] = c3
    return df


def build_dataset(win_s=2.0, overlap=0.5, pre_s=10.0, min_frac=0.5, progress=None):
    annots = load_annotations()
    recs = list_recordings()
    parts = []
    for i, r in enumerate(recs):
        parts.append(build_recording(r, annots, win_s, overlap, pre_s, min_frac))
        if progress:
            progress((i + 1) / len(recs), r)
    return pd.concat(parts, ignore_index=True)


META_COLS = ["rec_id", "t_start", "t_end", "frac_crise", "Class", "Class3"]


def feature_cols(df):
    return [c for c in df.columns if c not in META_COLS]
