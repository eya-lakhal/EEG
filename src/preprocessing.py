"""Prétraitement : filtrage passe-bande, normalisation robuste, étiquetage, fenêtrage."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfiltfilt

from .data_loader import EEG_CHANNELS

# 19 canaux EEG (système 10-20). Chaque patient utilise tous ceux présents dans son fichier
# (200625A_F n'en contient que 8). ECG/EMG/SLI ne sont pas de l'EEG et sont exclus.
COMMON_EEG = EEG_CHANNELS


def select_channels(x, ch, wanted=COMMON_EEG):
    """Retourne (signal, noms) pour tous les canaux EEG de `wanted` présents dans le fichier."""
    names = [c for c in wanted if c in ch]
    return x[:, [ch.index(c) for c in names]].astype(np.float64), names


def bandpass(x, fs, lo=0.5, hi=40.0, order=4):
    """Passe-bande Butterworth à phase nulle (retire dérive basse fréquence + bruit secteur 50 Hz)."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def robust_scale(x):
    """Normalisation par canal : (x - médiane) / IQR. Gère les différences d'amplitude entre patients."""
    med = np.median(x, axis=0)
    q1, q3 = np.percentile(x, [25, 75], axis=0)
    return (x - med) / np.maximum(q3 - q1, 1e-9)


def sample_labels(n, fs, crises):
    """Vecteur de 0/1 par échantillon (Question 5). crises = liste de (debut_s, fin_s)."""
    y = np.zeros(n, dtype=np.int8)
    for d, f in crises:
        y[int(round(d * fs)):int(round(f * fs)) + 1] = 1
    return y


def make_windows(x, y, fs, win_s=2.0, overlap=0.5, min_frac=0.5):
    """Découpe en fenêtres (Question 7).

    Retourne W[n_win, win, n_ch], frac_crise[n_win], t_debut[n_win] (s).
    Fenêtre positive si frac_crise >= min_frac.
    """
    w = int(win_s * fs)
    step = max(1, int(w * (1 - overlap)))
    W = sliding_window_view(x, w, axis=0)[::step]          # (n_win, n_ch, w)
    W = np.transpose(W, (0, 2, 1))                          # (n_win, w, n_ch)
    frac = sliding_window_view(y.astype(np.float32), w)[::step].mean(axis=1)
    t0 = np.arange(len(W)) * step / fs
    return W, frac, t0
