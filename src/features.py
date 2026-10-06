"""Extraction de caractéristiques temporelles + fréquentielles par fenêtre (Parties V et VI)."""
import numpy as np
from scipy.stats import kurtosis, skew

BANDS = {"Delta": (0.5, 4), "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30)}
# Crise d'absence typique : décharges pointe-onde à ~3 Hz => bande dédiée
EXTRA_BANDS = {"Hz3": (2.5, 4.5)}

TIME_FEATS = ["Mean", "Std", "Variance", "Min", "Max", "Amplitude", "RMS", "Energy",
              "LineLength", "HjorthMobility", "HjorthComplexity", "ZeroCross", "Skew", "Kurtosis"]


def channel_features(W, fs):
    """W: (n_win, n_samples, n_ch). Retourne dict nom -> (n_win, n_ch)."""
    n = W.shape[1]
    f = {}
    f["Mean"] = W.mean(1)
    f["Std"] = W.std(1)
    f["Variance"] = W.var(1)
    f["Min"] = W.min(1)
    f["Max"] = W.max(1)
    f["Amplitude"] = f["Max"] - f["Min"]
    f["RMS"] = np.sqrt((W ** 2).mean(1))
    f["Energy"] = (W ** 2).sum(1) / fs                       # énergie (par seconde de signal)
    d1 = np.diff(W, axis=1)
    d2 = np.diff(d1, axis=1)
    f["LineLength"] = np.abs(d1).sum(1)
    v0, v1, v2 = W.var(1) + 1e-12, d1.var(1) + 1e-12, d2.var(1) + 1e-12
    mob = np.sqrt(v1 / v0)
    f["HjorthMobility"] = mob
    f["HjorthComplexity"] = np.sqrt(v2 / v1) / mob
    f["ZeroCross"] = (np.diff(np.signbit(W - W.mean(1, keepdims=True)), axis=1)).sum(1) / (n / fs)
    f["Skew"] = skew(W, axis=1)
    f["Kurtosis"] = kurtosis(W, axis=1)

    # --- domaine fréquentiel : FFT fenêtrée (Hann) ---
    win = np.hanning(n)[None, :, None]
    F = np.fft.rfft((W - W.mean(1, keepdims=True)) * win, axis=1)
    psd = (np.abs(F) ** 2) / (fs * (win ** 2).sum())
    freqs = np.fft.rfftfreq(n, 1 / fs)
    df = freqs[1] - freqs[0]
    total = psd[:, (freqs >= 0.5) & (freqs <= 40)].sum(1) * df + 1e-12
    for name, (lo, hi) in {**BANDS, **EXTRA_BANDS}.items():
        m = (freqs >= lo) & (freqs < hi)
        p = psd[:, m].sum(1) * df
        f[f"Power_{name}"] = p
        f[f"Rel_{name}"] = p / total
    band = (freqs >= 0.5) & (freqs <= 40)
    pn = psd[:, band] / psd[:, band].sum(1, keepdims=True).clip(1e-12)
    f["SpecEntropy"] = -(pn * np.log(pn + 1e-12)).sum(1) / np.log(band.sum())
    f["DomFreq"] = freqs[band][np.argmax(psd[:, band], axis=1)]
    c = np.cumsum(pn, axis=1)
    f["SEF95"] = freqs[band][(c < 0.95).sum(1).clip(max=band.sum() - 1)]
    f["Ratio_Theta_Alpha"] = f["Power_Theta"] / (f["Power_Alpha"] + 1e-12)
    return f


def window_features(W, fs, aggs=("mean", "max")):
    """Agrège sur les canaux (moyenne et max) -> une ligne par fenêtre.

    L'agrégation rend les features indépendantes du nombre/ordre de canaux et garde la dimension raisonnable.
    """
    import pandas as pd
    cf = channel_features(W, fs)
    cols = {}
    for k, v in cf.items():
        if "mean" in aggs:
            cols[f"{k}_mean"] = v.mean(1)
        if "max" in aggs:
            cols[f"{k}_max"] = v.max(1)
    return pd.DataFrame(cols)


def spectrum(window_1ch, fs):
    """FFT d'une fenêtre (1 canal) pour l'affichage : (freqs, amplitude)."""
    n = len(window_1ch)
    x = (window_1ch - window_1ch.mean()) * np.hanning(n)
    return np.fft.rfftfreq(n, 1 / fs), np.abs(np.fft.rfft(x)) * 2 / n
