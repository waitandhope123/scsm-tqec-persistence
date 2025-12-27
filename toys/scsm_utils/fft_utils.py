"""
SCSM TOYBENCH — FFT / SPECTRAL UTILITIES

Provides:
- detrend
- compute PSD-like spectrum (simple, deterministic)
- peak finding helper

Used by open-system / resonance toys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x - np.mean(x)


def spectrum_rfft(x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple power spectrum: |FFT|^2 with rfft.
    Returns (freqs, power).
    """
    x = detrend(x)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    fft = np.fft.rfft(x)
    power = (np.abs(fft) ** 2) / max(n, 1)
    return freqs, power


@dataclass
class Peak:
    freq_hz: float
    power: float
    idx: int


def find_peak(freqs: np.ndarray, power: np.ndarray, *, fmin: float = 0.0, fmax: Optional[float] = None) -> Optional[Peak]:
    freqs = np.asarray(freqs)
    power = np.asarray(power)
    if fmax is None:
        fmax = float(freqs[-1]) if len(freqs) else 0.0

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None

    idxs = np.where(mask)[0]
    # avoid DC if present
    if len(idxs) > 1 and freqs[idxs[0]] == 0.0:
        idxs = idxs[1:]
        if len(idxs) == 0:
            return None

    best_local = idxs[np.argmax(power[idxs])]
    return Peak(freq_hz=float(freqs[best_local]), power=float(power[best_local]), idx=int(best_local))
