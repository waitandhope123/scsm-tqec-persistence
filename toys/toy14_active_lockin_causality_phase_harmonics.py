"""
Toy 14: Active lock-in causality, phase, and harmonic test

Purpose:
- Actively drive a system with a control signal u(t)
- Compare χ-mediated response vs null models
- Measure coherence, phase, causal lag, and harmonics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import json
import time
import uuid

# -----------------------------
# Parameters
# -----------------------------

@dataclass
class Toy14Params:
    fs_hz: float = 50_000.0
    t_sec: float = 2.0
    f_mod_hz: float = 37.0
    mod_index: float = 0.4
    meas_sigma: float = 0.15
    chi_gain: float = 1.0
    chi_lag_ms: float = 6.0
    chi_nonlinear: float = 0.3
    seed: int = 14001

    coh_min: float = 0.35
    coh_null_max: float = 0.12
    harm_db_min: float = 6.0
    lag_ms_max: float = 20.0


# -----------------------------
# Utilities
# -----------------------------

def coherence(x, y):
    X = np.fft.rfft(x)
    Y = np.fft.rfft(y)
    Pxy = X * np.conj(Y)
    Pxx = np.abs(X) ** 2
    Pyy = np.abs(Y) ** 2
    return np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-12)

def harmonic_ratio_db(x, fs, f0):
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    f1 = np.argmin(np.abs(freqs - f0))
    f2 = np.argmin(np.abs(freqs - 2 * f0))
    return 10 * np.log10((X[f2] + 1e-12) / (X[f1] + 1e-12))

def estimate_lag_ms(x, u, fs):
    corr = np.correlate(x, u, mode="full")
    lag = np.argmax(corr) - len(u) + 1
    return 1000 * lag / fs


# -----------------------------
# Signal models
# -----------------------------

def drive_signal(p, t):
    return np.sin(2 * np.pi * p.f_mod_hz * t)

def chi_response(p, u):
    lag_samples = int(p.chi_lag_ms * p.fs_hz / 1000)
    u_shift = np.roll(u, lag_samples)
    nonlinear = p.chi_nonlinear * np.sin(2 * np.pi * 2 * p.f_mod_hz * np.arange(len(u)) / p.fs_hz)
    return p.chi_gain * (u_shift + nonlinear)

def null_colored(u):
    return np.random.normal(0, 1, len(u))

def null_drift(u):
    return np.cumsum(np.random.normal(0, 0.01, len(u)))


# -----------------------------
# Main
# -----------------------------

def main():
    p = Toy14Params()
    rng = np.random.default_rng(p.seed)

    t = np.arange(0, p.t_sec, 1 / p.fs_hz)
    u = drive_signal(p, t)

    models = {
        "chi": chi_response(p, u),
        "colored": null_colored(u),
        "drift": null_drift(u),
    }

    results = []

    for name, x_raw in models.items():
        x = x_raw + rng.normal(0, p.meas_sigma, len(x_raw))

        coh = coherence(x, u)
        freqs = np.fft.rfftfreq(len(x), 1 / p.fs_hz)
        fbin = np.argmin(np.abs(freqs - p.f_mod_hz))

        coh_on = float(coh[fbin])
        coh_off = float(np.median(np.delete(coh, fbin)))
        coh_contrast = coh_on - coh_off

        phase_on = float(np.angle(np.fft.rfft(x)[fbin]))
        lag_ms = float(estimate_lag_ms(x, u, p.fs_hz))
        harm_db = float(harmonic_ratio_db(x, p.fs_hz, p.f_mod_hz))

        results.append({
            "model": name,
            "coh_on": coh_on,
            "coh_off": coh_off,
            "coh_contrast": coh_contrast,
            "phase_on_rad": phase_on,
            "lag_ms": lag_ms,
            "harmonic_ratio_db": harm_db
        })

    df = pd.DataFrame(results)

    # -----------------------------
    # Pass / fail logic
    # -----------------------------

    chi = df[df.model == "chi"].iloc[0]
    nulls = df[df.model != "chi"]

    passes = (
        chi.coh_on > p.coh_min
        and all(nulls.coh_on < p.coh_null_max)
        and abs(chi.lag_ms) < p.lag_ms_max
        and chi.harmonic_ratio_db > p.harm_db_min
    )

    # -----------------------------
    # Output
    # -----------------------------

    run_id = f"toy14_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out = Path("outputs") / "toy14_active_lockin" / run_id
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "data" / "metrics.csv", index=False)

    summary = {
        "status": "pass" if passes else "fail",
        "interpretation": (
            "pass=χ shows causal, phase-locked, nonlinear response not reproducible by nulls"
            if passes else
            "fail=χ response not sufficiently distinct under current SNR / coupling"
        ),
        "metrics": df.to_dict(orient="records"),
        "params": vars(p)
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
