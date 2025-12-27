"""
TOY 01 — χ–BIO OPEN SYSTEM (OPEN QUANTUM SYSTEM TEST)

Purpose
-------
Tests whether a χ-like mediator can induce kHz-scale dynamics
in a biological coherence mode while remaining dynamically stable
and not trivially overdamped.

This toy is a *mechanistic stress test* of the SCSM χ-bridge idea.

What this toy DOES:
- Simulates a bio oscillator coupled to a χ oscillator
- Includes damping + stochastic driving on χ (environment proxy)
- Produces time series, spectra, parameter scans
- Enforces explicit pass/fail criteria

What this toy DOES NOT do:
- Claim biological realism
- Assume AQG or real spacetime foam
- Validate consciousness claims

Outputs
-------
- data/time_series.h5
- data/spectrum.csv
- data/scan_lambda_vs_peak.csv
- figures/q_vs_t.(png, svg)
- figures/spectrum.(png, svg)
- figures/scan_lambda_vs_peak.(png, svg)
- summary.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure, plot_time_series, plot_spectrum, plot_scan
from scsm_utils.fft_utils import spectrum_rfft, find_peak


# -----------------------------
# Parameters
# -----------------------------

@dataclass
class Toy01Params:
    # frequencies
    f_bio_hz: float = 3000.0
    f_chi_hz: float = 3000.0

    # damping and coupling
    gamma_chi: float = 50.0        # damping rate (1/s)
    lambda_chi: float = 0.02       # coupling strength (dimensionless toy)

    # noise
    noise_sigma: float = 1e-3

    # integration
    dt: float = 1e-5
    T: float = 1.0

    # scan
    lambda_scan: List[float] = None

    # pass/fail thresholds
    pass_f_min_hz: float = 1e3
    pass_f_max_hz: float = 1e4
    max_allowed_rms: float = 1.0


# -----------------------------
# Core dynamics
# -----------------------------

def simulate(params: Toy01Params, seed: int = 0):
    rng = np.random.default_rng(seed)

    n = int(params.T / params.dt)
    t = np.arange(n) * params.dt

    wb = 2 * math.pi * params.f_bio_hz
    wc = 2 * math.pi * params.f_chi_hz
    k = params.lambda_chi
    g = params.gamma_chi

    # state: [q, qdot, x, xdot]
    y = np.zeros(4)
    q_hist = np.zeros(n)
    x_hist = np.zeros(n)

    def step(y):
        q, qd, x, xd = y
        qdd = -wb * wb * q - k * (q - x)
        xdd = -wc * wc * x - k * (x - q) - 2 * g * xd
        xdd += params.noise_sigma * rng.normal()
        return np.array([qd, qdd, xd, xdd])

    stable = True
    for i in range(n):
        k1 = step(y)
        k2 = step(y + 0.5 * params.dt * k1)
        y = y + params.dt * k2

        if not np.all(np.isfinite(y)) or np.any(np.abs(y) > 1e6):
            stable = False
            q_hist = q_hist[: i + 1]
            x_hist = x_hist[: i + 1]
            t = t[: i + 1]
            break

        q_hist[i] = y[0]
        x_hist[i] = y[2]

    return t, q_hist, x_hist, stable


# -----------------------------
# Main execution
# -----------------------------

def main():
    params = Toy01Params(
        lambda_scan=np.linspace(0.0, 0.05, 12).tolist()
    )

    run = create_run(
        toy_slug="toy01_chi_bio_open_system",
        toy_name="TOY 01 — χ–BIO OPEN SYSTEM",
        description="Coupled oscillator + noise toy probing χ-mediated kHz dynamics.",
        params=asdict(params),
    )

    # ---- single run ----
    t, q, x, stable = simulate(params, seed=1)

    freqs, power = spectrum_rfft(q, params.dt)
    peak = find_peak(freqs, power, fmin=10.0, fmax=2e4)

    q_rms = float(np.sqrt(np.mean(q ** 2)))

    # ---- save raw data ----
    run.write_h5(
        "time_series.h5",
        arrays={"t": t, "q": q, "x": x},
        attrs={"stable": stable},
    )

    run.write_csv(
        "spectrum.csv",
        rows=[
            {"freq_hz": float(f), "power": float(p)}
            for f, p in zip(freqs, power)
        ],
    )

    # ---- plots ----
    fig_ts = plot_time_series(
        t,
        [("bio q(t)", q), ("chi x(t)", x)],
        title="Toy 01 — Time Series",
    )
    save_figure(fig_ts, run.figure_path("q_vs_t.png"))

    fig_sp = plot_spectrum(
        freqs,
        power,
        title="Toy 01 — Power Spectrum",
        f_mark=(params.pass_f_min_hz, params.pass_f_max_hz),
    )
    save_figure(fig_sp, run.figure_path("spectrum.png"))

    # ---- parameter scan ----
    scan_rows = []
    peak_freqs = []

    for lam in params.lambda_scan:
        params.lambda_chi = lam
        t_s, q_s, _, st = simulate(params, seed=2)
        freqs_s, power_s = spectrum_rfft(q_s, params.dt)
        pk = find_peak(freqs_s, power_s, fmin=10.0, fmax=2e4)
        pf = pk.freq_hz if pk else float("nan")
        scan_rows.append(
            {
                "lambda_chi": lam,
                "peak_freq_hz": pf,
                "stable": st,
            }
        )
        peak_freqs.append(pf)

    run.write_csv("scan_lambda_vs_peak.csv", scan_rows)

    fig_scan = plot_scan(
        np.array(params.lambda_scan),
        np.array(peak_freqs),
        title="Toy 01 — Peak Frequency vs λχ",
        xlabel="λχ",
        ylabel="Peak frequency (Hz)",
    )
    save_figure(fig_scan, run.figure_path("scan_lambda_vs_peak.png"))

    # ---- pass / fail ----
    passes = (
        stable
        and peak is not None
        and params.pass_f_min_hz <= peak.freq_hz <= params.pass_f_max_hz
        and q_rms <= params.max_allowed_rms
    )

    summary = {
        "status": "pass" if passes else "fail",
        "stable": stable,
        "peak_frequency_hz": peak.freq_hz if peak else None,
        "q_rms": q_rms,
        "criteria": {
            "frequency_band_hz": [params.pass_f_min_hz, params.pass_f_max_hz],
            "max_allowed_rms": params.max_allowed_rms,
        },
    }

    run.save_summary(summary)


if __name__ == "__main__":
    main()
