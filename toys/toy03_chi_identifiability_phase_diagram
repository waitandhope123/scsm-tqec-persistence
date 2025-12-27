"""
TOY 03 — χ IDENTIFIABILITY PHASE DIAGRAM (COHERENCE-BASED, WITH NULL MARGIN)

Why this toy
------------
Toy02/02b/02c/02d showed: at a single parameter setting, χ is not identifiable.
Toy03 turns that into *useful constraints* by mapping "detectable vs not detectable"
across a sweep of coupling + noise parameters, using a sane coherence metric.

Key output of Toy03
-------------------
A "detectability phase diagram" in terms of:
  - modulation depth (mod_index)
  - measurement noise (meas_sigma)
(optionally also drive_sigma / Q_true, but we keep default manageable)

Metric
------
We inject u(t) at f_mod and simulate:
  A) χ FM oscillator: w(t) = w0*(1 + mod_index*u(t))
  B) Null: colored noise independent of u(t)
  C) Null: AR(2) pseudo-resonance independent of u(t)

We demod around f0 to get signed baseband I(t) and phase derivative dphi_hz,
then compute coherence at f_mod:
  coh_chi = max( coh(u, I), coh(u, dphi_hz) )
  coh_null = max( coh(u, I)_nulls and coh(u, dphi_hz)_nulls )

Define margin:
  margin = coh_chi - coh_null

PASS at a grid point if:
  coh_chi >= coh_abs_min  AND  margin >= margin_min

Outputs
-------
- data/grid_results.csv
  columns: mod_index, meas_sigma, coh_chi, coh_null, margin, pass
- data/sweep_details_mod<...>_meas<...>.csv (optional, can be large; default off)
- figures/heatmap_coh_chi.(png, svg)
- figures/heatmap_margin.(png, svg)
- figures/heatmap_pass.(png, svg)
- summary.json

Runtime
-------
Designed to be "minutes or less" on a laptop by using:
- moderate duration and segment sizes
- modest grid sizes (editable)

"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


# -----------------------------
# Parameters
# -----------------------------

@dataclass
class Toy03Params:
    # Sampling (kept smaller for sweeps)
    fs_hz: float = 25_000.0
    duration_s: float = 6.0
    seed: int = 2030

    # Carrier / χ oscillator baseline
    f0_hz: float = 3000.0
    Q_true: float = 80.0
    drive_sigma: float = 0.12

    # Control u(t)
    f_mod_hz: float = 37.0
    u_amp: float = 1.0

    # Nulls
    colored_beta: float = 1.0
    ar2_radius: float = 0.995

    # Demodulation
    bp_halfwidth_hz: float = 1200.0
    lp_cut_hz: float = 400.0

    # Coherence estimation
    nperseg: int = 4096
    overlap: float = 0.5
    fwin_hz: float = 2.0

    # Sweep grid
    mod_index_values: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)
    meas_sigma_values: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)

    # Decision thresholds (edit as desired)
    coh_abs_min: float = 0.20
    margin_min: float = 0.15

    # Save per-point detail CSVs (can be large)
    save_per_point_details: bool = False


# -----------------------------
# Welch coherence (minimal)
# -----------------------------

def welch_cpsd(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int, overlap: float):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    step = int(nperseg * (1.0 - overlap))
    step = max(1, step)

    if n < nperseg:
        pad = nperseg - n
        x = np.pad(x, (0, pad))
        y = np.pad(y, (0, pad))
        n = len(x)

    w = np.hanning(nperseg)
    wnorm = np.sum(w**2)

    acc = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        xs = x[start:start+nperseg] - np.mean(x[start:start+nperseg])
        ys = y[start:start+nperseg] - np.mean(y[start:start+nperseg])
        X = np.fft.rfft(xs * w)
        Y = np.fft.rfft(ys * w)
        Pxy = (X * np.conj(Y)) / (fs * wnorm)
        acc = Pxy if acc is None else (acc + Pxy)
        count += 1

    if acc is None:
        acc = np.zeros((nperseg//2 + 1,), dtype=complex)
        count = 1

    Pxy = acc / count
    f = np.fft.rfftfreq(nperseg, d=1.0/fs)
    return f, Pxy

def coherence_ms(u: np.ndarray, x: np.ndarray, fs: float, nperseg: int, overlap: float):
    f, Puu = welch_cpsd(u, u, fs, nperseg, overlap)
    _f2, Pxx = welch_cpsd(x, x, fs, nperseg, overlap)
    _f3, Pux = welch_cpsd(u, x, fs, nperseg, overlap)
    coh = (np.abs(Pux)**2) / ((np.real(Puu) + 1e-30) * (np.real(Pxx) + 1e-30))
    return f, coh

def idx_nearest(f: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(f - target)))

def coherence_at_fmod(u: np.ndarray, y: np.ndarray, p: Toy03Params) -> float:
    f, coh = coherence_ms(u, y, p.fs_hz, p.nperseg, p.overlap)
    i0 = idx_nearest(f, p.f_mod_hz)
    return float(coh[i0])


# -----------------------------
# Signals
# -----------------------------

def generate_u(p: Toy03Params, t: np.ndarray) -> np.ndarray:
    return p.u_amp * np.sin(2.0 * math.pi * p.f_mod_hz * t)

def simulate_chi_freqmod(p: Toy03Params, rng: np.random.Generator, u: np.ndarray, mod_index: float, meas_sigma: float) -> np.ndarray:
    dt = 1.0 / p.fs_hz
    n = len(u)
    w0 = 2.0 * math.pi * p.f0_hz

    x = 0.0
    v = 0.0
    xs = np.zeros(n, dtype=float)
    for i in range(n):
        w = w0 * (1.0 + mod_index * float(u[i]))
        gamma = w / p.Q_true
        drive = p.drive_sigma * rng.normal()
        v = v + dt * (-gamma * v - (w**2) * x + drive)
        x = x + dt * v
        xs[i] = x + meas_sigma * rng.normal()
    return xs

def simulate_colored_independent(p: Toy03Params, rng: np.random.Generator, n: int, meas_sigma: float) -> np.ndarray:
    w = rng.normal(size=n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0/p.fs_hz)
    f[0] = f[1] if len(f) > 1 else 1.0
    shape = 1.0 / (f ** (p.colored_beta / 2.0))
    Y = W * shape
    y = np.fft.irfft(Y, n=n)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    y = y + meas_sigma * rng.normal(size=n)
    return y

def simulate_ar2_independent(p: Toy03Params, rng: np.random.Generator, n: int, meas_sigma: float) -> np.ndarray:
    theta = 2.0 * math.pi * (p.f0_hz / p.fs_hz)
    r = p.ar2_radius
    a1 = 2.0 * r * math.cos(theta)
    a2 = -(r**2)

    x = np.zeros(n, dtype=float)
    e = rng.normal(size=n)
    x[0] = e[0]
    x[1] = a1 * x[0] + e[1]
    for i in range(2, n):
        x[i] = a1 * x[i-1] + a2 * x[i-2] + e[i]

    x = (x - np.mean(x)) / (np.std(x) + 1e-12)
    x = x + meas_sigma * rng.normal(size=n)
    return x


# -----------------------------
# Demod
# -----------------------------

def bandpass_fft(x: np.ndarray, fs: float, f0: float, halfwidth: float) -> np.ndarray:
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (f >= (f0 - halfwidth)) & (f <= (f0 + halfwidth))
    X2 = np.zeros_like(X)
    X2[mask] = X[mask]
    y = np.fft.irfft(X2, n=n)
    return y

def moving_average(x: np.ndarray, M: int) -> np.ndarray:
    if M <= 1:
        return x
    k = np.ones(M) / M
    return np.convolve(x, k, mode="same")

def demod_iq(x: np.ndarray, fs: float, f0: float, lp_cut_hz: float) -> Dict[str, np.ndarray]:
    n = len(x)
    t = np.arange(n) / fs
    loI = np.cos(2.0 * math.pi * f0 * t)
    loQ = -np.sin(2.0 * math.pi * f0 * t)

    I = x * loI
    Q = x * loQ
    M = int(max(3, round(fs / lp_cut_hz)))
    I = moving_average(I, M)
    Q = moving_average(Q, M)

    phi = np.unwrap(np.arctan2(Q, I))
    dphi_hz = np.gradient(phi) * fs / (2.0 * math.pi)
    return {"I": I, "Q": Q, "dphi_hz": dphi_hz, "M": np.array([M], dtype=int)}


# -----------------------------
# Main sweep
# -----------------------------

def main():
    p = Toy03Params()
    run = create_run(
        toy_slug="toy03_chi_identifiability_phase_diagram",
        toy_name="TOY 03 — χ identifiability phase diagram",
        description="Sweep mod_index and meas_sigma; compute coherence(u, I/dphi) margin vs nulls; output heatmaps + grid CSV.",
        params=asdict(p),
    )

    n = int(p.duration_s * p.fs_hz)
    t = np.arange(n) / p.fs_hz
    u = generate_u(p, t)

    run.write_csv("input_u.csv", [{"t_s": float(tt), "u": float(uu)} for tt, uu in zip(t, u)])

    mod_vals = list(p.mod_index_values)
    meas_vals = list(p.meas_sigma_values)

    grid_rows: List[Dict[str, float]] = []
    coh_chi_mat = np.zeros((len(meas_vals), len(mod_vals)), dtype=float)
    margin_mat = np.zeros_like(coh_chi_mat)
    pass_mat = np.zeros_like(coh_chi_mat)

    base = p.seed
    for i_ms, meas_sigma in enumerate(meas_vals):
        for j_m, mod_index in enumerate(mod_vals):
            # deterministic per-point RNG streams
            rngA = np.random.default_rng(base + 10_000 + i_ms * 100 + j_m)
            rngB = np.random.default_rng(base + 20_000 + i_ms * 100 + j_m)
            rngC = np.random.default_rng(base + 30_000 + i_ms * 100 + j_m)

            # Simulate
            xA = simulate_chi_freqmod(p, rngA, u, mod_index=mod_index, meas_sigma=meas_sigma)
            xB = simulate_colored_independent(p, rngB, n, meas_sigma=meas_sigma)
            xC = simulate_ar2_independent(p, rngC, n, meas_sigma=meas_sigma)

            # Demod all
            xAb = bandpass_fft(xA, p.fs_hz, p.f0_hz, p.bp_halfwidth_hz)
            xBb = bandpass_fft(xB, p.fs_hz, p.f0_hz, p.bp_halfwidth_hz)
            xCb = bandpass_fft(xC, p.fs_hz, p.f0_hz, p.bp_halfwidth_hz)

            dA = demod_iq(xAb, p.fs_hz, p.f0_hz, p.lp_cut_hz)
            dB = demod_iq(xBb, p.fs_hz, p.f0_hz, p.lp_cut_hz)
            dC = demod_iq(xCb, p.fs_hz, p.f0_hz, p.lp_cut_hz)

            # Coherence at f_mod for channels
            cohA_I = coherence_at_fmod(u, dA["I"], p)
            cohA_dphi = coherence_at_fmod(u, dA["dphi_hz"], p)
            coh_chi = float(max(cohA_I, cohA_dphi))

            cohB_I = coherence_at_fmod(u, dB["I"], p)
            cohB_dphi = coherence_at_fmod(u, dB["dphi_hz"], p)

            cohC_I = coherence_at_fmod(u, dC["I"], p)
            cohC_dphi = coherence_at_fmod(u, dC["dphi_hz"], p)

            coh_null = float(max(cohB_I, cohB_dphi, cohC_I, cohC_dphi))
            margin = float(coh_chi - coh_null)

            passed = (coh_chi >= p.coh_abs_min) and (margin >= p.margin_min)

            coh_chi_mat[i_ms, j_m] = coh_chi
            margin_mat[i_ms, j_m] = margin
            pass_mat[i_ms, j_m] = 1.0 if passed else 0.0

            grid_rows.append(
                {
                    "mod_index": float(mod_index),
                    "meas_sigma": float(meas_sigma),
                    "coh_chi": float(coh_chi),
                    "coh_chi_I": float(cohA_I),
                    "coh_chi_dphi": float(cohA_dphi),
                    "coh_null": float(coh_null),
                    "coh_null_colored_I": float(cohB_I),
                    "coh_null_colored_dphi": float(cohB_dphi),
                    "coh_null_ar2_I": float(cohC_I),
                    "coh_null_ar2_dphi": float(cohC_dphi),
                    "margin": float(margin),
                    "pass": float(1.0 if passed else 0.0),
                }
            )

            if p.save_per_point_details:
                # Store demod time series for this grid point (can get big)
                rel = f"details_mod{mod_index:.3f}_meas{meas_sigma:.3f}.csv"
                run.write_csv(
                    rel,
                    [
                        {
                            "t_s": float(tt),
                            "u": float(uu),
                            "I_chi": float(ii),
                            "dphi_chi": float(dd),
                            "I_colored": float(iib),
                            "dphi_colored": float(ddb),
                            "I_ar2": float(iic),
                            "dphi_ar2": float(ddc),
                        }
                        for tt, uu, ii, dd, iib, ddb, iic, ddc in zip(
                            t, u, dA["I"], dA["dphi_hz"], dB["I"], dB["dphi_hz"], dC["I"], dC["dphi_hz"]
                        )
                    ],
                )

    run.write_csv("grid_results.csv", grid_rows)

    # -----------------------------
    # Heatmap figures
    # -----------------------------
    # axes: x=mod_index, y=meas_sigma
    x = np.array(mod_vals, dtype=float)
    y = np.array(meas_vals, dtype=float)

    def heatmap(mat: np.ndarray, title: str, fname: str, vmin=None, vmax=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[x.min(), x.max(), y.min(), y.max()],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("mod_index")
        ax.set_ylabel("meas_sigma")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.9)
        save_figure(fig, run.figure_path(fname))

    heatmap(coh_chi_mat, "Toy 03 — coh_chi (max of I/dphi)", "heatmap_coh_chi.png", vmin=0.0, vmax=1.0)
    heatmap(margin_mat, "Toy 03 — margin = coh_chi - coh_null", "heatmap_margin.png", vmin=float(np.min(margin_mat)), vmax=float(np.max(margin_mat)))
    heatmap(pass_mat, "Toy 03 — PASS map (1=pass, 0=fail)", "heatmap_pass.png", vmin=0.0, vmax=1.0)

    # -----------------------------
    # Summary
    # -----------------------------
    # Find best margin point and best coh_chi point
    best_margin_idx = np.unravel_index(int(np.argmax(margin_mat)), margin_mat.shape)
    best_coh_idx = np.unravel_index(int(np.argmax(coh_chi_mat)), coh_chi_mat.shape)

    best_margin = {
        "meas_sigma": float(meas_vals[best_margin_idx[0]]),
        "mod_index": float(mod_vals[best_margin_idx[1]]),
        "margin": float(margin_mat[best_margin_idx]),
        "coh_chi": float(coh_chi_mat[best_margin_idx]),
    }
    best_coh = {
        "meas_sigma": float(meas_vals[best_coh_idx[0]]),
        "mod_index": float(mod_vals[best_coh_idx[1]]),
        "coh_chi": float(coh_chi_mat[best_coh_idx]),
        "margin": float(margin_mat[best_coh_idx]),
    }

    # Does there exist any pass point?
    any_pass = bool(np.any(pass_mat > 0.5))

    summary = {
        "status": "pass" if any_pass else "fail",
        "interpretation": "pass=there exists a region in (mod_index, meas_sigma) where χ is coherence-identifiable above null margin; fail=no such region under this toy sweep",
        "criteria": {"coh_abs_min": p.coh_abs_min, "margin_min": p.margin_min},
        "grid": {"n_mod": len(mod_vals), "n_meas": len(meas_vals), "mod_index_values": mod_vals, "meas_sigma_values": meas_vals},
        "best_points": {"best_margin": best_margin, "best_coh_chi": best_coh},
        "notes": [
            "If FAIL: either χ coupling (mod_index) must be larger than swept, or noise must be lower, or you need a different coupling channel than FM-only.",
            "If PASS: the pass region defines a quantitative detectability requirement you can map onto instrument noise floors / coupling estimates.",
        ],
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
