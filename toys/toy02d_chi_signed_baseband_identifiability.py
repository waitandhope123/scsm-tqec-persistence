"""
TOY 02d — χ IDENTIFIABILITY VIA SIGNED BASEBAND LOCK-IN (I/Q + dphi)

Fixes Toy02c issue:
- Do NOT rely on |z| = sqrt(I^2+Q^2) envelope (rectifies and can destroy coherence).
- Use signed baseband channels I(t), Q(t), and dphi/dt from complex baseband.

Pass condition:
1) Sanity check: adversarial AM injection MUST show high coherence with u(t) in I or Q.
2) χ FM model should show coherence with u(t) in dphi/dt (or I/Q if FM→AM converts through damping).
3) Independent nulls should NOT show high coherence.

Outputs:
- data/input_u.csv
- data/time_series_<model>.csv
- data/demod_<model>.csv (t, I, Q, phi, dphi_hz)
- data/coh_u_I_<model>.csv
- data/coh_u_Q_<model>.csv
- data/coh_u_dphi_<model>.csv
- figures/coherence_overlay_I.png
- figures/coherence_overlay_dphi.png
- summary.json
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


@dataclass
class Toy02dParams:
    fs_hz: float = 50_000.0
    duration_s: float = 10.0
    seed: int = 2029

    # Carrier / χ oscillator
    f0_hz: float = 3000.0
    Q_true: float = 80.0
    drive_sigma: float = 0.15
    meas_sigma: float = 0.05

    # Control u(t)
    f_mod_hz: float = 37.0
    mod_index: float = 0.10
    u_amp: float = 1.0

    # Nulls
    colored_beta: float = 1.0
    ar2_radius: float = 0.995

    # Adversarial AM injection (should PASS sanity check)
    adv_bb_gain: float = 0.5
    adv_bb_noise: float = 0.2

    # Demodulation
    bp_halfwidth_hz: float = 1200.0
    lp_cut_hz: float = 400.0

    # Coherence estimation (Welch)
    nperseg: int = 8192
    overlap: float = 0.5
    fwin_hz: float = 2.0

    # Thresholds
    coh_min_chi: float = 0.20         # χ can be weaker than crosstalk; start modest
    coh_min_adv: float = 0.60         # adversarial must be strong
    coh_null_max: float = 0.15        # independent nulls should be small
    peak_ratio_min: float = 4.0       # weaker than before; coherence absolute matters more


# ---------- Welch coherence (minimal, no SciPy) ----------

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

def local_median(arr: np.ndarray, center_idx: int, half_bins: int) -> float:
    a = max(0, center_idx - half_bins)
    b = min(len(arr), center_idx + half_bins + 1)
    window = arr[a:b]
    if len(window) <= 3:
        return float(np.median(window))
    mask = np.ones_like(window, dtype=bool)
    mask[min(center_idx - a, len(window)-1)] = False
    return float(np.median(window[mask]))


# ---------- Signal generators ----------

def generate_u(p: Toy02dParams, t: np.ndarray) -> np.ndarray:
    return p.u_amp * np.sin(2.0 * math.pi * p.f_mod_hz * t)

def simulate_chi_freqmod(p: Toy02dParams, rng: np.random.Generator, u: np.ndarray) -> np.ndarray:
    dt = 1.0 / p.fs_hz
    n = len(u)
    w0 = 2.0 * math.pi * p.f0_hz

    x = 0.0
    v = 0.0
    xs = np.zeros(n, dtype=float)

    for i in range(n):
        w = w0 * (1.0 + p.mod_index * float(u[i]))
        gamma = w / p.Q_true
        drive = p.drive_sigma * rng.normal()
        v = v + dt * (-gamma * v - (w**2) * x + drive)
        x = x + dt * v
        xs[i] = x + p.meas_sigma * rng.normal()

    return xs

def simulate_colored_independent(p: Toy02dParams, rng: np.random.Generator, n: int) -> np.ndarray:
    w = rng.normal(size=n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0/p.fs_hz)
    f[0] = f[1] if len(f) > 1 else 1.0
    shape = 1.0 / (f ** (p.colored_beta / 2.0))
    Y = W * shape
    y = np.fft.irfft(Y, n=n)
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    y = y + p.meas_sigma * rng.normal(size=n)
    return y

def simulate_ar2_independent(p: Toy02dParams, rng: np.random.Generator, n: int) -> np.ndarray:
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
    x = x + p.meas_sigma * rng.normal(size=n)
    return x

def simulate_adversarial_am(p: Toy02dParams, rng: np.random.Generator, u: np.ndarray) -> np.ndarray:
    """
    Signed baseband AM: bb(t) = gain*u(t) + noise, then x = bb(t)*cos(2πf0t).
    This SHOULD show high coherence with u in demod I/Q.
    """
    n = len(u)
    t = np.arange(n) / p.fs_hz
    bb = p.adv_bb_gain * u + p.adv_bb_noise * rng.normal(size=n)
    x = bb * np.cos(2.0 * math.pi * p.f0_hz * t)
    x = x + p.meas_sigma * rng.normal(size=n)
    return x


# ---------- Filtering & demod ----------

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
    return {"t": t, "I": I, "Q": Q, "phi": phi, "dphi_hz": dphi_hz, "M": np.array([M], dtype=int)}


# ---------- Decision metrics ----------

def coherence_metrics(u: np.ndarray, y: np.ndarray, p: Toy02dParams) -> Dict[str, float]:
    f, coh = coherence_ms(u, y, p.fs_hz, p.nperseg, p.overlap)
    i0 = idx_nearest(f, p.f_mod_hz)
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    half_bins = int(max(2, round(p.fwin_hz / df)))
    coh_at = float(coh[i0])
    loc_med = local_median(coh, i0, half_bins)
    peak_ratio = float((coh_at + 1e-12) / (loc_med + 1e-12))
    return {"coh_at_fmod": coh_at, "peak_ratio": peak_ratio, "local_median": loc_med, "f_bin_hz": float(f[i0]), "df_hz": df}


def main():
    p = Toy02dParams()
    run = create_run(
        toy_slug="toy02d_chi_signed_baseband_identifiability",
        toy_name="TOY 02d — χ identifiability via signed baseband I/Q",
        description="Signed baseband lock-in coherence: coherence(u, I/Q/dphi). Includes adversarial AM sanity check.",
        params=asdict(p),
    )

    n = int(p.duration_s * p.fs_hz)
    t = np.arange(n) / p.fs_hz
    u = generate_u(p, t)

    run.write_csv("input_u.csv", [{"t_s": float(tt), "u": float(uu)} for tt, uu in zip(t, u)])

    raw = {
        "A_chi_freqmod": simulate_chi_freqmod(p, np.random.default_rng(p.seed + 1), u),
        "B_colored_indep": simulate_colored_independent(p, np.random.default_rng(p.seed + 2), n),
        "C_ar2_indep": simulate_ar2_independent(p, np.random.default_rng(p.seed + 3), n),
        "D_adv_AM": simulate_adversarial_am(p, np.random.default_rng(p.seed + 4), u),
    }

    for name, x in raw.items():
        run.write_csv(f"time_series_{name}.csv", [{"t_s": float(tt), "x": float(xx)} for tt, xx in zip(t, x)])

    demod = {}
    for name, x in raw.items():
        xb = bandpass_fft(x, p.fs_hz, p.f0_hz, p.bp_halfwidth_hz)
        d = demod_iq(xb, p.fs_hz, p.f0_hz, p.lp_cut_hz)
        demod[name] = d
        run.write_csv(
            f"demod_{name}.csv",
            [
                {"t_s": float(tt), "I": float(ii), "Q": float(qq), "phi": float(ph), "dphi_hz": float(dfh)}
                for tt, ii, qq, ph, dfh in zip(d["t"], d["I"], d["Q"], d["phi"], d["dphi_hz"])
            ],
        )

    # Compute coherence metrics in channels
    metrics = {}
    for name, d in demod.items():
        metrics[name] = {
            "I": coherence_metrics(u, d["I"], p),
            "Q": coherence_metrics(u, d["Q"], p),
            "dphi": coherence_metrics(u, d["dphi_hz"], p),
        }

    # Save coherence spectra for plotting (I and dphi only to keep it light)
    coh_specs_I = {}
    coh_specs_dphi = {}
    for name, d in demod.items():
        fI, cI = coherence_ms(u, d["I"], p.fs_hz, p.nperseg, p.overlap)
        coh_specs_I[name] = (fI, cI)
        run.write_csv(f"coh_u_I_{name}.csv", [{"f_hz": float(ff), "coh": float(cc)} for ff, cc in zip(fI, cI)])

        fd, cd = coherence_ms(u, d["dphi_hz"], p.fs_hz, p.nperseg, p.overlap)
        coh_specs_dphi[name] = (fd, cd)
        run.write_csv(f"coh_u_dphi_{name}.csv", [{"f_hz": float(ff), "coh": float(cc)} for ff, cc in zip(fd, cd)])

    # Pass/fail
    def channel_ok(m: Dict[str, float], coh_min: float) -> bool:
        return (m["coh_at_fmod"] >= coh_min) and (m["peak_ratio"] >= p.peak_ratio_min)

    # 1) Sanity: adversarial must pass in I or Q
    adv_ok = channel_ok(metrics["D_adv_AM"]["I"], p.coh_min_adv) or channel_ok(metrics["D_adv_AM"]["Q"], p.coh_min_adv)

    # 2) χ should show some response (usually in dphi for FM)
    chi_ok = channel_ok(metrics["A_chi_freqmod"]["dphi"], p.coh_min_chi) or channel_ok(metrics["A_chi_freqmod"]["I"], p.coh_min_chi) or channel_ok(metrics["A_chi_freqmod"]["Q"], p.coh_min_chi)

    # 3) independent nulls should not
    nulls_ok = True
    for nm in ["B_colored_indep", "C_ar2_indep"]:
        if (metrics[nm]["I"]["coh_at_fmod"] > p.coh_null_max) or (metrics[nm]["dphi"]["coh_at_fmod"] > p.coh_null_max):
            nulls_ok = False

    passed = bool(adv_ok and chi_ok and nulls_ok)

    # Figures
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    for name, (fI, cI) in coh_specs_I.items():
        ax.plot(fI, cI, label=name)
    ax.axvline(p.f_mod_hz, linestyle="--", alpha=0.6)
    ax.set_xlim(0, max(5 * p.f_mod_hz, 400.0))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence(u, I)")
    ax.set_title("Toy 02d — Coherence with signed baseband I(t)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig1, run.figure_path("coherence_overlay_I.png"))

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for name, (fd, cd) in coh_specs_dphi.items():
        ax2.plot(fd, cd, label=name)
    ax2.axvline(p.f_mod_hz, linestyle="--", alpha=0.6)
    ax2.set_xlim(0, max(5 * p.f_mod_hz, 400.0))
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Coherence(u, dphi/dt)")
    ax2.set_title("Toy 02d — Coherence with phase-derivative dphi/dt")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    save_figure(fig2, run.figure_path("coherence_overlay_dphi.png"))

    summary = {
        "status": "pass" if passed else "fail",
        "interpretation": "pass=adversarial AM sanity check passes AND χ shows detectable signed-baseband coherence with control, while independent nulls do not",
        "criteria": {
            "f_mod_hz": p.f_mod_hz,
            "coh_min_adv": p.coh_min_adv,
            "coh_min_chi": p.coh_min_chi,
            "coh_null_max": p.coh_null_max,
            "peak_ratio_min": p.peak_ratio_min,
        },
        "sanity_adv_AM": {
            "adv_ok": adv_ok,
            "metrics_I": metrics["D_adv_AM"]["I"],
            "metrics_Q": metrics["D_adv_AM"]["Q"],
        },
        "chi_metrics": {
            "chi_ok": chi_ok,
            "I": metrics["A_chi_freqmod"]["I"],
            "Q": metrics["A_chi_freqmod"]["Q"],
            "dphi": metrics["A_chi_freqmod"]["dphi"],
        },
        "null_metrics": {
            "B_colored_indep": metrics["B_colored_indep"],
            "C_ar2_indep": metrics["C_ar2_indep"],
        },
        "note": "If sanity check fails, the demod/coherence pipeline is broken; do not interpret χ. If sanity passes but χ fails, χ coupling/observable is too weak under current parameters.",
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
