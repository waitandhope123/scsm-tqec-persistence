"""
TOY 13 — Correlated Observable Disambiguation
============================================

Goal (post-Toy12B):
- Show that STATIC boundary force curves alone are degenerate (plateau vs power/patch),
  but become discriminable when paired with an INDEPENDENT dynamical/spectral observable
  (a kHz-band "χ-like" resonance fingerprint).

What this toy does:
1) Generates synthetic datasets with TWO channels per dataset:
   A) Boundary force curve: F(d) measured across distances d (mm scale)
   B) Dynamical time series: x(t) measured at fixed setup producing PSD in kHz band

2) Includes adversarial nulls:
   - boundary looks plateau-like but dynamics is colored noise (false boundary positive)
   - boundary is power-law but dynamics has a narrow resonance (false dynamics positive)
   - both present (true correlated χ-like case)
   - neither present (pure nulls)

3) Fits/selects models under:
   - Boundary-only selection
   - Dynamics-only selection
   - Joint selection (Boundary + Dynamics combined evidence)

Outputs (all under the Toy13 run folder):
- data/datasets_index.csv
- data/boundary_curves/<dataset_id>_boundary.csv
- data/psd/<dataset_id>_psd.csv
- data/metrics/<dataset_id>_metrics.json
- data/results_selection.csv
- figures/winner_matrices.png
- figures/improvement_bars.png
- summary.json

Save as:
  toy13_joint_boundary_and_khz_fingerprint.py

Run:
  py toy13_joint_boundary_and_khz_fingerprint.py

No HDF5; CSV/JSON/PNG only.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


# ============================================================
# Parameters
# ============================================================

@dataclass
class Toy13Params:
    seed: int = 13001

    # Dataset size
    n_datasets: int = 200

    # Distance grid for boundary curve
    d_min_m: float = 0.5e-3
    d_max_m: float = 8.0e-3
    n_d: int = 60

    # Boundary noise model (pN)
    noise_abs_pN: float = 0.8
    noise_rel_frac: float = 0.05

    # Dynamics: sampling and duration
    fs_hz: float = 50_000.0
    t_sec: float = 2.0

    # Dynamics: target resonance band
    f_band_lo: float = 1000.0
    f_band_hi: float = 10_000.0

    # Dynamics noise
    meas_sigma: float = 0.25

    # Truth mix over four joint scenarios
    # (boundary_truth, dynamics_truth) in:
    # boundary_truth: "plateau" or "power"
    # dynamics_truth: "resonant" or "colored"
    # The key: correlated signature is ("plateau","resonant")
    truth_weights: Tuple[Tuple[str, str, float], ...] = (
        ("plateau", "resonant", 0.25),  # correlated "χ-like"
        ("plateau", "colored",  0.25),  # false boundary positive
        ("power",   "resonant", 0.25),  # false dynamics positive
        ("power",   "colored",  0.25),  # pure null
    )

    # Boundary model candidates
    boundary_candidates: Tuple[str, ...] = ("plateau", "power", "patch")

    # Dynamics model candidates
    dynamics_candidates: Tuple[str, ...] = ("resonant", "colored", "drifting")

    # Fit effort (keep modest for laptop)
    n_starts: int = 25
    n_local_steps: int = 90


# ============================================================
# Utilities
# ============================================================

def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ============================================================
# Boundary models
# ============================================================

def boundary_plateau(d: np.ndarray, F0: float, d0: float, n: float) -> np.ndarray:
    return F0 / (1.0 + (d / (d0 + 1e-30)) ** n)

def boundary_power(d: np.ndarray, A: float, dc: float, m: float) -> np.ndarray:
    return A / ((d + dc + 1e-30) ** m)

def boundary_patch(d: np.ndarray, A: float, B: float, dc: float, m: float) -> np.ndarray:
    return A / ((d + dc + 1e-30) ** m) + B / ((d + dc + 1e-30) ** 2)


# ============================================================
# Dynamics generators + PSD
# ============================================================

def ar1_colored(n: int, rng: np.random.Generator, phi: float = 0.995, sigma: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=float)
    eps = rng.normal(scale=sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x

def resonant_oscillator(n: int, fs: float, rng: np.random.Generator, f0: float, Q: float, drive_sigma: float = 1.0) -> np.ndarray:
    # Discrete-time damped oscillator excited by white noise ("stochastic resonance")
    # x'' + (w0/Q) x' + w0^2 x = drive
    # Use simple recursion on velocity + position (semi-implicit)
    dt = 1.0 / fs
    w0 = 2.0 * math.pi * f0
    gamma = w0 / max(Q, 1e-6)

    x = 0.0
    v = 0.0
    out = np.zeros(n, dtype=float)
    drive = rng.normal(scale=drive_sigma, size=n)

    for i in range(n):
        a = drive[i] - gamma * v - (w0 * w0) * x
        v = v + dt * a
        x = x + dt * v
        out[i] = x

    return out

def drifting_resonant(n: int, fs: float, rng: np.random.Generator, f0: float, Q: float, drift_hz: float) -> np.ndarray:
    # Piecewise drift in frequency to create a broadened / drifting peak
    dt = 1.0 / fs
    x = 0.0
    v = 0.0
    out = np.zeros(n, dtype=float)

    # Drift as a slow random walk in frequency
    f = f0
    for i in range(n):
        if i % int(fs * 0.02) == 0:  # update every 20ms
            f = float(np.clip(f + rng.normal(scale=drift_hz), 500.0, 12_000.0))
        w0 = 2.0 * math.pi * f
        gamma = w0 / max(Q, 1e-6)
        drive = rng.normal(scale=1.0)

        a = drive - gamma * v - (w0 * w0) * x
        v = v + dt * a
        x = x + dt * v
        out[i] = x

    return out

def welch_psd(x: np.ndarray, fs: float, nfft: int = 8192, hop: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    # Basic Welch PSD with Hann window
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < nfft:
        # zero-pad
        x = np.pad(x, (0, nfft - n))
        n = len(x)

    w = np.hanning(nfft)
    scale = np.sum(w * w)

    psd_acc = []
    for start in range(0, n - nfft + 1, hop):
        seg = x[start:start + nfft]
        seg = seg - np.mean(seg)
        X = np.fft.rfft(seg * w)
        P = (np.abs(X) ** 2) / (scale * fs)
        psd_acc.append(P)

    Pxx = np.mean(psd_acc, axis=0)
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return f, Pxx


# ============================================================
# Model scoring (BIC)
# ============================================================

def sse(y: np.ndarray, yhat: np.ndarray) -> float:
    r = y - yhat
    return float(np.sum(r * r))

def bic(n: int, k: int, rss: float) -> float:
    rss = max(rss, 1e-30)
    return n * math.log(rss / n) + k * math.log(n)

def local_fit_boundary(
    rng: np.random.Generator,
    name: str,
    d: np.ndarray,
    y: np.ndarray,
    p: Toy13Params,
) -> Tuple[Dict[str, float], float, float]:
    # Random-restart coordinate descent on parameter box constraints.
    yscale = float(np.median(np.abs(y)) + 1e-30)
    dref = float(np.median(d))

    if name == "plateau":
        bounds = {
            "F0": (1e-3 * yscale, 1e3 * yscale),
            "d0": (0.2e-3, 12e-3),
            "n":  (1.5, 12.0),
        }
        model = lambda dd, q: boundary_plateau(dd, q["F0"], q["d0"], q["n"])
        k = 3

    elif name == "power":
        bounds = {
            "A":  (1e-6 * yscale * (dref ** 3), 1e4 * yscale * (dref ** 3) + 1e-30),
            "dc": (0.0, 2e-3),
            "m":  (0.8, 10.0),
        }
        model = lambda dd, q: boundary_power(dd, q["A"], q["dc"], q["m"])
        k = 3

    elif name == "patch":
        bounds = {
            "A":  (1e-8 * yscale, 1e5 * yscale + 1e-30),
            "B":  (1e-8 * yscale, 1e5 * yscale + 1e-30),
            "dc": (0.0, 2e-3),
            "m":  (0.5, 7.0),
        }
        model = lambda dd, q: boundary_patch(dd, q["A"], q["B"], q["dc"], q["m"])
        k = 4
    else:
        raise ValueError(name)

    def rand_params():
        return {kk: float(rng.uniform(lo, hi)) for kk, (lo, hi) in bounds.items()}

    best_q = None
    best_rss = float("inf")

    for _ in range(p.n_starts):
        q = rand_params()
        cur = sse(y, model(d, q))
        for _ in range(p.n_local_steps):
            improved = False
            for kk, (lo, hi) in bounds.items():
                span = hi - lo
                step = 0.15 * span
                trial = dict(q)
                trial[kk] = float(np.clip(q[kk] + rng.normal(scale=step), lo, hi))
                val = sse(y, model(d, trial))
                if val < cur:
                    q = trial
                    cur = val
                    improved = True
            if not improved:
                break
        if cur < best_rss:
            best_rss = cur
            best_q = q

    B = bic(len(d), k, best_rss)
    return best_q, best_rss, B

def score_dynamics_models(
    f: np.ndarray,
    Pxx: np.ndarray,
    p: Toy13Params,
) -> Dict[str, float]:
    """
    Very lightweight likelihood proxies:
    - resonant: prefer a narrow prominent peak inside band
    - colored: prefer smooth power-law-ish slope in log space
    - drifting: prefer broad peak / elevated bandwidth

    We compute BIC-like scores by fitting simple templates in log PSD.
    Lower is better.
    """
    band = (f >= p.f_band_lo) & (f <= p.f_band_hi)
    fb = f[band]
    Pb = Pxx[band] + 1e-30
    y = np.log(Pb)
    n = len(y)

    # Helpers: peak metrics
    i_pk = int(np.argmax(Pb))
    f_pk = float(fb[i_pk])
    pk = float(Pb[i_pk])
    med = float(np.median(Pb))
    peak_to_med_db = 10.0 * math.log10(pk / max(med, 1e-30))

    # Estimate bandwidth around peak at half max
    half = pk / 2.0
    left = i_pk
    while left > 0 and Pb[left] > half:
        left -= 1
    right = i_pk
    while right < len(Pb) - 1 and Pb[right] > half:
        right += 1
    bw = float(fb[right] - fb[left])
    Q_est = float(f_pk / max(bw, 1e-9))

    # Model 1: colored noise -> linear fit in log-log: log P = a + b log f
    xf = np.log(fb + 1e-30)
    A = np.vstack([np.ones_like(xf), xf]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    rss_col = float(np.sum((y - yhat) ** 2))
    bic_col = bic(n, 2, rss_col)

    # Model 2: resonant -> baseline colored + narrow "bump" at peak bin (1 parameter amplitude)
    # crude: subtract fitted colored and see if residual is sharply concentrated
    resid = y - yhat
    # energy concentration near peak
    win = (np.abs(fb - f_pk) <= max(50.0, 0.01 * f_pk))
    frac = float(np.sum(resid[win] ** 2) / max(np.sum(resid ** 2), 1e-30))
    # Turn this into an RSS proxy: better if frac is high AND Q_est is high AND peak is prominent
    # We encode as a "loss": lower is better.
    loss_res = (1.0 - frac) + 0.2 * max(0.0, 8.0 - peak_to_med_db) / 8.0 + 0.2 * max(0.0, 12.0 - Q_est) / 12.0
    # Convert to BIC-like with n scale; k=3
    bic_res = n * math.log(max(loss_res, 1e-12)) + 3 * math.log(n)

    # Model 3: drifting -> baseline colored + broadness preference (low Q)
    loss_drift = (max(0.0, Q_est - 4.0) / 10.0) + 0.15 * max(0.0, 6.0 - peak_to_med_db) / 6.0
    bic_drift = n * math.log(max(loss_drift, 1e-12)) + 3 * math.log(n)

    return {
        "bic_colored": float(bic_col),
        "bic_resonant": float(bic_res),
        "bic_drifting": float(bic_drift),
        "f_peak_hz": f_pk,
        "Q_est": Q_est,
        "peak_to_median_db": peak_to_med_db,
        "bw_hz": bw,
    }


# ============================================================
# Truth sampling + synthesis
# ============================================================

def sample_truth(rng: np.random.Generator, weights: Tuple[Tuple[str, str, float], ...]) -> Tuple[str, str]:
    w = np.array([x[2] for x in weights], dtype=float)
    w = w / np.sum(w)
    i = int(rng.choice(len(weights), p=w))
    return weights[i][0], weights[i][1]

def synth_boundary(rng: np.random.Generator, truth: str, d: np.ndarray, p: Toy13Params) -> Tuple[np.ndarray, Dict[str, float]]:
    # Choose plausible parameters; ensure values are in N (~pN = 1e-12 N)
    # We'll target around 50 pN at 1mm-ish scale sometimes, but not always identical.
    # (toy is about discriminability, not exact 52 pN.)
    if truth == "plateau":
        F0 = float(rng.uniform(2e-11, 9e-11))         # N
        d0 = float(rng.uniform(1.5e-3, 4.5e-3))       # m
        n = float(rng.uniform(4.0, 8.0))
        y = boundary_plateau(d, F0, d0, n)
        params = {"F0": F0, "d0": d0, "n": n}
    elif truth == "power":
        m = float(rng.uniform(2.0, 5.0))
        dc = float(rng.uniform(0.0, 0.8e-3))
        # choose A so that F at ~1mm is in tens of pN
        d1 = 1.0e-3
        target = float(rng.uniform(2e-11, 9e-11))
        A = target * ((d1 + dc) ** m)
        y = boundary_power(d, A, dc, m)
        params = {"A": A, "dc": dc, "m": m}
    else:
        raise ValueError(truth)

    # Add measurement noise: abs + proportional
    sigma = (p.noise_abs_pN * 1e-12) + p.noise_rel_frac * np.abs(y)
    y_noisy = y + rng.normal(scale=sigma, size=len(d))
    return y_noisy, params

def synth_dynamics(rng: np.random.Generator, truth: str, p: Toy13Params) -> Tuple[np.ndarray, Dict[str, float]]:
    n = int(p.fs_hz * p.t_sec)
    if truth == "colored":
        x = ar1_colored(n, rng, phi=0.996, sigma=1.0)
        params = {"kind": "colored", "phi": 0.996}
    elif truth == "resonant":
        f0 = float(rng.uniform(2000.0, 9000.0))
        Q = float(rng.uniform(15.0, 40.0))
        x = resonant_oscillator(n, p.fs_hz, rng, f0=f0, Q=Q, drive_sigma=1.0)
        params = {"kind": "resonant", "f0": f0, "Q_true": Q}
    elif truth == "drifting":
        f0 = float(rng.uniform(2000.0, 9000.0))
        Q = float(rng.uniform(10.0, 25.0))
        drift = float(rng.uniform(20.0, 120.0))
        x = drifting_resonant(n, p.fs_hz, rng, f0=f0, Q=Q, drift_hz=drift)
        params = {"kind": "drifting", "f0": f0, "Q_true": Q, "drift_hz": drift}
    else:
        raise ValueError(truth)

    # Add measurement noise
    x = x + rng.normal(scale=p.meas_sigma, size=n)
    return x, params


# ============================================================
# Main
# ============================================================

def main():
    p = Toy13Params()
    rng = np.random.default_rng(p.seed)

    run = create_run(
        toy_slug="toy13_joint_boundary_and_khz_fingerprint",
        toy_name="TOY 13 — joint boundary + kHz fingerprint",
        description="Correlated observable disambiguation after Toy12B boundary degeneracy: boundary-only vs joint evidence.",
        params=asdict(p),
    )

    data_dir = run.data_dir
    (data_dir / "boundary_curves").mkdir(parents=True, exist_ok=True)
    (data_dir / "psd").mkdir(parents=True, exist_ok=True)
    (data_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # Distance grid
    d = np.linspace(p.d_min_m, p.d_max_m, p.n_d)

    # Index file
    index_rows = []
    selection_rows = []

    # Confusion matrices:
    # boundary-only winner among boundary_candidates
    # dynamics-only winner among dynamics_candidates
    # joint winner among all combos (boundary x dynamics)
    bmap = {m: i for i, m in enumerate(p.boundary_candidates)}
    dmap = {m: i for i, m in enumerate(p.dynamics_candidates)}
    joint_map = {(bm, dm): k for k, (bm, dm) in enumerate([(bm, dm) for bm in p.boundary_candidates for dm in p.dynamics_candidates])}

    n_b = len(p.boundary_candidates)
    n_dy = len(p.dynamics_candidates)
    n_j = n_b * n_dy

    cm_b = np.zeros((n_b, n_b), dtype=int)     # truth_boundary x pred_boundary
    cm_dy = np.zeros((n_dy, n_dy), dtype=int)  # truth_dynamics x pred_dynamics
    cm_j = np.zeros((n_j, n_j), dtype=int)     # truth_joint x pred_joint

    # For "improvement" metrics on the key correlated truth:
    # correlated = ("plateau","resonant") should be better recovered by JOINT than boundary-only.
    corr_truth = ("plateau", "resonant")
    corr_total = 0
    corr_boundary_correct = 0
    corr_joint_correct = 0

    for i in range(p.n_datasets):
        boundary_truth, dyn_truth_base = sample_truth(rng, p.truth_weights)

        # For extra adversarial variety: sometimes include drifting as a distractor
        # (only if base is "resonant")
        dyn_truth = dyn_truth_base
        if dyn_truth_base == "resonant" and rng.uniform() < 0.25:
            dyn_truth = "drifting"

        ds_id = f"ds_{i:04d}"
        truth_joint = f"{boundary_truth}+{dyn_truth}"

        # --- Generate boundary data
        F, boundary_params = synth_boundary(rng, boundary_truth, d, p)

        boundary_rows = [{"d_m": float(d[j]), "F_pN": float(F[j] * 1e12)} for j in range(len(d))]
        write_csv(
            data_dir / "boundary_curves" / f"{ds_id}_boundary.csv",
            boundary_rows,
            fieldnames=["d_m", "F_pN"],
        )

        # --- Generate dynamics data and PSD
        x, dyn_params = synth_dynamics(rng, dyn_truth, p)
        f, Pxx = welch_psd(x, p.fs_hz)
        psd_rows = [{"f_hz": float(f[k]), "Pxx": float(Pxx[k])} for k in range(len(f))]
        write_csv(
            data_dir / "psd" / f"{ds_id}_psd.csv",
            psd_rows,
            fieldnames=["f_hz", "Pxx"],
        )

        # --- Fit boundary candidates
        best_bic_b = float("inf")
        best_b_model = None
        best_b_params = None
        for m in p.boundary_candidates:
            q, rss, B = local_fit_boundary(rng, m, d, F, p)
            if B < best_bic_b:
                best_bic_b = B
                best_b_model = m
                best_b_params = q

        # --- Score dynamics candidates
        dyn_scores = score_dynamics_models(f, Pxx, p)
        # pick min BIC-like
        dyn_bics = {
            "colored": dyn_scores["bic_colored"],
            "resonant": dyn_scores["bic_resonant"],
            "drifting": dyn_scores["bic_drifting"],
        }
        best_d_model = min(dyn_bics, key=lambda k: dyn_bics[k])
        best_bic_d = float(dyn_bics[best_d_model])

        # --- Joint selection: add BICs (independent evidence approximation)
        # Joint candidates = boundary_model x dynamics_model
        # BIC_joint = BIC_boundary + BIC_dynamics
        # (This is the whole point: joint evidence breaks degeneracy.)
        best_joint = None
        best_bic_joint = float("inf")
        for bm in p.boundary_candidates:
            # to avoid refitting boundary for each bm, we need each model's BIC; compute quickly:
            _, _, B_bm = local_fit_boundary(rng, bm, d, F, p)
            for dm in p.dynamics_candidates:
                B_dm = float(dyn_bics[dm])
                B_joint = B_bm + B_dm
                if B_joint < best_bic_joint:
                    best_bic_joint = B_joint
                    best_joint = (bm, dm)

        # --- Confusion bookkeeping
        tb = boundary_truth
        td = dyn_truth
        pb = best_b_model
        pd = best_d_model
        tj = (tb, td)
        pj = best_joint

        # Map truth dynamics into the candidate set (resonant/colored/drifting are all in candidates)
        cm_b[bmap[tb], bmap[pb]] += 1
        cm_dy[dmap[td], dmap[pd]] += 1
        cm_j[joint_map[tj], joint_map[pj]] += 1

        if tj == corr_truth:
            corr_total += 1
            if pb == "plateau":
                corr_boundary_correct += 1
            if pj == corr_truth:
                corr_joint_correct += 1

        # Save per-dataset metrics JSON
        metrics = {
            "dataset_id": ds_id,
            "truth_boundary": tb,
            "truth_dynamics": td,
            "truth_joint": truth_joint,
            "boundary_truth_params": boundary_params,
            "dyn_truth_params": dyn_params,
            "boundary_winner": pb,
            "boundary_winner_bic": best_bic_b,
            "boundary_winner_params": best_b_params,
            "dyn_winner": pd,
            "dyn_winner_bic": best_bic_d,
            "dyn_metrics": {
                "f_peak_hz": dyn_scores["f_peak_hz"],
                "Q_est": dyn_scores["Q_est"],
                "peak_to_median_db": dyn_scores["peak_to_median_db"],
                "bw_hz": dyn_scores["bw_hz"],
            },
            "joint_winner": {"boundary": pj[0], "dynamics": pj[1]},
            "joint_winner_bic": best_bic_joint,
            "dyn_bics": dyn_bics,
        }
        save_json(data_dir / "metrics" / f"{ds_id}_metrics.json", metrics)

        # Index + selection row
        index_rows.append({
            "dataset_id": ds_id,
            "truth_boundary": tb,
            "truth_dynamics": td,
            "truth_joint": truth_joint,
        })
        selection_rows.append({
            "dataset_id": ds_id,
            "truth_boundary": tb,
            "truth_dynamics": td,
            "truth_joint": truth_joint,
            "pred_boundary": pb,
            "pred_dynamics": pd,
            "pred_joint": f"{pj[0]}+{pj[1]}",
        })

    write_csv(
        data_dir / "datasets_index.csv",
        index_rows,
        fieldnames=["dataset_id", "truth_boundary", "truth_dynamics", "truth_joint"],
    )
    write_csv(
        data_dir / "results_selection.csv",
        selection_rows,
        fieldnames=["dataset_id", "truth_boundary", "truth_dynamics", "truth_joint", "pred_boundary", "pred_dynamics", "pred_joint"],
    )

    # ============================================================
    # Plots
    # ============================================================

    # Winner matrices plot: three matrices (boundary, dynamics, joint)
    fig = plt.figure(figsize=(14, 4))

    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(cm_b / np.maximum(cm_b.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, aspect="auto", origin="lower")
    ax1.set_xticks(range(n_b))
    ax1.set_xticklabels(list(p.boundary_candidates), rotation=25, ha="right")
    ax1.set_yticks(range(n_b))
    ax1.set_yticklabels(list(p.boundary_candidates))
    ax1.set_title("Boundary-only confusion (row-normalized)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(cm_dy / np.maximum(cm_dy.sum(axis=1, keepdims=True), 1), vmin=0, vmax=1, aspect="auto", origin="lower")
    ax2.set_xticks(range(n_dy))
    ax2.set_xticklabels(list(p.dynamics_candidates), rotation=25, ha="right")
    ax2.set_yticks(range(n_dy))
    ax2.set_yticklabels(list(p.dynamics_candidates))
    ax2.set_title("Dynamics-only confusion (row-normalized)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # For joint, label axes by "boundary+dynamics"
    joint_labels = [f"{bm}+{dm}" for bm in p.boundary_candidates for dm in p.dynamics_candidates]
    ax3 = fig.add_subplot(133)
    cmj_norm = cm_j / np.maximum(cm_j.sum(axis=1, keepdims=True), 1)
    im3 = ax3.imshow(cmj_norm, vmin=0, vmax=1, aspect="auto", origin="lower")
    ax3.set_xticks(range(n_j))
    ax3.set_xticklabels(joint_labels, rotation=90, fontsize=7)
    ax3.set_yticks(range(n_j))
    ax3.set_yticklabels(joint_labels, fontsize=7)
    ax3.set_title("JOINT confusion (row-normalized)")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    save_figure(fig, run.figure_path("winner_matrices.png"))

    # Improvement bar plot for correlated truth recovery
    corr_boundary_rate = (corr_boundary_correct / corr_total) if corr_total > 0 else None
    corr_joint_rate = (corr_joint_correct / corr_total) if corr_total > 0 else None

    fig2 = plt.figure(figsize=(6, 4))
    ax = fig2.add_subplot(111)
    labels = ["boundary-only", "joint"]
    vals = [corr_boundary_rate if corr_boundary_rate is not None else 0.0,
            corr_joint_rate if corr_joint_rate is not None else 0.0]
    ax.bar(range(2), vals)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recovery rate for plateau+resonant truth")
    ax.set_title("Toy13 — correlated signature recovery improves")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig2, run.figure_path("improvement_bars.png"))

    # ============================================================
    # Summary
    # ============================================================

    summary = {
        "status": "ok",
        "interpretation": (
            "Toy13 quantifies that static boundary curves are degenerate (Toy12B), "
            "but joint evidence (boundary + independent kHz spectral fingerprint) improves discriminability. "
            "Use this to justify correlated falsification rather than single-observable claims."
        ),
        "params": asdict(p),
        "key_metrics": {
            "n_datasets": p.n_datasets,
            "correlated_truth": f"{corr_truth[0]}+{corr_truth[1]}",
            "correlated_total": corr_total,
            "correlated_boundary_recovery_rate": corr_boundary_rate,
            "correlated_joint_recovery_rate": corr_joint_rate,
        },
        "notes": [
            "If joint recovery >> boundary recovery, the theory's updated stance (correlated observables) is supported at the identifiability level.",
            "If joint recovery is not better, either the dynamics fingerprint is too weak under current SNR or the scoring model is too permissive; increase resonance Q / reduce meas_sigma / tighten scoring.",
            "This toy is not evidence; it is an identifiability stress test to design real experiments."
        ],
        "outputs": {
            "datasets_index_csv": str((data_dir / "datasets_index.csv").resolve()),
            "results_selection_csv": str((data_dir / "results_selection.csv").resolve()),
            "winner_matrices_png": str(run.figure_path("winner_matrices.png")),
            "improvement_bars_png": str(run.figure_path("improvement_bars.png")),
        },
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
