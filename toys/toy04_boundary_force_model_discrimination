"""
TOY 04 — BOUNDARY FORCE CLASS DISCRIMINATION (PLATEAU vs YUKAWA vs POWER vs PATCH)

Goal
----
Your strongest surviving empirical pillar is the "somatic/boundary force vs distance" claim.
Toy05b favored a plateau/saturation class over a simple Yukawa shape *by broadness*.
Toy04 makes this falsifiable in the way experiments actually work:

    Given noisy force-vs-distance data, can we *distinguish* a plateau model
    from plausible alternatives (Yukawa, power-law correction, patch potential surrogate)?

We:
1) Choose a "ground truth" generator (default: plateau)
2) Generate synthetic data F(d) on a realistic distance grid with noise (Gaussian + relative)
3) Fit competing models via grid-search least squares (no SciPy needed)
4) Compare via AIC/BIC + cross-validated holdout error
5) Output full CSVs and diagnostic plots

Models
------
Plateau:   F(d) = F0 / (1 + (d/d0)^n)
Yukawa:    F(d) = A * exp(-d/lambda) / d^2           (toy effective form)
PowerLaw:  F(d) = K / d^p
Patch:     F(d) = P0 / d^2 + P1 / d^4               (surrogate for electrostatic patches + residual Casimir)

Outputs
-------
- data/synthetic_data.csv              (d_m, F_N, sigma_N)
- data/fits_summary.csv                (model, best_params_json, sse, aic, bic, rmse, cv_rmse)
- data/fitted_curves.csv               (d_m, F_true, F_noisy, F_plateau, F_yukawa, F_power, F_patch)
- figures/fit_overlay.(png, svg)
- figures/residuals.(png, svg)
- summary.json

How this helps the theory
-------------------------
- If plateau is distinguishable under plausible noise floors and distance ranges,
  then the boundary-force claim has a real experimental handle.
- If not distinguishable, then "52 pN at 1mm" is not operationally specific enough,
  and you must refine the predicted functional form or add additional observables.

"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


# -----------------------------
# Parameters
# -----------------------------

@dataclass
class Toy04Params:
    seed: int = 2031

    # Distance grid (meters)
    d_min_m: float = 0.2e-3
    d_max_m: float = 6.0e-3
    n_points: int = 55

    # Ground truth selection: "plateau", "yukawa", "power", "patch"
    truth_model: str = "plateau"

    # Force scale around 52 pN at 1 mm
    target_d_m: float = 1.0e-3
    target_force_pN: float = 52.0

    # Noise model: absolute + relative (mimics AFM + systematics)
    noise_abs_pN: float = 0.8
    noise_rel_frac: float = 0.05

    # Cross-validation holdout fraction
    cv_holdout_frac: float = 0.25

    # Parameter search ranges (coarse but robust)
    # Plateau: d0 in [0.5mm, 6mm], n in [2, 10]
    plateau_d0_mm_grid: Tuple[float, ...] = (0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    plateau_n_grid: Tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)

    # Yukawa: lambda in [0.2mm, 5mm]
    yukawa_lambda_mm_grid: Tuple[float, ...] = (0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0)

    # Power: p in [1, 6]
    power_p_grid: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)

    # Patch: treat as linear in parameters (P0, P1), no grid needed

    # Plot styling
    show_loglog: bool = True


# -----------------------------
# Model functions
# -----------------------------

def F_plateau(d: np.ndarray, F0: float, d0: float, n: float) -> np.ndarray:
    return F0 / (1.0 + (d / d0) ** n)

def F_yukawa(d: np.ndarray, A: float, lam: float) -> np.ndarray:
    # Effective "Yukawa-like" decay with geometric factor, purely toy.
    return A * np.exp(-d / lam) / (d ** 2 + 1e-30)

def F_power(d: np.ndarray, K: float, p: float) -> np.ndarray:
    return K / (d ** p + 1e-30)

def F_patch(d: np.ndarray, P0: float, P1: float) -> np.ndarray:
    return P0 / (d ** 2 + 1e-30) + P1 / (d ** 4 + 1e-30)


# -----------------------------
# Calibration to match target at d_target
# -----------------------------

def calibrate_truth(p: Toy04Params) -> Dict[str, float]:
    """
    Choose "truth" parameters such that F(target_d)=target_force.
    We'll use reasonable default shape parameters and solve for the amplitude.
    """
    dT = p.target_d_m
    FT = p.target_force_pN * 1e-12

    if p.truth_model == "plateau":
        d0 = 3.0e-3
        n = 6.0
        # Solve F0
        F0 = FT * (1.0 + (dT / d0) ** n)
        return {"F0": F0, "d0": d0, "n": n}

    if p.truth_model == "yukawa":
        lam = 1.0e-3
        A = FT * (dT ** 2) * math.exp(dT / lam)
        return {"A": A, "lam": lam}

    if p.truth_model == "power":
        pp = 3.0
        K = FT * (dT ** pp)
        return {"K": K, "p": pp}

    if p.truth_model == "patch":
        # Set P1 small; solve P0 for target
        P1 = 0.05 * FT * (dT ** 4)
        P0 = (FT - P1 / (dT ** 4 + 1e-30)) * (dT ** 2)
        return {"P0": P0, "P1": P1}

    raise ValueError(f"Unknown truth_model: {p.truth_model}")


def generate_truth_curve(p: Toy04Params, d: np.ndarray, truth_params: Dict[str, float]) -> np.ndarray:
    if p.truth_model == "plateau":
        return F_plateau(d, truth_params["F0"], truth_params["d0"], truth_params["n"])
    if p.truth_model == "yukawa":
        return F_yukawa(d, truth_params["A"], truth_params["lam"])
    if p.truth_model == "power":
        return F_power(d, truth_params["K"], truth_params["p"])
    if p.truth_model == "patch":
        return F_patch(d, truth_params["P0"], truth_params["P1"])
    raise ValueError(p.truth_model)


# -----------------------------
# Fitting utilities
# -----------------------------

def sse(y: np.ndarray, yhat: np.ndarray, sigma: np.ndarray) -> float:
    r = (y - yhat) / (sigma + 1e-30)
    return float(np.sum(r**2))

def aic_bic(sse_val: float, n: int, k: int) -> Tuple[float, float]:
    # Gaussian log-likelihood up to constant: -0.5 * sse
    # AIC = 2k + sse
    # BIC = k ln n + sse
    aic = 2.0 * k + sse_val
    bic = float(k * math.log(max(n, 2)) + sse_val)
    return float(aic), float(bic)

def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat)**2)))

def split_train_test(n: int, frac: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    rng.shuffle(idx)
    m = int(round(frac * n))
    test = idx[:m]
    train = idx[m:]
    return train, test


def fit_plateau_grid(d: np.ndarray, y: np.ndarray, sigma: np.ndarray, p: Toy04Params) -> Dict:
    best = {"sse": float("inf")}
    for d0_mm in p.plateau_d0_mm_grid:
        d0 = d0_mm * 1e-3
        for nn in p.plateau_n_grid:
            # linear in F0
            X = 1.0 / (1.0 + (d / d0) ** nn)
            # weighted least squares for F0
            w = 1.0 / (sigma**2 + 1e-30)
            F0 = float(np.sum(w * X * y) / np.sum(w * X * X))
            yhat = F_plateau(d, F0, d0, nn)
            ss = sse(y, yhat, sigma)
            if ss < best["sse"]:
                best = {"model": "plateau", "params": {"F0": F0, "d0": d0, "n": float(nn)}, "sse": ss, "yhat": yhat}
    return best

def fit_yukawa_grid(d: np.ndarray, y: np.ndarray, sigma: np.ndarray, p: Toy04Params) -> Dict:
    best = {"sse": float("inf")}
    for lam_mm in p.yukawa_lambda_mm_grid:
        lam = lam_mm * 1e-3
        X = np.exp(-d / lam) / (d**2 + 1e-30)
        w = 1.0 / (sigma**2 + 1e-30)
        A = float(np.sum(w * X * y) / np.sum(w * X * X))
        yhat = F_yukawa(d, A, lam)
        ss = sse(y, yhat, sigma)
        if ss < best["sse"]:
            best = {"model": "yukawa", "params": {"A": A, "lam": lam}, "sse": ss, "yhat": yhat}
    return best

def fit_power_grid(d: np.ndarray, y: np.ndarray, sigma: np.ndarray, p: Toy04Params) -> Dict:
    best = {"sse": float("inf")}
    for pp in p.power_p_grid:
        X = 1.0 / (d**pp + 1e-30)
        w = 1.0 / (sigma**2 + 1e-30)
        K = float(np.sum(w * X * y) / np.sum(w * X * X))
        yhat = F_power(d, K, pp)
        ss = sse(y, yhat, sigma)
        if ss < best["sse"]:
            best = {"model": "power", "params": {"K": K, "p": float(pp)}, "sse": ss, "yhat": yhat}
    return best

def fit_patch_linear(d: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> Dict:
    # y = P0/d^2 + P1/d^4
    X0 = 1.0 / (d**2 + 1e-30)
    X1 = 1.0 / (d**4 + 1e-30)
    W = np.diag(1.0 / (sigma**2 + 1e-30))
    X = np.vstack([X0, X1]).T  # n x 2
    # weighted least squares: beta = (X^T W X)^-1 X^T W y
    XtW = X.T @ W
    beta = np.linalg.pinv(XtW @ X) @ (XtW @ y)
    P0, P1 = float(beta[0]), float(beta[1])
    yhat = F_patch(d, P0, P1)
    ss = sse(y, yhat, sigma)
    return {"model": "patch", "params": {"P0": P0, "P1": P1}, "sse": ss, "yhat": yhat}


def fit_all(d: np.ndarray, y: np.ndarray, sigma: np.ndarray, p: Toy04Params) -> List[Dict]:
    fits = [
        fit_plateau_grid(d, y, sigma, p),
        fit_yukawa_grid(d, y, sigma, p),
        fit_power_grid(d, y, sigma, p),
        fit_patch_linear(d, y, sigma),
    ]
    # add info criteria
    n = len(d)
    k_map = {"plateau": 3, "yukawa": 2, "power": 2, "patch": 2}
    for f in fits:
        k = k_map[f["model"]]
        aic, bic = aic_bic(f["sse"], n, k)
        f["aic"] = aic
        f["bic"] = bic
        f["rmse"] = rmse(y, f["yhat"])
    return fits


def cv_score(d: np.ndarray, y: np.ndarray, sigma: np.ndarray, p: Toy04Params, rng: np.random.Generator) -> Dict[str, float]:
    train_idx, test_idx = split_train_test(len(d), p.cv_holdout_frac, rng)
    dtr, ytr, str_ = d[train_idx], y[train_idx], sigma[train_idx]
    dte, yte, ste = d[test_idx], y[test_idx], sigma[test_idx]

    fits_tr = fit_all(dtr, ytr, str_, p)
    scores = {}
    for f in fits_tr:
        model = f["model"]
        params = f["params"]
        if model == "plateau":
            yhat = F_plateau(dte, params["F0"], params["d0"], params["n"])
        elif model == "yukawa":
            yhat = F_yukawa(dte, params["A"], params["lam"])
        elif model == "power":
            yhat = F_power(dte, params["K"], params["p"])
        elif model == "patch":
            yhat = F_patch(dte, params["P0"], params["P1"])
        else:
            continue
        scores[model] = rmse(yte, yhat)
    return scores


# -----------------------------
# Main
# -----------------------------

def main():
    p = Toy04Params()
    run = create_run(
        toy_slug="toy04_boundary_force_model_discrimination",
        toy_name="TOY 04 — boundary force class discrimination",
        description="Generate synthetic force-vs-distance data and compare plateau vs Yukawa vs power-law vs patch via AIC/BIC + CV.",
        params=asdict(p),
    )

    rng = np.random.default_rng(p.seed)

    # Distance grid (log-spaced is typical for force curves)
    d = np.geomspace(p.d_min_m, p.d_max_m, p.n_points)

    truth_params = calibrate_truth(p)
    F_true = generate_truth_curve(p, d, truth_params)

    # Noise model
    sigma = (p.noise_abs_pN * 1e-12) + (p.noise_rel_frac * np.abs(F_true))
    F_noisy = F_true + rng.normal(size=len(d)) * sigma

    # Save synthetic data
    run.write_csv(
        "synthetic_data.csv",
        [{"d_m": float(dd), "F_N": float(ff), "sigma_N": float(ss)} for dd, ff, ss in zip(d, F_noisy, sigma)],
    )

    # Fit all models on full data
    fits = fit_all(d, F_noisy, sigma, p)

    # Cross-validated score (single split)
    cv = cv_score(d, F_noisy, sigma, p, rng)

    # Assemble summary table rows
    summary_rows = []
    for f in fits:
        model = f["model"]
        row = {
            "model": model,
            "sse": float(f["sse"]),
            "aic": float(f["aic"]),
            "bic": float(f["bic"]),
            "rmse": float(f["rmse"]),
            "cv_rmse": float(cv.get(model, float("nan"))),
            "best_params_json": json.dumps(f["params"]),
        }
        summary_rows.append(row)

    # sort by BIC (harsher)
    summary_rows_sorted = sorted(summary_rows, key=lambda r: r["bic"])
    run.write_csv("fits_summary.csv", summary_rows_sorted)

    # Fitted curves on full grid
    curve = {"d_m": d, "F_true": F_true, "F_noisy": F_noisy}
    # get best yhat per model
    yhat_map = {f["model"]: f["yhat"] for f in fits}
    curve["F_plateau"] = yhat_map["plateau"]
    curve["F_yukawa"] = yhat_map["yukawa"]
    curve["F_power"] = yhat_map["power"]
    curve["F_patch"] = yhat_map["patch"]

    run.write_csv(
        "fitted_curves.csv",
        [
            {
                "d_m": float(dd),
                "F_true_N": float(ft),
                "F_noisy_N": float(fn),
                "F_plateau_N": float(fp),
                "F_yukawa_N": float(fy),
                "F_power_N": float(fpow),
                "F_patch_N": float(fpa),
            }
            for dd, ft, fn, fp, fy, fpow, fpa in zip(
                d,
                curve["F_true"],
                curve["F_noisy"],
                curve["F_plateau"],
                curve["F_yukawa"],
                curve["F_power"],
                curve["F_patch"],
            )
        ],
    )

    # Plots
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.errorbar(d, F_noisy * 1e12, yerr=sigma * 1e12, fmt="o", markersize=3, label="synthetic data")
    ax.plot(d, F_true * 1e12, linewidth=2, label=f"truth: {p.truth_model}")
    ax.plot(d, curve["F_plateau"] * 1e12, label="fit plateau")
    ax.plot(d, curve["F_yukawa"] * 1e12, label="fit yukawa")
    ax.plot(d, curve["F_power"] * 1e12, label="fit power")
    ax.plot(d, curve["F_patch"] * 1e12, label="fit patch")

    ax.set_xlabel("distance d (m)")
    ax.set_ylabel("Force (pN)")
    ax.set_title("Toy 04 — Force-vs-distance model fits")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    if p.show_loglog:
        ax.set_xscale("log")
        # forces can be negative in real Casimir; here positive toy, so safe
        ax.set_yscale("log")
    save_figure(fig1, run.figure_path("fit_overlay.png"))

    # Residuals
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(d, (F_noisy - curve["F_plateau"]) / (sigma + 1e-30), label="plateau")
    ax2.plot(d, (F_noisy - curve["F_yukawa"]) / (sigma + 1e-30), label="yukawa")
    ax2.plot(d, (F_noisy - curve["F_power"]) / (sigma + 1e-30), label="power")
    ax2.plot(d, (F_noisy - curve["F_patch"]) / (sigma + 1e-30), label="patch")
    ax2.axhline(0.0, linestyle="--", alpha=0.6)
    ax2.set_xlabel("distance d (m)")
    ax2.set_ylabel("normalized residual (σ units)")
    ax2.set_title("Toy 04 — Residuals by model")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_xscale("log")
    save_figure(fig2, run.figure_path("residuals.png"))

    # Decide "winner" by BIC and by CV
    winner_bic = summary_rows_sorted[0]["model"]
    winner_cv = min(summary_rows, key=lambda r: r["cv_rmse"])["model"]

    # How decisive? delta_BIC between best and second-best
    if len(summary_rows_sorted) >= 2:
        delta_bic = float(summary_rows_sorted[1]["bic"] - summary_rows_sorted[0]["bic"])
    else:
        delta_bic = float("nan")

    summary = {
        "status": "ok",
        "truth_model": p.truth_model,
        "truth_params": truth_params,
        "noise": {"noise_abs_pN": p.noise_abs_pN, "noise_rel_frac": p.noise_rel_frac},
        "winners": {"bic": winner_bic, "cv": winner_cv},
        "delta_bic_best_vs_second": delta_bic,
        "interpretation": (
            "Large delta_BIC (>10) usually means strong evidence for best model. "
            "If plateau wins decisively under plausible noise, plateau is operationally testable; "
            "if multiple models tie, boundary predictions need more structure/observables."
        ),
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
