"""
TOY 05b — CASIMIR ANOMALY SIGNATURE DISCRIMINATOR (YUKAWA vs PLATEAU)

Purpose
-------
Your Toy 05(v2) showed that a simple Yukawa-like χ correction hits 52 pN at 1 mm
only as a tuned point (narrow in distance). This toy compares two *classes* of
distance-scaling fingerprints:

Model A (Yukawa-like):
    F_A(d) = A * exp(-d/λ) / d^2

Model B (Plateau / saturating boundary-like):
    F_B(d) = F0 / (1 + (d/d0)^n)

Both are auto-calibrated to match the target at exactly d = 1 mm, and then we
evaluate "broadness": fraction of distances where F(d) stays within ±tol.

This toy is designed to help define a *defining characteristic curve*, not just
a single-point match.

Outputs
-------
- data/models_force_vs_distance.csv
- data/plateau_scan.csv
- figures/force_fingerprints.(png, svg)
- figures/within_band_masks.(png, svg)
- figures/plateau_scan_within_fraction.(png, svg)
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

HBAR_C_eVm = 1.973269804e-7  # ħc in eV·m


@dataclass
class Toy05bParams:
    # Distance grid for fingerprint evaluation
    d_min_m: float = 2e-4
    d_max_m: float = 5e-3
    n_points: int = 240

    # Target "somatic field" force (we treat this toy as testing the new-physics component)
    # You can later swap in "total = Casimir + correction"; here Casimir is negligible at 1mm for typical areas.
    target_force_pN: float = 52.0
    tolerance_pN: float = 3.0

    # Model A (Yukawa-like) range
    m_chi_eV: float = 2e-4  # ~1mm Compton wavelength

    # Plateau scan ranges (Model B)
    d0_min_m: float = 2e-4
    d0_max_m: float = 3e-3
    d0_points: int = 40

    n_min: float = 0.5
    n_max: float = 6.0
    n_points_scan: int = 40

    # Broadness criterion (same spirit as earlier)
    min_within_band_fraction: float = 0.05

    # Reference point for calibration
    d_cal_m: float = 1e-3


def compton_lambda_m(m_eV: float) -> float:
    return HBAR_C_eVm / m_eV


# -------- Model A: Yukawa-like --------
def model_yukawa(d: np.ndarray, lam: float, amp_Nm2: float) -> np.ndarray:
    # F = amp * exp(-d/lam) / d^2
    return amp_Nm2 * np.exp(-d / lam) / (d**2)


def solve_amp_yukawa(d0: float, lam: float, F0_N: float) -> float:
    # amp = F0 * d^2 * exp(d/lam)
    return float(F0_N * (d0**2) * math.exp(d0 / lam))


# -------- Model B: Plateau --------
def model_plateau(d: np.ndarray, F0_N: float, d0: float, n: float) -> np.ndarray:
    return F0_N / (1.0 + (d / d0) ** n)


def solve_F0_plateau(d_cal: float, F_cal_N: float, d0: float, n: float) -> float:
    # F_cal = F0 / (1 + (d_cal/d0)^n)  =>  F0 = F_cal * (1 + (d_cal/d0)^n)
    return float(F_cal_N * (1.0 + (d_cal / d0) ** n))


def within_band_fraction(F_pN: np.ndarray, target_pN: float, tol_pN: float) -> float:
    return float(np.mean(np.abs(F_pN - target_pN) <= tol_pN))


def local_log_slope(d: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Estimate local slope s(d) = d(log|F|)/d(log d) via finite differences.
    Useful to visualize effective power-law behavior.
    """
    eps = 1e-300
    x = np.log(d)
    y = np.log(np.abs(F) + eps)
    dy = np.gradient(y, x)
    return dy


def main():
    p = Toy05bParams()
    run = create_run(
        toy_slug="toy05b_casimir_signature_models",
        toy_name="TOY 05b — Casimir anomaly fingerprints (Yukawa vs Plateau)",
        description="Compare distance-scaling fingerprints; auto-calibrate at 1mm; evaluate broadness across d.",
        params=asdict(p),
    )

    d = np.linspace(p.d_min_m, p.d_max_m, p.n_points)
    d_cal = p.d_cal_m

    target_N = p.target_force_pN * 1e-12
    tol_pN = p.tolerance_pN

    # ---- Model A (Yukawa-like), auto-calibrated ----
    lam = compton_lambda_m(p.m_chi_eV)
    amp = solve_amp_yukawa(d_cal, lam, target_N)
    F_A = model_yukawa(d, lam, amp)
    F_A_pN = F_A * 1e12
    frac_A = within_band_fraction(F_A_pN, p.target_force_pN, tol_pN)

    # ---- Model B (Plateau), scan for best broadness ----
    d0_grid = np.linspace(p.d0_min_m, p.d0_max_m, p.d0_points)
    n_grid = np.linspace(p.n_min, p.n_max, p.n_points_scan)

    best: Dict[str, float] = {"within_frac": -1.0, "d0": float("nan"), "n": float("nan"), "F0_N": float("nan")}
    scan_rows = []

    for d0 in d0_grid:
        for n in n_grid:
            F0 = solve_F0_plateau(d_cal, target_N, d0, n)
            F_B = model_plateau(d, F0, d0, n)
            frac = within_band_fraction(F_B * 1e12, p.target_force_pN, tol_pN)

            scan_rows.append(
                {
                    "d0_m": float(d0),
                    "n": float(n),
                    "F0_N": float(F0),
                    "within_band_fraction": float(frac),
                }
            )
            if frac > best["within_frac"]:
                best = {"within_frac": frac, "d0": float(d0), "n": float(n), "F0_N": float(F0)}

    run.write_csv("plateau_scan.csv", scan_rows)

    # Recompute best plateau model curve
    F0_best = best["F0_N"]
    d0_best = best["d0"]
    n_best = best["n"]
    F_B = model_plateau(d, F0_best, d0_best, n_best)
    F_B_pN = F_B * 1e12
    frac_B = within_band_fraction(F_B_pN, p.target_force_pN, tol_pN)

    # ---- Save combined fingerprint data ----
    rows = []
    for di, fa, fb in zip(d, F_A_pN, F_B_pN):
        rows.append(
            {
                "distance_m": float(di),
                "distance_mm": float(di * 1e3),
                "F_yukawa_pN": float(fa),
                "F_plateau_pN": float(fb),
                "within_yukawa": int(abs(fa - p.target_force_pN) <= tol_pN),
                "within_plateau": int(abs(fb - p.target_force_pN) <= tol_pN),
            }
        )
    run.write_csv("models_force_vs_distance.csv", rows)

    # ---- Plots: fingerprints ----
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(d * 1e3, F_A_pN, label="Model A: Yukawa-like")
    ax1.plot(d * 1e3, F_B_pN, label=f"Model B: Plateau (best d0={d0_best*1e3:.3f}mm, n={n_best:.2f})")
    ax1.axhline(p.target_force_pN, linestyle="--", alpha=0.6, label="target")
    ax1.axhspan(p.target_force_pN - tol_pN, p.target_force_pN + tol_pN, alpha=0.15, label="±tol band")
    ax1.axvline(d_cal * 1e3, linestyle="--", alpha=0.4)
    ax1.set_xlabel("distance (mm)")
    ax1.set_ylabel("Force (pN)")
    ax1.set_title("Toy 05b — Force fingerprints (calibrated at 1mm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    save_figure(fig1, run.figure_path("force_fingerprints.png"))

    # ---- Plots: within-band masks ----
    within_A = (np.abs(F_A_pN - p.target_force_pN) <= tol_pN).astype(float)
    within_B = (np.abs(F_B_pN - p.target_force_pN) <= tol_pN).astype(float)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(d * 1e3, within_A, label=f"Yukawa within band (frac={frac_A:.3f})")
    ax2.plot(d * 1e3, within_B, label=f"Plateau within band (frac={frac_B:.3f})")
    ax2.set_xlabel("distance (mm)")
    ax2.set_ylabel("within band (0/1)")
    ax2.set_title(f"Toy 05b — Within ±{tol_pN} pN band")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    save_figure(fig2, run.figure_path("within_band_masks.png"))

    # ---- Plot: plateau scan summary (as scatter heat proxy) ----
    # For clarity without extra deps, make a scatter where color is within-band fraction.
    d0_arr = np.array([r["d0_m"] for r in scan_rows])
    n_arr = np.array([r["n"] for r in scan_rows])
    frac_arr = np.array([r["within_band_fraction"] for r in scan_rows])

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    sc = ax3.scatter(d0_arr * 1e3, n_arr, c=frac_arr)
    ax3.set_xlabel("d0 (mm)")
    ax3.set_ylabel("n")
    ax3.set_title("Toy 05b — Plateau scan (color = within-band fraction)")
    ax3.grid(True, alpha=0.3)
    fig3.colorbar(sc, ax=ax3, label="within-band fraction")
    save_figure(fig3, run.figure_path("plateau_scan_within_fraction.png"))

    # ---- Optional diagnostic: local slopes (not required, but useful) ----
    slope_A = local_log_slope(d, F_A)
    slope_B = local_log_slope(d, F_B)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(d * 1e3, slope_A, label="Yukawa local log-slope")
    ax4.plot(d * 1e3, slope_B, label="Plateau local log-slope")
    ax4.axvline(d_cal * 1e3, linestyle="--", alpha=0.4)
    ax4.set_xlabel("distance (mm)")
    ax4.set_ylabel("d log|F| / d log d")
    ax4.set_title("Toy 05b — Effective scaling exponent vs distance")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    save_figure(fig4, run.figure_path("local_scaling_exponent.png"))

    # ---- Pass/fail logic ----
    # We "pass" if at least one model class yields broadness >= threshold,
    # and we report which model wins.
    passes_A = frac_A >= p.min_within_band_fraction
    passes_B = frac_B >= p.min_within_band_fraction

    winner = "plateau" if frac_B > frac_A else "yukawa"
    summary = {
        "status": "pass" if (passes_A or passes_B) else "fail",
        "target_force_pN": p.target_force_pN,
        "tolerance_pN": tol_pN,
        "min_within_band_fraction": p.min_within_band_fraction,
        "model_A_yukawa": {
            "m_chi_eV": p.m_chi_eV,
            "lambda_compton_m": float(lam),
            "amp_Nm2": float(amp),
            "within_band_fraction": float(frac_A),
            "passes_broadness": bool(passes_A),
        },
        "model_B_plateau_best": {
            "d0_m": float(d0_best),
            "d0_mm": float(d0_best * 1e3),
            "n": float(n_best),
            "F0_N": float(F0_best),
            "within_band_fraction": float(frac_B),
            "passes_broadness": bool(passes_B),
        },
        "winner_by_broadness": winner,
        "note": "Both models are calibrated to match target exactly at d=1mm analytically; broadness is evaluated across the distance grid.",
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
