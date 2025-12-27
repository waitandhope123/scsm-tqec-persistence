"""
TOY 12 — BOUNDARY BATTLEFIELD: PLATEAU vs NULLS + SYSTEMATICS + MIXTURES
(UPDATED: fixed RunContext.read_csv bug + ticklabel warning)

Save as:
toy12_boundary_battlefield_plateau_vs_nulls.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


# ============================================================
# Parameters
# ============================================================

@dataclass
class Toy12Params:
    seed: int = 2040

    d_min_m: float = 0.2e-3
    d_max_m: float = 6.0e-3
    n_d: int = 60

    d_ref_m: float = 1.0e-3
    target_force_pN: float = 52.0

    noise_abs_pN: float = 0.8
    noise_rel_frac: float = 0.05
    ar1_rho: float = 0.6
    outlier_prob: float = 0.03
    outlier_scale: float = 6.0

    dist_drift_sigma: float = 0.01
    dist_drift_corr_len: int = 8

    force_mult_drift_sigma: float = 0.03
    force_mult_corr_len: int = 10
    force_add_drift_pN: float = 0.8
    force_add_corr_len: int = 12

    n_datasets_per_truth: int = 40

    plateau_d0_m: float = 3.0e-3
    plateau_n: float = 6.0

    yukawa_lambda_m_range: Tuple[float, float] = (0.6e-3, 3.0e-3)
    power_m_range: Tuple[float, float] = (2.0, 6.0)
    power_dc_m_range: Tuple[float, float] = (0.0, 0.5e-3)
    patch_m_range: Tuple[float, float] = (1.0, 3.5)
    patch_dc_m_range: Tuple[float, float] = (0.0, 0.8e-3)
    mix_patch_frac_range: Tuple[float, float] = (0.05, 0.6)

    k_folds: int = 5
    n_starts: int = 40
    n_local_steps: int = 120

    candidate_models = ("plateau", "yukawa", "power", "patch", "plateau_plus_patch")


# ============================================================
# Utilities
# ============================================================

def smooth_field(rng, n, L):
    x = rng.normal(size=n)
    k = np.ones(max(1, L)) / max(1, L)
    y = np.convolve(x, k, mode="same")
    return (y - y.mean()) / (y.std() + 1e-12)


def plateau_F(d, F0, d0, n):
    return F0 / (1.0 + (d / d0) ** n)


def calibrate_F0(d_ref, F_ref, d0, n):
    return F_ref * (1.0 + (d_ref / d0) ** n)


def yukawa_F(d, A, lam):
    return A * np.exp(-d / lam) / (d ** 2 + 1e-18)


def power_F(d, A, dc, m):
    return A / ((d + dc + 1e-12) ** m)


def patch_F(d, A, B, dc, m):
    return A / ((d + dc + 1e-12) ** m) + B / ((d + dc + 1e-12) ** 2)


def apply_systematics(rng, p, d, F):
    n = len(d)

    eta = smooth_field(rng, n, p.dist_drift_corr_len) * p.dist_drift_sigma
    d_eff = d * (1.0 + eta)

    delta = smooth_field(rng, n, p.force_mult_corr_len) * p.force_mult_drift_sigma
    b = smooth_field(rng, n, p.force_add_corr_len) * (p.force_add_drift_pN * 1e-12)

    F_sys = (1.0 + delta) * F + b

    sigma = p.noise_abs_pN * 1e-12 + p.noise_rel_frac * np.abs(F_sys)
    eps = rng.normal(size=n)
    ar = np.zeros(n)
    for i in range(n):
        ar[i] = eps[i] if i == 0 else p.ar1_rho * ar[i-1] + math.sqrt(1 - p.ar1_rho**2) * eps[i]
    noise = sigma * ar

    out = rng.random(size=n) < p.outlier_prob
    noise[out] += rng.normal(size=out.sum()) * p.outlier_scale * sigma.mean()

    return d_eff, F_sys + noise


# ============================================================
# Main
# ============================================================

def main():
    p = Toy12Params()
    run = create_run(
        toy_slug="toy12_boundary_battlefield",
        toy_name="Toy12 Boundary Battlefield",
        description="Plateau vs null force laws under realistic systematics",
        params=asdict(p),
    )

    rng = np.random.default_rng(p.seed)
    d = np.geomspace(p.d_min_m, p.d_max_m, p.n_d)

    truth_models = ["plateau", "yukawa", "power", "patch", "plateau_plus_patch"]

    datasets_by_id = {}
    dataset_index = []

    # -----------------------------
    # Generate datasets
    # -----------------------------

    for truth in truth_models:
        for k in range(p.n_datasets_per_truth):
            F_ref = p.target_force_pN * 1e-12

            if truth == "plateau":
                d0 = rng.uniform(1.5e-3, 5e-3)
                n = rng.uniform(3, 9)
                F0 = calibrate_F0(p.d_ref_m, F_ref, d0, n)
                F = plateau_F(d, F0, d0, n)

            elif truth == "yukawa":
                lam = rng.uniform(*p.yukawa_lambda_m_range)
                A = F_ref * d[0]**2 * np.exp(d[0]/lam)
                F = yukawa_F(d, A, lam)

            elif truth == "power":
                m = rng.uniform(*p.power_m_range)
                dc = rng.uniform(*p.power_dc_m_range)
                A = F_ref * (p.d_ref_m + dc)**m
                F = power_F(d, A, dc, m)

            elif truth == "patch":
                m = rng.uniform(*p.patch_m_range)
                dc = rng.uniform(*p.patch_dc_m_range)
                split = rng.uniform(0.2, 0.8)
                A = split * F_ref * (p.d_ref_m + dc)**m
                B = (1 - split) * F_ref * (p.d_ref_m + dc)**2
                F = patch_F(d, A, B, dc, m)

            else:
                d0 = rng.uniform(1.5e-3, 5e-3)
                n = rng.uniform(3, 9)
                frac = rng.uniform(*p.mix_patch_frac_range)
                F0 = calibrate_F0(p.d_ref_m, (1-frac)*F_ref, d0, n)

                m = rng.uniform(*p.patch_m_range)
                dc = rng.uniform(*p.patch_dc_m_range)
                split = rng.uniform(0.2, 0.8)
                A = split * frac * F_ref * (p.d_ref_m + dc)**m
                B = (1-split) * frac * F_ref * (p.d_ref_m + dc)**2

                F = plateau_F(d, F0, d0, n) + patch_F(d, A, B, dc, m)

            d_eff, F_obs = apply_systematics(rng, p, d, F)

            ds_id = f"{truth}_{k:03d}"
            datasets_by_id[ds_id] = (d_eff.copy(), F_obs.copy())
            dataset_index.append({"dataset_id": ds_id, "truth": truth})

            run.write_csv(
                f"datasets/{ds_id}.csv",
                [{"d_m": float(dd), "F_pN": float(ff * 1e12)} for dd, ff in zip(d_eff, F_obs)]
            )

    run.write_csv("datasets_index.csv", dataset_index)

    # -----------------------------
    # Simple visualization sanity check
    # -----------------------------

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    for i, (ds_id, (dd, ff)) in enumerate(datasets_by_id.items()):
        if i >= 6:
            break
        ax.plot(dd*1e3, ff*1e12, alpha=0.7)

    ax.set_xlabel("d (mm)")
    ax.set_ylabel("F (pN)")
    ax.set_title("Toy12 — Example synthetic datasets")
    ax.grid(True, alpha=0.3)

    save_figure(fig, run.figure_path("example_datasets.png"))

    run.save_summary({
        "status": "ok",
        "interpretation": (
            "Toy12 datasets successfully generated with realistic drift, noise, and mixtures. "
            "Ready for model-selection and null-discrimination analysis."
        ),
        "n_datasets": len(datasets_by_id),
        "truth_models": truth_models
    })


if __name__ == "__main__":
    main()
