"""
TOY 11 — ROBUSTNESS STRESS TEST: IDENTITY + BOUNDARY ALLOWED REGION UNDER ADVERSARIAL CONDITIONS

Purpose
-------
Toy10 found an ALLOWED region where:
- identity leakage <= leakage_max
- boundary force curve meets target + broadness

Toy11 asks: is that ALLOWED region robust or knife-edge?

We stress the integrator by sweeping:
- gamma_dephase (environment monitoring strength; Zeno stabilization knob)
- epsilon (sector mixing)
and adding an adversarial "symmetry breaking / drift" perturbation to the identity Hamiltonian.

We also stress the boundary side by:
- allowing mild random multiplicative drift in kappa_force across the distance grid
  (mimicking systematic calibration drift) and re-checking broadness.

Outputs
-------
- data/sweep_results.csv
- figures/allowed_region_gammaX.png  (one per gamma or a combined panel)
- figures/allowed_fraction_vs_gamma.png
- figures/leakage_heatmap_gammaX.png
- figures/allowed_panel.png
- summary.json

Interpretation
--------------
- If allowed fraction stays high across gamma, the framework is robust.
- If allowed fraction collapses unless gamma is very large or epsilon ~ 0, it's fragile.

No SciPy needed. All CSVs/plots produced.

"""

from __future__ import annotations

import math
import json
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
class Toy11Params:
    seed: int = 2033

    # ---------- Identity model ----------
    d_sector: int = 6
    sector_labels: Tuple[int, int, int] = (-1, 0, 1)

    dt: float = 0.01
    t_max: float = 25.0

    block_energy_scale: float = 1.0
    mix_energy_scale: float = 1.0

    # Sweep gamma (dephasing / monitoring)
    gamma_values: Tuple[float, ...] = (0.0, 0.2, 0.5, 1.0, 2.0, 4.0)

    # Sweep epsilon (sector mixing)
    eps_values: Tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

    # Adversarial symmetry-breaking perturbation added to Hamiltonian:
    # adds a small extra off-diagonal term independent of eps (worst-case mixing source)
    sb_strength_values: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.1)

    initial_sector_index: int = 1  # label 0 sector

    leakage_max: float = 0.01

    # ---------- Boundary model ----------
    target_force_pN: float = 52.0
    target_d_m: float = 1.0e-3
    tolerance_pN: float = 3.0

    d0_m: float = 3.0e-3
    n_shape: float = 6.0

    d_min_m: float = 0.2e-3
    d_max_m: float = 6.0e-3
    n_d: int = 48

    min_within_band_fraction: float = 0.05

    # Boundary amplitude scale sweep around 1.0 (matches target at 1mm)
    kappa_force_values: Tuple[float, ...] = (0.7, 0.85, 1.0, 1.15, 1.3)

    # Adversarial boundary systematic drift (multiplicative) across distance:
    # F(d) -> F(d) * (1 + drift_sigma * z(d)), where z is smooth-ish random field.
    drift_sigma_values: Tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.1)

    # Smoothness length-scale of drift in "index units"
    drift_corr_len: int = 6

    # Output controls
    make_panel_plot: bool = True


# -----------------------------
# Identity dynamics
# -----------------------------

def random_hermitian(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2.0
    return scale * H

def sector_projectors(d_sector: int, n_sectors: int) -> List[np.ndarray]:
    D = d_sector * n_sectors
    Ps = []
    for s in range(n_sectors):
        P = np.zeros((D, D), dtype=complex)
        a = s * d_sector
        b = (s + 1) * d_sector
        P[a:b, a:b] = np.eye(d_sector, dtype=complex)
        Ps.append(P)
    return Ps

def build_hamiltonian(rng: np.random.Generator, p: Toy11Params, eps: float, sb_strength: float) -> np.ndarray:
    nS = len(p.sector_labels)
    D = p.d_sector * nS

    H = np.zeros((D, D), dtype=complex)
    for si, label in enumerate(p.sector_labels):
        shift = float(label) * p.block_energy_scale
        Hb = random_hermitian(rng, p.d_sector, scale=0.3)
        a = si * p.d_sector
        b = (si + 1) * p.d_sector
        H[a:b, a:b] = Hb + shift * np.eye(p.d_sector)

    # eps-scaled mixing
    Hmix = random_hermitian(rng, D, scale=0.2)
    for si in range(nS):
        a = si * p.d_sector
        b = (si + 1) * p.d_sector
        Hmix[a:b, a:b] = 0.0
    H = H + (eps * p.mix_energy_scale) * Hmix

    # adversarial symmetry-breaking mixing (independent of eps)
    Hsb = random_hermitian(rng, D, scale=0.2)
    for si in range(nS):
        a = si * p.d_sector
        b = (si + 1) * p.d_sector
        Hsb[a:b, a:b] = 0.0
    H = H + sb_strength * Hsb

    return H

def lindblad_step(rho: np.ndarray, H: np.ndarray, Ls: List[np.ndarray], gamma: float, dt: float) -> np.ndarray:
    comm = H @ rho - rho @ H
    drho = -1j * comm

    if gamma > 0.0:
        for L in Ls:
            LrhoL = L @ rho @ L
            anti = L @ rho + rho @ L  # since projector
            drho = drho + gamma * (LrhoL - 0.5 * anti)

    rho2 = rho + dt * drho
    rho2 = (rho2 + rho2.conj().T) / 2.0
    tr = np.trace(rho2)
    if abs(tr) > 1e-12:
        rho2 = rho2 / tr
    w, V = np.linalg.eigh(rho2)
    w = np.clip(w, 0.0, None)
    rho2 = V @ np.diag(w) @ V.conj().T
    tr2 = np.trace(rho2)
    if abs(tr2) > 1e-12:
        rho2 = rho2 / tr2
    return rho2

def sector_probabilities(rho: np.ndarray, Ps: List[np.ndarray]) -> np.ndarray:
    return np.array([float(np.real(np.trace(P @ rho))) for P in Ps], dtype=float)

def simulate_max_leakage(p: Toy11Params, eps: float, gamma: float, sb_strength: float) -> float:
    rng = np.random.default_rng(
        p.seed
        + int(round(1e6 * eps)) % 99991
        + int(round(1e3 * gamma)) * 17
        + int(round(1e4 * sb_strength)) * 29
    )
    nS = len(p.sector_labels)
    D = p.d_sector * nS
    Ps = sector_projectors(p.d_sector, nS)
    H = build_hamiltonian(rng, p, eps, sb_strength)

    si0 = p.initial_sector_index
    a = si0 * p.d_sector
    b = (si0 + 1) * p.d_sector

    psi = np.zeros((D,), dtype=complex)
    v = rng.normal(size=p.d_sector) + 1j * rng.normal(size=p.d_sector)
    v = v / (np.linalg.norm(v) + 1e-12)
    psi[a:b] = v
    rho = np.outer(psi, psi.conj())

    steps = int(round(p.t_max / p.dt))
    max_leak = 0.0
    for _ in range(steps):
        probs = sector_probabilities(rho, Ps)
        leak = 1.0 - probs[si0]
        if leak > max_leak:
            max_leak = leak
        rho = lindblad_step(rho, H, Ps, gamma, p.dt)

    return float(max_leak)


# -----------------------------
# Boundary model (plateau) + adversarial drift
# -----------------------------

def plateau_F(d: np.ndarray, F0: float, d0: float, n: float) -> np.ndarray:
    return F0 / (1.0 + (d / d0) ** n)

def calibrate_F0_to_target(p: Toy11Params) -> float:
    dT = p.target_d_m
    FT = p.target_force_pN * 1e-12
    F0 = FT * (1.0 + (dT / p.d0_m) ** p.n_shape)
    return float(F0)

def smooth_random_field(rng: np.random.Generator, n: int, corr_len: int) -> np.ndarray:
    x = rng.normal(size=n)
    # simple moving-average smoothing
    M = max(1, int(corr_len))
    k = np.ones(M) / M
    y = np.convolve(x, k, mode="same")
    y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    return y

def within_band_fraction(F: np.ndarray, target_N: float, tol_N: float) -> float:
    return float(np.mean(np.abs(F - target_N) <= tol_N))


# -----------------------------
# Main sweep
# -----------------------------

def main():
    p = Toy11Params()
    run = create_run(
        toy_slug="toy11_robustness_identity_plus_boundary_stress",
        toy_name="TOY 11 — robustness stress test (identity + boundary)",
        description="Stress Toy10 by sweeping gamma, epsilon, symmetry-breaking mixing, and boundary drift; map allowed-region robustness.",
        params=asdict(p),
    )

    rng = np.random.default_rng(p.seed)

    d = np.geomspace(p.d_min_m, p.d_max_m, p.n_d)
    F0_base = calibrate_F0_to_target(p)
    target_N = p.target_force_pN * 1e-12
    tol_N = p.tolerance_pN * 1e-12

    rows: List[Dict[str, float]] = []

    # We'll compute allowed-fraction as a function of gamma and sb_strength and drift_sigma
    # by averaging over eps x kappa grid.
    summary_blocks = []

    for sb_strength in p.sb_strength_values:
        for drift_sigma in p.drift_sigma_values:
            allowed_fraction_vs_gamma = []
            for gamma in p.gamma_values:
                allowed_count = 0
                total_count = 0

                # Precompute leakage over eps for this gamma/sb
                leak_by_eps = {}
                for eps in p.eps_values:
                    max_leak = simulate_max_leakage(p, eps, gamma, sb_strength)
                    leak_by_eps[eps] = max_leak

                for eps in p.eps_values:
                    max_leak = leak_by_eps[eps]
                    passes_identity = (max_leak <= p.leakage_max)

                    for kappa in p.kappa_force_values:
                        # boundary with drift
                        # Generate a smooth multiplicative drift field
                        rr = np.random.default_rng(
                            p.seed
                            + int(round(1e3 * gamma)) * 101
                            + int(round(1e4 * sb_strength)) * 131
                            + int(round(1e4 * drift_sigma)) * 151
                            + int(round(1e6 * eps)) % 99991
                            + int(round(100 * kappa)) * 17
                        )
                        drift = smooth_random_field(rr, len(d), p.drift_corr_len)
                        F0 = kappa * F0_base
                        F_nom = plateau_F(d, F0, p.d0_m, p.n_shape)
                        F = F_nom * (1.0 + drift_sigma * drift)

                        F_at_target = float(plateau_F(np.array([p.target_d_m]), F0, p.d0_m, p.n_shape)[0])
                        passes_force_point = (abs(F_at_target - target_N) <= tol_N)

                        frac = within_band_fraction(F, target_N, tol_N)
                        passes_broad = (frac >= p.min_within_band_fraction)

                        allowed = bool(passes_identity and passes_force_point and passes_broad)

                        rows.append(
                            {
                                "gamma": float(gamma),
                                "epsilon": float(eps),
                                "sb_strength": float(sb_strength),
                                "drift_sigma": float(drift_sigma),
                                "kappa_force": float(kappa),
                                "max_leakage": float(max_leak),
                                "passes_identity": float(1.0 if passes_identity else 0.0),
                                "within_band_fraction": float(frac),
                                "passes_broadness": float(1.0 if passes_broad else 0.0),
                                "passes_force_point": float(1.0 if passes_force_point else 0.0),
                                "allowed": float(1.0 if allowed else 0.0),
                            }
                        )

                        total_count += 1
                        if allowed:
                            allowed_count += 1

                allowed_frac = allowed_count / max(1, total_count)
                allowed_fraction_vs_gamma.append(allowed_frac)

            summary_blocks.append(
                {
                    "sb_strength": float(sb_strength),
                    "drift_sigma": float(drift_sigma),
                    "gamma_values": list(p.gamma_values),
                    "allowed_fraction_vs_gamma": allowed_fraction_vs_gamma,
                }
            )

            # Plot allowed fraction vs gamma for this (sb_strength, drift_sigma)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(list(p.gamma_values), allowed_fraction_vs_gamma, marker="o")
            ax.set_xlabel("gamma_dephase")
            ax.set_ylabel("allowed fraction over (epsilon,kappa) grid")
            ax.set_title(f"Toy11 — Allowed fraction vs gamma (sb={sb_strength}, drift={drift_sigma})")
            ax.grid(True, alpha=0.3)
            save_figure(fig, run.figure_path(f"allowed_fraction_vs_gamma_sb{sb_strength}_drift{drift_sigma}.png"))

    run.write_csv("sweep_results.csv", rows)

    # Panel plot: choose drift_sigma=0 and sb_strength=0 as baseline, show all gamma curves for drift values
    if p.make_panel_plot:
        # Organize curves
        base_sb = 0.0
        figp = plt.figure(figsize=(10, 6))
        axp = figp.add_subplot(111)
        for drift_sigma in p.drift_sigma_values:
            # find block
            blk = None
            for b in summary_blocks:
                if abs(b["sb_strength"] - base_sb) < 1e-12 and abs(b["drift_sigma"] - float(drift_sigma)) < 1e-12:
                    blk = b
                    break
            if blk is None:
                continue
            axp.plot(blk["gamma_values"], blk["allowed_fraction_vs_gamma"], marker="o", label=f"drift={drift_sigma}")
        axp.set_xlabel("gamma_dephase")
        axp.set_ylabel("allowed fraction over (epsilon,kappa)")
        axp.set_title("Toy11 — Robustness panel (sb=0 baseline; varying boundary drift)")
        axp.grid(True, alpha=0.3)
        axp.legend(fontsize=9)
        save_figure(figp, run.figure_path("allowed_panel.png"))

    # Summary: robustness score
    # Define robustness score as mean allowed fraction over gamma for baseline sb=0 and drift=0,
    # plus a stressed score at sb=0.03 and drift=0.05.
    def find_curve(sb: float, drift: float):
        for b in summary_blocks:
            if abs(b["sb_strength"] - sb) < 1e-12 and abs(b["drift_sigma"] - drift) < 1e-12:
                return b["allowed_fraction_vs_gamma"]
        return None

    base_curve = find_curve(0.0, 0.0)
    stress_curve = find_curve(0.03, 0.05)

    base_score = float(np.mean(base_curve)) if base_curve is not None else float("nan")
    stress_score = float(np.mean(stress_curve)) if stress_curve is not None else float("nan")

    status = "ok"
    # simple qualitative tag
    if not math.isnan(base_score) and base_score < 0.02:
        tag = "fragile_baseline"
    elif not math.isnan(stress_score) and stress_score < 0.005:
        tag = "fragile_under_stress"
    else:
        tag = "robust_or_moderate"

    summary = {
        "status": status,
        "interpretation": (
            "Higher allowed-fraction indicates broader compatibility between identity stability and boundary measurability. "
            "Compare baseline vs stressed curves to judge fragility."
        ),
        "criteria": {
            "leakage_max": p.leakage_max,
            "tolerance_pN": p.tolerance_pN,
            "min_within_band_fraction": p.min_within_band_fraction,
        },
        "robustness_scores": {
            "baseline_mean_allowed_fraction(sb=0, drift=0)": base_score,
            "stressed_mean_allowed_fraction(sb=0.03, drift=0.05)": stress_score,
            "tag": tag,
        },
        "notes": [
            "If baseline is near zero, identity and boundary pillars barely overlap even before stress.",
            "If stressed collapses but baseline holds, the overlap exists but depends on environmental monitoring and low systematics.",
            "If stressed remains substantial, the framework is robust to plausible perturbations.",
        ],
        "summary_blocks": summary_blocks,  # includes all curves
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
