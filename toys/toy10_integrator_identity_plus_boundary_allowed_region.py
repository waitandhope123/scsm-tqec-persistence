"""
TOY 10 — INTEGRATOR: IDENTITY SUPERSELECTION + BOUNDARY-COUPLING (ALLOWED REGION MAP)

Purpose
-------
Toy04 strengthened the boundary-force pillar (plateau form is operationally distinguishable).
Toy09 strengthened the identity pillar (superselection/leakage can be small below a mixing bound).

Toy10 integrates them:
- We model a 3-sector "identity" Hilbert space with weak sector mixing (epsilon),
  plus environment-induced dephasing (gamma) that can Zeno-stabilize sector labels.
- We also model a boundary-coupling channel that produces a measurable force-vs-distance
  signature with plateau form.
- We sweep parameters and find the region where:
    (A) Identity is stable: leakage <= leakage_max
    (B) Boundary effect is measurable: force at 1mm within tolerance, and curve broadness >= min fraction

This produces an "allowed region" map: a concrete, toy-level consistency check for
"identity persistence" AND "measurable boundary signature" simultaneously.

What this does NOT do
---------------------
- It does not validate SCSM physically.
- It does not simulate AQG or real Casimir.
It tests internal *compatibility* of two pillars under reasonable abstract dynamics.

Outputs
-------
- data/sweep_results.csv
- figures/allowed_region.png
- figures/leakage_heatmap.png
- figures/force_broadness_heatmap.png
- summary.json

Runtime
-------
Designed to be manageable (minutes) with a modest grid.

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
class Toy10Params:
    seed: int = 2032

    # ---------- Identity (superselection) model ----------
    # Three sectors: labels [-1, 0, +1], each a block of dimension d_sector
    d_sector: int = 6
    sector_labels: Tuple[int, int, int] = (-1, 0, 1)

    # Time evolution
    dt: float = 0.01
    t_max: float = 25.0

    # Block energy scale and mixing scale (toy)
    block_energy_scale: float = 1.0
    mix_energy_scale: float = 1.0

    # Dephasing (environment monitoring sector label)
    gamma_dephase: float = 1.0

    # Initial pure state in sector 0
    initial_sector_index: int = 1  # corresponds to label 0

    # Leakage criterion
    leakage_max: float = 0.01

    # ---------- Boundary force toy (plateau) ----------
    # Force curve: F(d) = F0 / (1 + (d/d0)^n)
    target_force_pN: float = 52.0
    target_d_m: float = 1.0e-3
    tolerance_pN: float = 3.0

    # Default plateau shape parameters
    d0_m: float = 3.0e-3
    n_shape: float = 6.0

    # Distance grid for broadness test
    d_min_m: float = 0.2e-3
    d_max_m: float = 6.0e-3
    n_d: int = 48
    min_within_band_fraction: float = 0.05  # fraction of distances within tolerance band

    # ---------- Coupling linkage between pillars ----------
    # Here we link boundary coupling amplitude to mixing epsilon:
    # - epsilon governs sector mixing strength (identity risk)
    # - boundary amplitude kappa_force governs plateau force scale (measurability)
    # We sweep both and look for compatibility.
    eps_values: Tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    kappa_force_values: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 2.0)

    # Interpretation:
    # kappa_force rescales the plateau force curve around target:
    # F0_actual = kappa_force * F0_calibrated_to_target
    # So kappa_force=1 corresponds to matching target exactly at 1mm.
    #
    # You can choose to impose a hypothesis that epsilon scales with kappa_force
    # (e.g., epsilon = eps0 * kappa_force). Here we keep them independent, then
    # compute an "allowed region" with both dimensions.

    # ---------- Output options ----------
    save_example_curves: bool = True
    example_curve_points: Tuple[Tuple[float, float], ...] = (
        (1e-3, 1.0),   # (epsilon, kappa_force)
        (1e-2, 1.0),
        (1e-3, 1.6),
        (1e-2, 1.6),
    )


# -----------------------------
# Identity dynamics (density matrix) — toy Lindblad dephasing + mixing
# -----------------------------

def random_hermitian(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2.0
    return scale * H

def sector_projectors(d_sector: int, n_sectors: int) -> List[np.ndarray]:
    """
    Projector onto each sector block in the full Hilbert space.
    """
    D = d_sector * n_sectors
    Ps = []
    for s in range(n_sectors):
        P = np.zeros((D, D), dtype=complex)
        a = s * d_sector
        b = (s + 1) * d_sector
        P[a:b, a:b] = np.eye(d_sector, dtype=complex)
        Ps.append(P)
    return Ps

def build_hamiltonian(rng: np.random.Generator, p: Toy10Params, eps: float) -> np.ndarray:
    """
    Block-diagonal energies + weak random off-diagonal mixing scaled by eps.
    """
    nS = len(p.sector_labels)
    D = p.d_sector * nS

    # Block-diagonal H0
    H = np.zeros((D, D), dtype=complex)
    for si, label in enumerate(p.sector_labels):
        # energy shift per sector label (toy)
        shift = float(label) * p.block_energy_scale
        Hb = random_hermitian(rng, p.d_sector, scale=0.3)
        a = si * p.d_sector
        b = (si + 1) * p.d_sector
        H[a:b, a:b] = Hb + shift * np.eye(p.d_sector)

    # Off-diagonal mixing blocks
    Hmix = random_hermitian(rng, D, scale=0.2)
    # remove diagonal blocks to isolate mixing
    for si in range(nS):
        a = si * p.d_sector
        b = (si + 1) * p.d_sector
        Hmix[a:b, a:b] = 0.0
    H = H + (eps * p.mix_energy_scale) * Hmix
    return H

def lindblad_step(rho: np.ndarray, H: np.ndarray, Ls: List[np.ndarray], gamma: float, dt: float) -> np.ndarray:
    """
    Euler step for:
      dρ = -i[H,ρ] + γ Σ ( L ρ L - 1/2 {L^2, ρ} )
    Here L are projectors (monitoring sectors).
    """
    comm = H @ rho - rho @ H
    drho = -1j * comm

    if gamma > 0.0:
        for L in Ls:
            LrhoL = L @ rho @ L
            L2 = L @ L  # projector => itself
            anti = L2 @ rho + rho @ L2
            drho = drho + gamma * (LrhoL - 0.5 * anti)

    rho2 = rho + dt * drho
    # enforce Hermiticity + trace=1 + positivity (soft)
    rho2 = (rho2 + rho2.conj().T) / 2.0
    tr = np.trace(rho2)
    if abs(tr) > 1e-12:
        rho2 = rho2 / tr
    # clip tiny negative eigenvalues
    w, V = np.linalg.eigh(rho2)
    w = np.clip(w, 0.0, None)
    rho2 = V @ np.diag(w) @ V.conj().T
    tr2 = np.trace(rho2)
    if abs(tr2) > 1e-12:
        rho2 = rho2 / tr2
    return rho2

def sector_probabilities(rho: np.ndarray, Ps: List[np.ndarray]) -> np.ndarray:
    return np.array([float(np.real(np.trace(P @ rho))) for P in Ps], dtype=float)

def simulate_leakage(p: Toy10Params, eps: float) -> Dict[str, float]:
    """
    Simulate evolution and compute max leakage out of initial sector.
    Leakage = 1 - p_init_sector(t).
    """
    rng = np.random.default_rng(p.seed + int(round(1e6 * eps)) % 99991)
    nS = len(p.sector_labels)
    D = p.d_sector * nS

    Ps = sector_projectors(p.d_sector, nS)
    H = build_hamiltonian(rng, p, eps)

    # Initial pure state in chosen sector
    si0 = p.initial_sector_index
    a = si0 * p.d_sector
    b = (si0 + 1) * p.d_sector
    psi = np.zeros((D,), dtype=complex)
    # random vector inside block
    v = rng.normal(size=p.d_sector) + 1j * rng.normal(size=p.d_sector)
    v = v / (np.linalg.norm(v) + 1e-12)
    psi[a:b] = v
    rho = np.outer(psi, psi.conj())

    steps = int(round(p.t_max / p.dt))
    max_leak = 0.0
    for _ in range(steps):
        probs = sector_probabilities(rho, Ps)
        p0 = probs[si0]
        leak = 1.0 - p0
        if leak > max_leak:
            max_leak = leak
        rho = lindblad_step(rho, H, Ps, p.gamma_dephase, p.dt)

    return {"max_leakage": float(max_leak)}


# -----------------------------
# Boundary force model (plateau) + broadness test
# -----------------------------

def plateau_F(d: np.ndarray, F0: float, d0: float, n: float) -> np.ndarray:
    return F0 / (1.0 + (d / d0) ** n)

def calibrate_F0_to_target(p: Toy10Params) -> float:
    dT = p.target_d_m
    FT = p.target_force_pN * 1e-12
    F0 = FT * (1.0 + (dT / p.d0_m) ** p.n_shape)
    return float(F0)

def broadness_fraction(d: np.ndarray, F: np.ndarray, target_N: float, tol_N: float) -> float:
    ok = np.abs(F - target_N) <= tol_N
    return float(np.mean(ok))


# -----------------------------
# Main sweep
# -----------------------------

def main():
    p = Toy10Params()
    run = create_run(
        toy_slug="toy10_integrator_identity_plus_boundary_allowed_region",
        toy_name="TOY 10 — integrator: identity + boundary allowed region",
        description="Sweep sector-mixing epsilon and boundary amplitude kappa_force; require leakage<=max AND boundary curve broadness + target match.",
        params=asdict(p),
    )

    eps_vals = list(p.eps_values)
    kappa_vals = list(p.kappa_force_values)

    # distance grid for boundary tests
    d = np.geomspace(p.d_min_m, p.d_max_m, p.n_d)
    F0_base = calibrate_F0_to_target(p)

    rows: List[Dict[str, float]] = []
    leak_mat = np.zeros((len(eps_vals), len(kappa_vals)), dtype=float)
    broad_mat = np.zeros_like(leak_mat)
    allowed_mat = np.zeros_like(leak_mat)

    target_N = p.target_force_pN * 1e-12
    tol_N = p.tolerance_pN * 1e-12

    for i, eps in enumerate(eps_vals):
        leak_info = simulate_leakage(p, eps)
        max_leak = leak_info["max_leakage"]

        for j, kappa in enumerate(kappa_vals):
            F0 = kappa * F0_base
            F = plateau_F(d, F0, p.d0_m, p.n_shape)
            F_at_target = float(plateau_F(np.array([p.target_d_m]), F0, p.d0_m, p.n_shape)[0])

            within_frac = broadness_fraction(d, F, target_N, tol_N)
            passes_force_point = (abs(F_at_target - target_N) <= tol_N)
            passes_broad = (within_frac >= p.min_within_band_fraction)

            passes_identity = (max_leak <= p.leakage_max)

            allowed = bool(passes_identity and passes_force_point and passes_broad)

            leak_mat[i, j] = max_leak
            broad_mat[i, j] = within_frac
            allowed_mat[i, j] = 1.0 if allowed else 0.0

            rows.append(
                {
                    "epsilon": float(eps),
                    "kappa_force": float(kappa),
                    "max_leakage": float(max_leak),
                    "passes_identity": float(1.0 if passes_identity else 0.0),
                    "force_at_1mm_pN": float(F_at_target * 1e12),
                    "passes_force_point": float(1.0 if passes_force_point else 0.0),
                    "within_band_fraction": float(within_frac),
                    "passes_broadness": float(1.0 if passes_broad else 0.0),
                    "allowed": float(1.0 if allowed else 0.0),
                }
            )

    run.write_csv("sweep_results.csv", rows)

    # Heatmaps
    eps_arr = np.array(eps_vals, dtype=float)
    k_arr = np.array(kappa_vals, dtype=float)

    def heatmap(mat: np.ndarray, title: str, fname: str, vmin=None, vmax=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[k_arr.min(), k_arr.max(), eps_arr.min(), eps_arr.max()],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("kappa_force (boundary amplitude scale)")
        ax.set_ylabel("epsilon (sector mixing)")
        ax.set_title(title)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.9)
        ax.set_yscale("log")
        save_figure(fig, run.figure_path(fname))

    heatmap(leak_mat, "Toy 10 — max leakage out of initial sector", "leakage_heatmap.png", vmin=float(np.min(leak_mat)), vmax=float(np.max(leak_mat)))
    heatmap(broad_mat, "Toy 10 — boundary broadness fraction within tolerance band", "force_broadness_heatmap.png", vmin=0.0, vmax=1.0)
    heatmap(allowed_mat, "Toy 10 — ALLOWED region (1=allowed)", "allowed_region.png", vmin=0.0, vmax=1.0)

    # Optionally save example curves for a few points
    if p.save_example_curves:
        for (eps_ex, k_ex) in p.example_curve_points:
            F0 = k_ex * F0_base
            F = plateau_F(d, F0, p.d0_m, p.n_shape)
            rel = f"example_curve_eps{eps_ex:.0e}_kappa{k_ex:.2f}.csv"
            run.write_csv(rel, [{"d_m": float(dd), "F_pN": float(ff * 1e12)} for dd, ff in zip(d, F)])

    # Summary
    allowed_any = bool(np.any(allowed_mat > 0.5))
    # Find best (largest epsilon) allowed if any
    best_allowed = None
    if allowed_any:
        # among allowed, maximize epsilon then minimize abs(kappa-1)
        candidates = [r for r in rows if r["allowed"] > 0.5]
        candidates = sorted(candidates, key=lambda r: (r["epsilon"], -abs(r["kappa_force"] - 1.0)))
        best_allowed = candidates[-1]

    summary = {
        "status": "pass" if allowed_any else "fail",
        "interpretation": (
            "pass=exists (epsilon,kappa_force) where identity leakage is below threshold AND boundary force curve meets target+broadness. "
            "fail=no overlap; pillars conflict under this integrator."
        ),
        "criteria": {
            "leakage_max": p.leakage_max,
            "tolerance_pN": p.tolerance_pN,
            "min_within_band_fraction": p.min_within_band_fraction,
        },
        "grid": {"n_eps": len(eps_vals), "n_kappa": len(kappa_vals), "eps_values": eps_vals, "kappa_force_values": kappa_vals},
        "best_allowed_point": best_allowed,
        "notes": [
            "If FAIL: reduce epsilon (stronger superselection) or increase gamma_dephase (Zeno stabilization) to recover identity stability, or relax boundary tolerance/broadness.",
            "If PASS: this identifies a quantitative compatibility region; you can reinterpret epsilon as effective sector mixing induced by coupling and bound it experimentally.",
        ],
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
