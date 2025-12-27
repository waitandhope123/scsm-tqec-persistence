"""
TOY 09 — SUPERSELECTION / SECTOR-MIXING SUPPRESSION (AQG-STYLE IDENTITY TEST)

Purpose
-------
You’ve now seen that “identity as a localized skyrmion lump” is not robust in our
field toys. Toy 09 pivots to what your AQG framing *actually says*: identity is a
(superselection) sector label Q ∈ ℤ, i.e. dynamics are block-diagonal in Q, or
sector-mixing is strongly suppressed.

This toy builds a finite-dimensional quantum model with:
- Multiple sectors (Q labels), each with internal states
- Hamiltonian H = H_block + ε H_mix
- Optional open-system Lindblad dephasing (to mimic environment)
- We measure "leakage" between sectors vs ε and environment strength

If leakage stays negligible across reasonable ε and timescales, you have a concrete,
data-backed version of “identity persistence” as sector stability (not soliton survival).

Model
-----
Sectors Q ∈ {-1, 0, +1}, each with dim d_sector.
Total dim D = 3 * d_sector.

H_block: random Hermitian blocks per sector (internal dynamics)
H_mix:  connects neighboring sectors (Q↔Q±1) with random couplings

Open system (optional):
- dephasing in the *sector basis* to represent environment monitoring Q (which
  tends to *stabilize* superselection, quantum-Zeno style)
Lindblad operators: L_Q = sqrt(gamma) * Π_Q (projector onto sector Q)

We evolve density matrix ρ(t) by Lindblad master equation:
  dρ/dt = -i[H,ρ] + Σ_Q (L_Q ρ L_Q† - 1/2{L_Q†L_Q, ρ})

Outputs
-------
- data/epsilon_sweep_summary.csv
- data/time_series_leakage_eps_<...>.csv   (for a few representative epsilons)
- data/transition_matrix_eps_<...>.csv     (final sector-to-sector probabilities)
- figures/leakage_vs_time.(png, svg)
- figures/leakage_vs_epsilon.(png, svg)
- figures/transition_matrix_heatmap.(png, svg)
- summary.json

Run
---
python toy09_superselection_sector_mixing.py
"""

from __future__ import annotations

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
class Toy09Params:
    # Sector structure
    sector_labels: Tuple[int, ...] = (-1, 0, 1)
    d_sector: int = 6  # internal dimension per sector
    seed: int = 1337

    # Hamiltonian scaling
    block_energy_scale: float = 1.0
    mix_energy_scale: float = 1.0

    # Lindblad (sector-basis dephasing)
    # gamma=0 => unitary only
    gamma_dephase: float = 1.0

    # Time evolution
    t_max: float = 20.0
    dt: float = 0.01

    # Epsilon sweep (sector mixing strength)
    eps_values: Tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)

    # Which epsilons to save full time-series CSV for
    eps_time_series: Tuple[float, ...] = (0.0, 1e-3, 1e-2, 1e-1)

    # Initial condition: start fully in one sector (identity label)
    initial_sector: int = 0
    initial_pure_state: bool = True  # if False, use maximally mixed within that sector

    # Pass/fail criterion (practical superselection)
    # "Leakage" = 1 - P(stay in initial sector) at final time
    max_allowed_leakage: float = 0.01  # 1% leakage
    pass_epsilon_threshold: float = 1e-2  # we "pass" if leakage<=max_allowed for eps <= this


# -----------------------------
# Helpers: matrices, projectors
# -----------------------------

def hermitian_random(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (A + A.conj().T) / 2.0
    # normalize rough spectral scale
    H = H / (np.linalg.norm(H, ord="fro") / math.sqrt(n) + 1e-12)
    return scale * H

def build_projectors(sector_labels: Tuple[int, ...], d_sector: int) -> Dict[int, np.ndarray]:
    P = {}
    for i, q in enumerate(sector_labels):
        D = len(sector_labels) * d_sector
        M = np.zeros((D, D), dtype=complex)
        sl = slice(i * d_sector, (i + 1) * d_sector)
        M[sl, sl] = np.eye(d_sector, dtype=complex)
        P[q] = M
    return P

def sector_probabilities(rho: np.ndarray, projectors: Dict[int, np.ndarray]) -> Dict[int, float]:
    probs = {}
    for q, Pq in projectors.items():
        probs[q] = float(np.real(np.trace(Pq @ rho)))
    # numerical cleanup
    s = sum(probs.values())
    if s != 0:
        for q in probs:
            probs[q] /= s
    return probs

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def lindblad_rhs(rho: np.ndarray, H: np.ndarray, Ls: List[np.ndarray]) -> np.ndarray:
    drho = -1j * commutator(H, rho)
    for L in Ls:
        LdL = L.conj().T @ L
        drho += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return drho

def rk4_step(rho: np.ndarray, dt: float, H: np.ndarray, Ls: List[np.ndarray]) -> np.ndarray:
    k1 = lindblad_rhs(rho, H, Ls)
    k2 = lindblad_rhs(rho + 0.5 * dt * k1, H, Ls)
    k3 = lindblad_rhs(rho + 0.5 * dt * k2, H, Ls)
    k4 = lindblad_rhs(rho + dt * k3, H, Ls)
    rho2 = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    # enforce Hermiticity + trace 1
    rho2 = (rho2 + rho2.conj().T) / 2.0
    tr = np.real(np.trace(rho2))
    if tr != 0:
        rho2 = rho2 / tr
    return rho2


# -----------------------------
# Build Hamiltonians
# -----------------------------

def build_H_block(rng: np.random.Generator, sector_labels: Tuple[int, ...], d_sector: int, scale: float) -> np.ndarray:
    D = len(sector_labels) * d_sector
    H = np.zeros((D, D), dtype=complex)
    for i, _q in enumerate(sector_labels):
        sl = slice(i * d_sector, (i + 1) * d_sector)
        H[sl, sl] = hermitian_random(rng, d_sector, scale)
    return H

def build_H_mix(rng: np.random.Generator, sector_labels: Tuple[int, ...], d_sector: int, scale: float) -> np.ndarray:
    """
    Build mixing that couples neighboring sectors (by order in sector_labels list).
    """
    D = len(sector_labels) * d_sector
    Hm = np.zeros((D, D), dtype=complex)

    for i in range(len(sector_labels) - 1):
        a = slice(i * d_sector, (i + 1) * d_sector)
        b = slice((i + 1) * d_sector, (i + 2) * d_sector)
        C = rng.normal(size=(d_sector, d_sector)) + 1j * rng.normal(size=(d_sector, d_sector))
        C = C / (np.linalg.norm(C, ord="fro") / math.sqrt(d_sector) + 1e-12)
        C = scale * C
        Hm[a, b] = C
        Hm[b, a] = C.conj().T

    # ensure Hermitian (it is by construction)
    return Hm


# -----------------------------
# Initial state
# -----------------------------

def initial_rho(sector_labels: Tuple[int, ...], d_sector: int, q0: int, pure: bool, rng: np.random.Generator) -> np.ndarray:
    D = len(sector_labels) * d_sector
    rho = np.zeros((D, D), dtype=complex)
    i0 = list(sector_labels).index(q0)
    sl = slice(i0 * d_sector, (i0 + 1) * d_sector)

    if pure:
        v = rng.normal(size=(d_sector,)) + 1j * rng.normal(size=(d_sector,))
        v = v / (np.linalg.norm(v) + 1e-12)
        psi = np.zeros((D,), dtype=complex)
        psi[sl] = v
        rho = np.outer(psi, psi.conj())
    else:
        rho[sl, sl] = np.eye(d_sector, dtype=complex) / d_sector

    return rho


# -----------------------------
# Main
# -----------------------------

def main():
    p = Toy09Params()
    run = create_run(
        toy_slug="toy09_superselection_sector_mixing",
        toy_name="TOY 09 — Superselection / sector mixing suppression",
        description="Finite-dimensional sector model; sweep mixing strength ε; quantify leakage; optional sector-dephasing Lindblad stabilizer.",
        params=asdict(p),
    )

    rng = np.random.default_rng(p.seed)

    # Build projectors
    P = build_projectors(p.sector_labels, p.d_sector)

    # Hamiltonian components
    H_block = build_H_block(rng, p.sector_labels, p.d_sector, p.block_energy_scale)
    H_mix = build_H_mix(rng, p.sector_labels, p.d_sector, p.mix_energy_scale)

    # Lindblad operators: sector dephasing (monitor Q)
    Ls = []
    if p.gamma_dephase > 0:
        for q in p.sector_labels:
            Ls.append(math.sqrt(p.gamma_dephase) * P[q])

    # time grid
    n_steps = int(round(p.t_max / p.dt))
    t_grid = np.linspace(0.0, p.t_max, n_steps + 1)

    # initial state
    rho0 = initial_rho(p.sector_labels, p.d_sector, p.initial_sector, p.initial_pure_state, rng)

    # Sweep eps
    sweep_rows: List[Dict[str, object]] = []
    rep_time_series_cache: Dict[float, Dict[str, np.ndarray]] = {}
    rep_transition_cache: Dict[float, np.ndarray] = {}

    for eps in p.eps_values:
        H = H_block + float(eps) * H_mix

        rho = rho0.copy()
        probs0 = sector_probabilities(rho, P)
        q0_prob0 = probs0[p.initial_sector]

        # store time series if requested
        store_ts = any(abs(eps - e) < 1e-15 for e in p.eps_time_series)
        if store_ts:
            ts_probs = {q: np.zeros_like(t_grid) for q in p.sector_labels}

        for k, t in enumerate(t_grid):
            probs = sector_probabilities(rho, P)
            if store_ts:
                for q in p.sector_labels:
                    ts_probs[q][k] = probs[q]
            if k < len(t_grid) - 1:
                rho = rk4_step(rho, p.dt, H, Ls)

        probsT = sector_probabilities(rho, P)
        leakage = 1.0 - probsT[p.initial_sector]

        sweep_rows.append(
            {
                "epsilon": float(eps),
                "gamma_dephase": float(p.gamma_dephase),
                "P_init_sector_t0": float(q0_prob0),
                "P_init_sector_tmax": float(probsT[p.initial_sector]),
                "leakage_tmax": float(leakage),
            }
        )

        # Build final sector-to-sector "transition matrix" by repeating with each sector as initial
        # (quick but informative; still small dimension)
        T = np.zeros((len(p.sector_labels), len(p.sector_labels)), dtype=float)
        for i, q_init in enumerate(p.sector_labels):
            rho_init = initial_rho(p.sector_labels, p.d_sector, q_init, p.initial_pure_state, rng)
            rho = rho_init.copy()
            for k in range(len(t_grid) - 1):
                rho = rk4_step(rho, p.dt, H, Ls)
            probs_end = sector_probabilities(rho, P)
            for j, q_end in enumerate(p.sector_labels):
                T[i, j] = probs_end[q_end]
        rep_transition_cache[float(eps)] = T

        if store_ts:
            rep_time_series_cache[float(eps)] = {q: ts_probs[q] for q in p.sector_labels}

            # write time series CSV
            rows = []
            for i, t in enumerate(t_grid):
                row = {"t": float(t), "epsilon": float(eps)}
                for q in p.sector_labels:
                    row[f"P_Q{q}"] = float(ts_probs[q][i])
                row["leakage_from_init"] = float(1.0 - ts_probs[p.initial_sector][i])
                rows.append(row)
            run.write_csv(f"time_series_leakage_eps_{eps:g}.csv", rows)

            # write transition matrix CSV
            mat_rows = []
            for i, q_init in enumerate(p.sector_labels):
                r = {"Q_init": int(q_init)}
                for j, q_end in enumerate(p.sector_labels):
                    r[f"Q_end_{q_end}"] = float(T[i, j])
                mat_rows.append(r)
            run.write_csv(f"transition_matrix_eps_{eps:g}.csv", mat_rows)

    run.write_csv("epsilon_sweep_summary.csv", sweep_rows)

    # -----------------------------
    # Plots
    # -----------------------------

    # 1) Leakage vs epsilon (final time)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    eps_arr = np.array([r["epsilon"] for r in sweep_rows], dtype=float)
    leak_arr = np.array([r["leakage_tmax"] for r in sweep_rows], dtype=float)
    ax1.plot(eps_arr, leak_arr, marker="o")
    ax1.axhline(p.max_allowed_leakage, linestyle="--", alpha=0.5, label="max_allowed_leakage")
    ax1.axvline(p.pass_epsilon_threshold, linestyle="--", alpha=0.5, label="pass_epsilon_threshold")
    ax1.set_xscale("log" if np.all(eps_arr[1:] > 0) else "linear")
    ax1.set_xlabel("epsilon (sector mixing strength)")
    ax1.set_ylabel("leakage at t_max = 1 - P(stay in init sector)")
    ax1.set_title("Toy 09 — Practical superselection: leakage vs epsilon")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    save_figure(fig1, run.figure_path("leakage_vs_epsilon.png"))

    # 2) Leakage vs time for representative eps values
    if len(rep_time_series_cache) > 0:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        for eps, series in rep_time_series_cache.items():
            leakage_t = 1.0 - series[p.initial_sector]
            ax2.plot(t_grid, leakage_t, label=f"eps={eps:g}")
        ax2.set_xlabel("t")
        ax2.set_ylabel("leakage from initial sector")
        ax2.set_title("Toy 09 — Leakage vs time (selected eps)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        save_figure(fig2, run.figure_path("leakage_vs_time.png"))

    # 3) Transition matrix heatmap for one representative epsilon (largest one in eps_time_series if available)
    eps_for_heat = max(rep_transition_cache.keys()) if len(rep_transition_cache) else float(p.eps_values[-1])
    T = rep_transition_cache[eps_for_heat]

    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_subplot(111)
    im = ax3.imshow(T, origin="lower", vmin=0.0, vmax=1.0)
    ax3.set_xticks(range(len(p.sector_labels)))
    ax3.set_yticks(range(len(p.sector_labels)))
    ax3.set_xticklabels([str(q) for q in p.sector_labels])
    ax3.set_yticklabels([str(q) for q in p.sector_labels])
    ax3.set_xlabel("Q_end")
    ax3.set_ylabel("Q_init")
    ax3.set_title(f"Toy 09 — Sector transition matrix at eps={eps_for_heat:g}")
    fig3.colorbar(im, ax=ax3, label="P(Q_end | Q_init)")
    save_figure(fig3, run.figure_path("transition_matrix_heatmap.png"))

    # -----------------------------
    # Pass/fail for the toy
    # -----------------------------
    # We "pass" if for all eps <= pass_epsilon_threshold, leakage <= max_allowed_leakage
    passed = True
    worst = {"epsilon": None, "leakage": -1.0}
    for r in sweep_rows:
        eps = float(r["epsilon"])
        leak = float(r["leakage_tmax"])
        if eps <= p.pass_epsilon_threshold:
            if leak > p.max_allowed_leakage:
                passed = False
            if leak > worst["leakage"]:
                worst = {"epsilon": eps, "leakage": leak}

    summary = {
        "status": "pass" if passed else "fail",
        "interpretation": "pass=practical superselection holds: leakage stays below threshold for epsilon up to pass_epsilon_threshold",
        "criteria": {
            "max_allowed_leakage": p.max_allowed_leakage,
            "pass_epsilon_threshold": p.pass_epsilon_threshold,
        },
        "params": asdict(p),
        "worst_case_within_threshold": worst,
        "note": "If fail, increase gamma_dephase (environment monitoring Q) to see Zeno-like stabilization, or reduce eps to define an effective mixing bound.",
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
