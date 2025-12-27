"""
TOY 08d — BABY-SKYRME STABILITY WITH RELAXATION + POTENTIAL + HIGHER RESOLUTION

Purpose
-------
Toy 08c failed to find stable Q sectors even at noise=0. That can happen if:
- the lattice is too coarse (N too small),
- the time step is too large for stable gradient flow,
- the configuration is not first relaxed to a metastable local minimum,
- a stabilizing potential is required (common in standard baby-Skyrme models).

Toy 08d is the "last fair shot" for the identity=topology mechanism in a practical toy:
1) Higher resolution (default N=128)
2) Small stabilizing potential k_potential > 0
3) Two-stage run:
   (a) RELAXATION: pure gradient flow (noise=0) to a local minimum
   (b) HOLD TEST: continue gradient flow (noise configurable) and check Q persistence

If this STILL shows unwinding at noise=0 after relaxation, then in this toy suite:
- "persistent identity via skyrmion sectors" is not supported as a robust mechanism.

Outputs
-------
- data/relax_QE.csv
- data/hold_QE.csv
- data/snapshots.h5
- figures/relax_QE.(png, svg)
- figures/hold_QE.(png, svg)
- figures/nz_snapshots.(png, svg)
- summary.json
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from scsm_utils.io_utils import create_run
from scsm_utils.plot_utils import save_figure


# -----------------------------
# Parameters
# -----------------------------

@dataclass
class Toy08dParams:
    # Grid (higher resolution)
    N: int = 128
    L: float = 10.0

    # Relaxation stage
    dt_relax: float = 0.004
    steps_relax: int = 4000

    # Hold stage
    dt_hold: float = 0.004
    steps_hold: int = 4000
    noise_sigma_hold: float = 0.0

    # Energy weights
    k_grad: float = 1.0
    k_skyrme: float = 1.0
    k_potential: float = 0.2

    # Snapshotting
    snapshot_every: int = 400

    # Topology / event detection
    Q_loss_threshold: float = 0.5

    # Seeds
    n_seeds: int = 3
    base_seed: int = 777

    # ✅ ADD THIS LINE
    target_unwound_fraction: float = 0.25


# -----------------------------
# Field + finite differences
# -----------------------------

def normalize_field(n: np.ndarray) -> np.ndarray:
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

def init_baby_skyrmion(N: int, L: float, *, seed_jitter: int = 0) -> np.ndarray:
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    y = np.linspace(-L/2, L/2, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    r = np.sqrt(X**2 + Y**2) + 1e-12
    theta = np.arctan2(Y, X)

    R0 = L / 6.0
    f = 2.0 * np.arctan2(R0, r)
    nx = np.sin(f) * np.cos(theta)
    ny = np.sin(f) * np.sin(theta)
    nz = np.cos(f)
    n = np.stack([nx, ny, nz], axis=-1)

    # small deterministic jitter to avoid perfectly symmetric artifacts
    if seed_jitter != 0:
        rng = np.random.default_rng(seed_jitter)
        n = n + 1e-3 * rng.normal(size=n.shape)

    return normalize_field(n)

def ddx(n: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(n, -1, axis=0) - np.roll(n, 1, axis=0)) / (2.0 * dx)

def ddy(n: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)) / (2.0 * dx)

def laplacian(n: np.ndarray, dx: float) -> np.ndarray:
    return (
        np.roll(n, -1, axis=0) + np.roll(n, 1, axis=0) +
        np.roll(n, -1, axis=1) + np.roll(n, 1, axis=1) -
        4.0 * n
    ) / (dx * dx)


# -----------------------------
# Observables
# -----------------------------

def compute_Q(n: np.ndarray, dx: float) -> float:
    nx = ddx(n, dx)
    ny = ddy(n, dx)
    B = np.cross(nx, ny)
    density = np.sum(n * B, axis=-1)
    return float(np.sum(density) * (dx * dx) / (4.0 * math.pi))

def compute_energy(n: np.ndarray, dx: float, k_grad: float, k_skyrme: float, k_potential: float) -> float:
    nx = ddx(n, dx)
    ny = ddy(n, dx)

    grad_term = np.sum(nx * nx, axis=-1) + np.sum(ny * ny, axis=-1)
    B = np.cross(nx, ny)
    sk_term = np.sum(B * B, axis=-1)

    nz = n[..., 2]
    pot_term = (1.0 - nz) ** 2

    density = k_grad * grad_term + k_skyrme * sk_term + k_potential * pot_term
    return float(np.sum(density) * (dx * dx))


# -----------------------------
# Functional derivative δE/δn
# -----------------------------

def dE_dn(n: np.ndarray, dx: float, k_grad: float, k_skyrme: float, k_potential: float) -> np.ndarray:
    nx = ddx(n, dx)
    ny = ddy(n, dx)
    B = np.cross(nx, ny)

    # sigma model
    term_sigma = -2.0 * k_grad * laplacian(n, dx)

    # potential
    term_pot = np.zeros_like(n)
    term_pot[..., 2] = 2.0 * k_potential * (n[..., 2] - 1.0)

    # skyrme
    A = np.cross(ny, B)       # ∂y n × B
    C = np.cross(B, nx)       # B × ∂x n
    term_sk = -2.0 * k_skyrme * (ddx(A, dx) + ddy(C, dx))

    return term_sigma + term_sk + term_pot

def project_tangent(n: np.ndarray, v: np.ndarray) -> np.ndarray:
    dot = np.sum(v * n, axis=-1, keepdims=True)
    return v - dot * n


# -----------------------------
# Simulation stage
# -----------------------------

def run_stage(
    n0: np.ndarray,
    *,
    dt: float,
    steps: int,
    dx: float,
    k_grad: float,
    k_skyrme: float,
    k_potential: float,
    noise_sigma: float,
    seed: int,
    snapshot_every: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    n = n0.copy()

    times = np.zeros(steps + 1, dtype=float)
    Qs = np.zeros(steps + 1, dtype=float)
    Es = np.zeros(steps + 1, dtype=float)

    snap_ts: List[float] = []
    snap_nz: List[np.ndarray] = []

    event_time = None

    for k in range(steps + 1):
        t = k * dt
        times[k] = t
        Qs[k] = compute_Q(n, dx)
        Es[k] = compute_energy(n, dx, k_grad, k_skyrme, k_potential)

        if k % snapshot_every == 0 or k == steps:
            snap_ts.append(float(t))
            snap_nz.append(n[..., 2].copy())

        if event_time is None and abs(Qs[k]) < 0.5:
            event_time = float(t)

        if k == steps:
            break

        grad = dE_dn(n, dx, k_grad, k_skyrme, k_potential)
        flow = -project_tangent(n, grad)

        if noise_sigma > 0:
            noise = rng.normal(size=n.shape)
            flow = flow + math.sqrt(dt) * noise_sigma * project_tangent(n, noise)

        n = n + dt * flow
        n = normalize_field(n)

        if not np.all(np.isfinite(n)):
            return {
                "times": times[: k + 1],
                "Q": Qs[: k + 1],
                "E": Es[: k + 1],
                "n_final": n,
                "event_time": event_time,
                "snap_ts": np.array(snap_ts, dtype=float),
                "snap_nz": np.stack(snap_nz, axis=0),
                "stable_numeric": False,
            }

    return {
        "times": times,
        "Q": Qs,
        "E": Es,
        "n_final": n,
        "event_time": event_time,
        "snap_ts": np.array(snap_ts, dtype=float),
        "snap_nz": np.stack(snap_nz, axis=0),
        "stable_numeric": True,
    }


def first_crossing_time(times: np.ndarray, Q: np.ndarray, thr: float) -> float:
    m = np.abs(Q) < thr
    if np.any(m):
        return float(times[int(np.argmax(m))])
    return float("inf")


# -----------------------------
# Main
# -----------------------------

def main():
    p = Toy08dParams()
    run = create_run(
        toy_slug="toy08d_baby_skyrme_relax_and_hold",
        toy_name="TOY 08d — Baby-Skyrme relax + hold stability",
        description="High-res + potential + relaxation; then hold test for Q persistence.",
        params=asdict(p),
    )

    dx = p.L / p.N

    per_seed_summaries = []
    all_relax_rows = []
    all_hold_rows = []

    # We'll store snapshots only for the first seed to keep file sizes reasonable.
    stored_snapshots = None

    for i in range(p.n_seeds):
        seed = p.base_seed + i
        n0 = init_baby_skyrmion(p.N, p.L, seed_jitter=seed)

        # --- Relaxation (noise=0) ---
        relax = run_stage(
            n0,
            dt=p.dt_relax,
            steps=p.steps_relax,
            dx=dx,
            k_grad=p.k_grad,
            k_skyrme=p.k_skyrme,
            k_potential=p.k_potential,
            noise_sigma=0.0,
            seed=seed + 10_000,
            snapshot_every=p.snapshot_every,
        )

        # save relax CSV
        for t, qv, ev in zip(relax["times"], relax["Q"], relax["E"]):  # type: ignore
            all_relax_rows.append({"seed": seed, "t": float(t), "Q": float(qv), "E": float(ev)})

        # --- Hold stage (noise configurable) ---
        hold = run_stage(
            relax["n_final"],  # type: ignore
            dt=p.dt_hold,
            steps=p.steps_hold,
            dx=dx,
            k_grad=p.k_grad,
            k_skyrme=p.k_skyrme,
            k_potential=p.k_potential,
            noise_sigma=p.noise_sigma_hold,
            seed=seed + 20_000,
            snapshot_every=p.snapshot_every,
        )

        for t, qv, ev in zip(hold["times"], hold["Q"], hold["E"]):  # type: ignore
            all_hold_rows.append({"seed": seed, "t": float(t), "Q": float(qv), "E": float(ev)})

        tau_relax = first_crossing_time(relax["times"], relax["Q"], p.Q_loss_threshold)  # type: ignore
        tau_hold = first_crossing_time(hold["times"], hold["Q"], p.Q_loss_threshold)    # type: ignore

        per_seed_summaries.append(
            {
                "seed": seed,
                "numeric_stable_relax": bool(relax["stable_numeric"]),
                "numeric_stable_hold": bool(hold["stable_numeric"]),
                "Q_end_relax": float(relax["Q"][-1]),  # type: ignore
                "Q_end_hold": float(hold["Q"][-1]),    # type: ignore
                "tau_Qloss_relax": float(tau_relax),
                "tau_Qloss_hold": float(tau_hold),
                "unwound_in_relax": bool(math.isfinite(tau_relax)),
                "unwound_in_hold": bool(math.isfinite(tau_hold)),
            }
        )

        if i == 0:
            stored_snapshots = {
                "relax_snap_ts": relax["snap_ts"],
                "relax_nz": relax["snap_nz"],
                "hold_snap_ts": hold["snap_ts"],
                "hold_nz": hold["snap_nz"],
                "seed": seed,
            }

    # Write CSVs
    run.write_csv("relax_QE.csv", all_relax_rows)
    run.write_csv("hold_QE.csv", all_hold_rows)

    # Write snapshots
    if stored_snapshots is not None:
        run.write_h5(
            "snapshots.h5",
            arrays={
                "relax_snap_ts": stored_snapshots["relax_snap_ts"],
                "relax_nz": stored_snapshots["relax_nz"],
                "hold_snap_ts": stored_snapshots["hold_snap_ts"],
                "hold_nz": stored_snapshots["hold_nz"],
            },
            attrs={
                "seed": int(stored_snapshots["seed"]),
                "N": p.N,
                "L": p.L,
                "dt_relax": p.dt_relax,
                "steps_relax": p.steps_relax,
                "dt_hold": p.dt_hold,
                "steps_hold": p.steps_hold,
                "noise_sigma_hold": p.noise_sigma_hold,
                "k_grad": p.k_grad,
                "k_skyrme": p.k_skyrme,
                "k_potential": p.k_potential,
            },
        )

        # Snapshot figure (show a few)
        def snapshot_panel(nz_stack: np.ndarray, ts: np.ndarray, title: str, outname: str):
            pick = np.linspace(0, nz_stack.shape[0] - 1, min(nz_stack.shape[0], 6)).astype(int)
            fig = plt.figure(figsize=(10, 4))
            for j, idx in enumerate(pick, start=1):
                ax = fig.add_subplot(1, len(pick), j)
                im = ax.imshow(nz_stack[idx], origin="lower", vmin=-1, vmax=1)
                ax.set_title(f"t={ts[idx]:.2f}")
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(title)
            fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.02)
            save_figure(fig, run.figure_path(outname))

        snapshot_panel(
            stored_snapshots["relax_nz"],
            stored_snapshots["relax_snap_ts"],
            "Relaxation stage: n_z snapshots",
            "nz_snapshots_relax.png",
        )
        snapshot_panel(
            stored_snapshots["hold_nz"],
            stored_snapshots["hold_snap_ts"],
            "Hold stage: n_z snapshots",
            "nz_snapshots_hold.png",
        )

    # Plot QE for relaxation and hold (all seeds)
    def plot_QE(rows: List[Dict[str, object]], title: str, outname: str):
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        seeds = sorted(set(int(r["seed"]) for r in rows))
        for s in seeds:
            rs = [r for r in rows if int(r["seed"]) == s]
            t = np.array([float(r["t"]) for r in rs])
            Q = np.array([float(r["Q"]) for r in rs])
            E = np.array([float(r["E"]) for r in rs])
            ax1.plot(t, Q, label=f"{s}")
            ax2.plot(t, E, label=f"{s}")
        ax1.axhline(p.Q_loss_threshold, linestyle="--", alpha=0.5)
        ax1.axhline(-p.Q_loss_threshold, linestyle="--", alpha=0.5)
        ax1.set_xlabel("t")
        ax1.set_ylabel("Q")
        ax1.set_title("Q(t)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7, ncol=1)

        ax2.set_xlabel("t")
        ax2.set_ylabel("E")
        ax2.set_title("E(t)")
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        save_figure(fig, run.figure_path(outname))

    plot_QE(all_relax_rows, "Toy 08d — Relaxation stage", "relax_QE.png")
    plot_QE(all_hold_rows, "Toy 08d — Hold stage", "hold_QE.png")

    # Summary + pass/fail
    unwound_hold = [s["unwound_in_hold"] for s in per_seed_summaries]
    unwound_fraction_hold = float(np.mean(unwound_hold))

    passes = (unwound_fraction_hold <= p.target_unwound_fraction)

    summary = {
        "status": "pass" if passes else "fail",
        "interpretation": "pass=after relaxation, Q remains stable during hold stage for most seeds (noise_sigma_hold may be 0 for baseline)",
        "params": asdict(p),
        "metrics": {
            "unwound_fraction_hold": unwound_fraction_hold,
            "per_seed": per_seed_summaries,
        },
        "notes": [
            "If this fails at noise_sigma_hold=0, reduce dt further (e.g., 0.002) and/or increase k_skyrme or k_potential.",
            "If it passes at noise=0, next step is to increase noise_sigma_hold gradually and map robustness margins.",
        ],
    }
    run.save_summary(summary)


if __name__ == "__main__":
    main()
