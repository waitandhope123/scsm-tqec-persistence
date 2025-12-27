"""
TOY 12B — Boundary Model Selection (FINAL ROBUST VERSION)

Auto-discovers Toy12A outputs under:
  scsm/outputs/toy12*_boundary*_battlefield*/run_*/data/

This version is immune to:
- spaces vs underscores
- display-name confusion
- multiple Toy12 runs
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scsm_utils.io_utils import create_run


# ============================================================
# Parameters
# ============================================================

@dataclass
class Toy12BParams:
    seed: int = 2041
    candidate_models: Tuple[str, ...] = (
        "plateau",
        "yukawa",
        "power",
        "patch",
        "plateau_plus_patch",
    )
    n_starts: int = 25
    n_local_steps: int = 80


# ============================================================
# Robust discovery of Toy12A data
# ============================================================

def find_toy12a_data() -> Path:
    outputs = Path.cwd() / "outputs"
    if not outputs.exists():
        raise RuntimeError(f"'outputs/' not found under {Path.cwd()}")

    # find any toy12 boundary battlefield folder (underscores or spaces)
    candidates = []
    for p in outputs.iterdir():
        if p.is_dir() and "toy12" in p.name and "battlefield" in p.name:
            candidates.extend(p.glob("run_*/data"))

    if not candidates:
        raise RuntimeError(
            "No Toy12A data folders found under outputs/. "
            "Expected something like outputs/toy12_boundary_battlefield/run_*/data/"
        )

    # choose most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    data_dir = candidates[0]

    if not (data_dir / "datasets_index.csv").exists():
        raise RuntimeError(f"datasets_index.csv missing in {data_dir}")
    if not (data_dir / "datasets").exists():
        raise RuntimeError(f"datasets/ missing in {data_dir}")

    print(f"[Toy12B] Using Toy12A data directory:\n  {data_dir}")
    return data_dir


# ============================================================
# CSV helpers
# ============================================================

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def read_dataset(path: Path):
    rows = read_csv(path)
    d = np.array([float(r["d_m"]) for r in rows])
    y = np.array([float(r["F_pN"]) for r in rows]) * 1e-12
    return d, y


# ============================================================
# Models
# ============================================================

def plateau_F(d, F0, d0, n):
    return F0 / (1 + (d / d0) ** n)

def yukawa_F(d, A, lam):
    return A * np.exp(-d / lam) / (d**2 + 1e-18)

def power_F(d, A, dc, m):
    return A / ((d + dc) ** m + 1e-18)

def patch_F(d, A, B, dc, m):
    return A / ((d + dc) ** m + 1e-18) + B / ((d + dc) ** 2 + 1e-18)

def plateau_plus_patch_F(d, F0, d0, n, A, B, dc, m):
    return plateau_F(d, F0, d0, n) + patch_F(d, A, B, dc, m)


# ============================================================
# Metrics
# ============================================================

def sse(y, yhat):
    return float(np.sum((y - yhat) ** 2))

def bic(n, k, rss):
    return n * math.log(max(rss / n, 1e-30)) + k * math.log(n)


# ============================================================
# Fitting
# ============================================================

def fit_model(rng, name, d, y, p):
    yscale = np.median(np.abs(y)) + 1e-30
    dref = np.median(d)

    if name == "plateau":
        bounds = dict(F0=(1e-3*yscale, 1e3*yscale),
                      d0=(0.2e-3, 12e-3),
                      n=(1.5, 12))
        model = lambda dd, **q: plateau_F(dd, **q)
        k = 3

    elif name == "yukawa":
        bounds = dict(A=(1e-6*yscale*dref**2, 1e4*yscale*dref**2),
                      lam=(0.2e-3, 12e-3))
        model = lambda dd, **q: yukawa_F(dd, **q)
        k = 2

    elif name == "power":
        bounds = dict(A=(1e-6*yscale*dref**3, 1e4*yscale*dref**3),
                      dc=(0, 2e-3),
                      m=(0.8, 10))
        model = lambda dd, **q: power_F(dd, **q)
        k = 3

    elif name == "patch":
        bounds = dict(A=(1e-8*yscale, 1e5*yscale),
                      B=(1e-8*yscale, 1e5*yscale),
                      dc=(0, 2e-3),
                      m=(0.5, 7))
        model = lambda dd, **q: patch_F(dd, **q)
        k = 4

    else:
        bounds = dict(F0=(1e-3*yscale, 1e3*yscale),
                      d0=(0.2e-3, 12e-3),
                      n=(1.5, 12),
                      A=(1e-8*yscale, 1e6*yscale),
                      B=(1e-8*yscale, 1e6*yscale),
                      dc=(0, 2e-3),
                      m=(0.5, 7))
        model = lambda dd, **q: plateau_plus_patch_F(dd, **q)
        k = 7

    best = (None, float("inf"))
    for _ in range(p.n_starts):
        params = {k: rng.uniform(*v) for k, v in bounds.items()}
        for _ in range(p.n_local_steps):
            for key, (lo, hi) in bounds.items():
                trial = dict(params)
                trial[key] = np.clip(
                    params[key] + rng.normal(scale=0.15*(hi-lo)), lo, hi
                )
                val = sse(y, model(d, **trial))
                if val < best[1]:
                    params = trial
                    best = (dict(params), val)

    return best[0], best[1], bic(len(d), k, best[1])


# ============================================================
# Main
# ============================================================

def main():
    data_root = find_toy12a_data()

    p = Toy12BParams()
    run = create_run(
        toy_slug="toy12b_boundary_model_selection",
        toy_name="Toy12B Boundary Model Selection",
        description="Robust model selection over Toy12A outputs",
        params=asdict(p),
    )

    index = read_csv(data_root / "datasets_index.csv")
    dsets = data_root / "datasets"

    rng = np.random.default_rng(p.seed)

    truth_models = sorted({r["truth"] for r in index})
    candidates = p.candidate_models

    wins = {t: {m: 0 for m in candidates} for t in truth_models}

    for row in index:
        d, y = read_dataset(dsets / f"{row['dataset_id']}.csv")

        best_bic = float("inf")
        best_model = None

        for m in candidates:
            _, rss, bic_val = fit_model(rng, m, d, y, p)
            if bic_val < best_bic:
                best_bic = bic_val
                best_model = m

        wins[row["truth"]][best_model] += 1

    run.save_summary({
        "status": "ok",
        "interpretation": "Toy12B completed successfully. Boundary model discrimination evaluated.",
        "wins": wins,
        "truth_models": truth_models,
        "candidates": candidates,
    })


if __name__ == "__main__":
    main()
