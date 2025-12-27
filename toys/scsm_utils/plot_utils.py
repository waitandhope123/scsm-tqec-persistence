"""
SCSM TOYBENCH — PLOTTING UTILITIES

Standard plotting helpers:
- consistent saving (PNG + SVG)
- safe figure closure
- common plot patterns (time series, spectrum, scan curves)

All toys should save plots into:
    run.figure_path("...png") / run.figure_path("...svg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, png_path: Path, *, also_svg: bool = True, dpi: int = 160) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if also_svg:
        svg_path = png_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(
    t: np.ndarray,
    series: Sequence[Tuple[str, np.ndarray]],
    *,
    title: str,
    xlabel: str = "t (s)",
    ylabel: str = "value",
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for label, y in series:
        ax.plot(t, y, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    return fig


def plot_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    title: str,
    xlabel: str = "frequency (Hz)",
    ylabel: str = "power (a.u.)",
    f_mark: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(freqs, power)
    if f_mark is not None:
        fmin, fmax = f_mark
        ax.axvspan(fmin, fmax, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig


def plot_scan(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return fig
