"""
SCSM TOYBENCH — I/O + OUTPUT CONTRACT UTILITIES

This module standardizes how every toy:
- creates an output directory
- writes params + summaries (JSON)
- writes tabular data (CSV)
- writes arrays/series/grids (HDF5)

OUTPUT CONTRACT (fixed):
outputs/<toy_slug>/run_<timestamp>__<run_id>/
  params.json
  summary.json
  README.md
  data/      (csv, h5, npz)
  figures/   (png, svg)

Every toy should call:
    run = create_run(toy_slug="toy01_chi_bio_open_system", toy_name="TOY 01 — ...", description="...")
and then use:
    run.save_params(...)
    run.save_summary(...)
    run.write_csv(...)
    run.write_h5(...)
    run.save_text(...)
"""

from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import h5py
except Exception as e:
    h5py = None  # type: ignore


Jsonable = Union[str, int, float, bool, None, Dict[str, "Jsonable"], List["Jsonable"]]


def _now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _safe_json(obj: Any) -> Jsonable:
    """
    Convert common scientific/python objects to JSON-serializable types.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return _safe_json(asdict(obj))
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        # avoid dumping massive arrays to JSON; store shape/dtype only
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, Mapping):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    # fallback: string repr
    return str(obj)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Union[str, Path], obj: Any, *, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(_safe_json(obj), f, indent=indent, sort_keys=True)


def write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        f.write(text)


def write_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """
    Write a list of dict-like rows to CSV. Infers columns from union of keys if fieldnames not provided.
    """
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        # create empty file with header if fieldnames provided
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames) if fieldnames else [])
            if fieldnames:
                writer.writeheader()
        return

    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                ks = str(k)
                if ks not in seen:
                    seen.add(ks)
                    keys.append(ks)
        fieldnames = keys

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            # json-safe scalars
            out2 = {k: (_safe_json(v) if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in out.items()}
            writer.writerow(out2)


def require_h5py() -> None:
    if h5py is None:
        raise RuntimeError(
            "h5py is required for HDF5 outputs but is not installed. "
            "Install with: pip install h5py"
        )


def write_h5(
    path: Union[str, Path],
    arrays: Mapping[str, np.ndarray],
    *,
    attrs: Optional[Mapping[str, Any]] = None,
    compression: Optional[str] = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Write named numpy arrays to an HDF5 file with optional root attrs.
    """
    require_h5py()
    p = Path(path)
    ensure_dir(p.parent)

    with h5py.File(p, "w") as f:  # type: ignore
        if attrs:
            for k, v in attrs.items():
                f.attrs[str(k)] = _safe_json(v)  # h5py will store scalar/strings OK
        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if compression:
                f.create_dataset(
                    name,
                    data=arr,
                    compression=compression,
                    compression_opts=compression_opts,
                    shuffle=True,
                )
            else:
                f.create_dataset(name, data=arr)


class RunContext:
    """
    A single toy run output bundle.
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        toy_slug: str,
        toy_name: str,
        description: str,
        run_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.base_dir = Path(base_dir)
        self.toy_slug = toy_slug
        self.toy_name = toy_name
        self.description = description

        self.run_id = run_id or uuid.uuid4().hex[:10]
        self.timestamp = timestamp or _now_timestamp()

        self.run_dir = self.base_dir / toy_slug / f"run_{self.timestamp}__{self.run_id}"
        self.data_dir = self.run_dir / "data"
        self.fig_dir = self.run_dir / "figures"

        ensure_dir(self.data_dir)
        ensure_dir(self.fig_dir)

    def save_params(self, params: Any) -> None:
        write_json(self.run_dir / "params.json", params)

    def save_summary(self, summary: Any) -> None:
        write_json(self.run_dir / "summary.json", summary)

    def save_text(self, filename: str, text: str) -> None:
        write_text(self.run_dir / filename, text)

    def write_csv(self, relpath: str, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
        write_csv(self.data_dir / relpath, rows, fieldnames=fieldnames)

    def write_h5(self, relpath: str, arrays: Mapping[str, np.ndarray], attrs: Optional[Mapping[str, Any]] = None) -> None:
        write_h5(self.data_dir / relpath, arrays, attrs=attrs)

    def figure_path(self, filename: str) -> Path:
        return self.fig_dir / filename

    def data_path(self, filename: str) -> Path:
        return self.data_dir / filename


def create_run(
    *,
    toy_slug: str,
    toy_name: str,
    description: str,
    base_dir: Union[str, Path] = "outputs",
    params: Optional[Any] = None,
    extra_readme: Optional[str] = None,
) -> RunContext:
    """
    Create a run directory and write a README.md documenting what this run is.
    """
    run = RunContext(base_dir=base_dir, toy_slug=toy_slug, toy_name=toy_name, description=description)

    readme = [
        f"# {toy_name}",
        "",
        f"**toy_slug:** `{toy_slug}`",
        f"**run_id:** `{run.run_id}`",
        f"**timestamp:** `{run.timestamp}`",
        "",
        "## Description",
        description.strip(),
        "",
        "## Output Contract",
        "This run follows the standard toybench output layout:",
        "",
        "```text",
        f"{run.run_dir}/",
        "  params.json",
        "  summary.json",
        "  README.md",
        "  data/",
        "  figures/",
        "```",
        "",
    ]
    if extra_readme:
        readme.extend(["## Notes", extra_readme.strip(), ""])

    run.save_text("README.md", "\n".join(readme))

    if params is not None:
        run.save_params(params)

    return run
