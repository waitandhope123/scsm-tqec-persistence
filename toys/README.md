# SCSM Phase-I Toy Models

This directory contains the **adversarial toy models** used to stress-test,
refine, and (where necessary) eliminate claims made by SCSM during Phase-I.

The toy program is intentionally transparent: both positive and negative
results are preserved.

---

## How to Read This Directory

- Each `toyXX_*` folder is a **self-contained experiment**.
- Toys were allowed to break claims; failure is treated as a valid outcome.
- Earlier toys are retained for provenance and context.
- **Toy11–Toy15 define the official Phase-I closure set.**

No toy in this directory is required to be run to understand the theory;
their purpose is validation, not pedagogy.

---

## Phase-I Canonical Toys

The following toys directly define Phase-I closure:

- **Toy11** — Identity / superselection stability under open-system dynamics  
- **Toy12 / Toy12B** — Boundary phenomenology and model degeneracy  
- **Toy13** — Failure of joint passive evidence  
- **Toy14** — Limits of weak χ identifiability under active probing  
- **Toy15** — Terminal closure of nonlinear / threshold χ phenomenology  

These toys collectively determine which claims survived Phase-I.

---

## Outputs and Utilities

- `outputs/`  
  Centralized storage of generated datasets, figures, and summaries,
  including retrofitted outputs from Toy15.

- `scsm_utils/`  
  Shared utilities used by earlier toys. Toy15 was intentionally implemented
  independently to avoid structural bias.

---

## Reproducibility Notes

- Toys were designed for **conceptual stress-testing**, not numerical
  optimization or production benchmarking.
- Numerical instability, null mimicry, and degeneracy are considered
  meaningful outcomes.
- Results are interpreted qualitatively and comparatively, not as precision
  predictions.

---

## Status

- **Phase-I:** Complete  
- **Remaining claims:** See top-level `FALSIFICATION.md`  
- **Negative results:** Explicitly documented in `NEGATIVE_RESULTS.md`
