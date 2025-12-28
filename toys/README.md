> **Reproducibility note:**  
> Output artifacts (datasets, figures, summaries) are not version-controlled.  
> All results in this directory are reproducible by rerunning the corresponding toy scripts.  
> Only decision-driving toy models are preserved; exploratory or superseded runs were not retained.

# SCSM Phase-I Toy Model Program (Exhaustive, File-Accurate)

This directory contains **all adversarial toy models** developed during  
Phase-I of the Speculative Consciousness Substrate Model (SCSM).

Every toy listed here corresponds to a concrete `.py` file in this directory.  
No toys are omitted, merged, or summarized away.

Negative results are preserved intentionally.

---

## Shared Utilities (Toybench)

The following utilities are used by **Toy01–Toy14** unless explicitly stated:

- `scsm_utils/io_utils.py` — output contract, run directories, CSV/JSON/HDF5  
- `scsm_utils/plot_utils.py` — safe plotting helpers  
- `scsm_utils/spectral_utils.py` — FFT / PSD / peak helpers  

⚠️ **Toy15+ do NOT use toybench utilities.**

---

## Toy01 — χ Bio Open System Prototype
**File:** `toy01_chi_bio_open_system.py`

**Question:**  
Can χ-like coupling be inferred from noisy open-system biological dynamics?

**Outcome:**  
Signal collapses under null noise; no identifiable χ signature.

**Status:** Retained for provenance only.

---

## Toy02 — χ Identifiability (Single Setting)
**File:** `toy02_chi_identifiability.py`

**Question:**  
Is χ identifiable at a fixed parameter point?

**Outcome:**  
No — null models match observables.

---

## Toy02B — χ Identifiability (Variant Nulls)
**File:** `toy02b_chi_identifiability.py`

**Outcome:**  
Identifiability failure persists under alternative null constructions.

---

## Toy02C — χ Identifiability (Demod Variant)
**File:** `toy02c_chi_identifiability.py`

**Outcome:**  
Demodulation does not rescue identifiability.

---

## Toy02D — χ Identifiability (Extended Diagnostics)
**File:** `toy02d_chi_identifiability.py`

**Outcome:**  
All diagnostics fail to distinguish χ from adversarial nulls.

---

## Toy03 — χ Identifiability Phase Diagram
**File:** `toy03_chi_identifiability_phase_diagram.py`

**Question:**  
Where (if anywhere) is χ detectable across coupling × noise space?

**Outcome:**  
Detectability regions shrink to zero under realistic margins.

---

## Toy04 — Boundary Force Plateau Prototype
**File:** `toy04_boundary_force_plateau.py`

**Question:**  
Is a plateau-shaped boundary force operationally measurable?

**Outcome:**  
Yes — but form is not unique.

---

## Toy05 — Boundary Null Mimicry I
**File:** `toy05_boundary_nulls.py`

**Outcome:**  
Null forces reproduce plateau-like behavior.

---

## Toy06 — Boundary Scaling Stress Test
**File:** `toy06_boundary_scaling.py`

**Outcome:**  
Scaling arguments do not isolate boundary physics.

---

## Toy07 — Boundary Drift & Noise
**File:** `toy07_boundary_drift.py`

**Outcome:**  
Drift dominates over functional distinctions.

---

## Toy08 — Boundary Degeneracy Sweep
**File:** `toy08_boundary_degeneracy.py`

**Outcome:**  
Multiple models coexist within error tolerance.

---

## Toy09 — Identity Leakage (Precursor)
**File:** `toy09_identity_leakage.py`

**Question:**  
Does sector identity leak under weak mixing?

**Outcome:**  
Leakage can be bounded → motivates Toy11.

---

## Toy10 — Identity + Boundary Integrator
**File:** `toy10_integrator_identity_plus_boundary_allowed_region.py`

**Question:**  
Can identity stability and boundary measurability coexist?

**Outcome:**  
Yes, but only in a restricted region.

---

## Toy11 — Identity Stability Under Stress
**File:** `toy11_robustness_identity_plus_boundary_stress.py`

**Outcome:**  
Identity persistence is mathematically viable.

**Status:** **Survives Phase-I**

---

## Toy12 — Boundary Battlefield (Dataset Generator)
**File:** `toy12_boundary_battlefield_plateau_vs_nulls.py`

**Purpose:**  
Generate realistic, adversarial boundary datasets.

---

## Toy12B — Boundary Model Selection
**File:** `toy12b_boundary_model_selection.py`

**Outcome:**  
Static boundary curves are not diagnostic.

---

## Toy13 — Joint Passive Evidence Failure
**File:** `toy13_joint_boundary_and_khz_fingerprint.py`

**Outcome:**  
Joint passive evidence performs worse than boundary-only.

**Claim eliminated.**

---

## Toy14 — Active χ Identifiability
**File:** `toy14_active_lockin_causality_phase_harmonic.py`

**Outcome:**  
Causal lag exists in principle, but not experimentally usable.

**Weak χ eliminated.**

---

## Toy15 — Nonlinear / Threshold χ Closure
**File:** `toy15_final_FIXED.py`

⚠️ **Standalone (no toybench utilities)**

**Outcome:**  
- 0% reliable recovery  
- Null false positives dominate  
- Numerical instability overwhelms signal  

**All χ phenomenology eliminated.**

---

## Toy16 — Phase-Ia Real Data Validator
**File:** `toy16_aalto_casimir_scsm_validator.py`

**Question:**  
Does SCSM v11.5.3 scoping survive first confrontation with real experimental data?

**Dataset:**  
Aalto 2025 superconducting Casimir drums  
(`Casimir_pressure_dist_v3.h5`)

**Extraction:**  
101 points: `['Gap (nm)', 'P Drude (Pa)', 'P BCS (Pa)']`

**SCSM Prediction:**  
Superconducting Casimir exhibits **no detectable anomalous force**  
(TQEC foam modes gapped).

**Outcome:**  
- Measured force at 1 mm: **F = 0.0 pN** (biological target: 52 pN)  
- Best SCSM fit: **F₀ = 0.4 pN**, **d₀ = 0.0 mm**, **n = 3.2**  
- **Perfect null confirmation** within experimental resolution

![Toy16: SCSM fit on Aalto Casimir data (null force at 1 mm as predicted)](toy16_aalto_casimir_scsm_validator.png)

**Status:** **PASS** — First real-data validation survived

---

## Phase-I Closure

After Toy15, no experimentally accessible χ pathway remains.

Remaining SCSM content is:
- topological / identity persistence  
- boundary observables as measurement channels only  

---

## Status

- **Phase-I:** COMPLETE  
- **Phase-Ia:** Real-data validation passed (Toy16)  
- **Surviving pillars:** Identity stability, boundary measurability  
- **Eliminated:** χ phenomenology (weak, strong, nonlinear)

See `NEGATIVE_RESULTS.md` and `FALSIFICATION.md` for claim disposition.
