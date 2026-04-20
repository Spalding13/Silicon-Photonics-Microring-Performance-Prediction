# Predicting Silicon Photonics Microring Performance from Inline Metrology

## Executive Summary

This project develops a **synthetic but physically plausible surrogate model** for predicting downstream wafer-level optical performance in silicon photonics manufacturing from inline metrology measurements. Specifically, we predict **microring resonator (MRR) resonance wavelength** from noisy, partially-sampled inline measurements under realistic manufacturing process variations and sampling constraints.

The project is framed around a real industrial challenge: **metrology-to-test correlation** in semiconductor manufacturing, where exhaustive downstream testing is expensive ("cycle-time constraints"), necessitating predictive models trained on abundant but noisy inline measurements.

## Problem Formulation

### The Manufacturing Context

Silicon photonics devices (waveguides, microring resonators, modulators) are **extremely sensitive to nanometer-scale fabrication variations**. Key sources of variation include:

- **Waveguide width** (CDU – critical dimension uniformity variations)
- **Silicon device-layer thickness** (SOI uniformity)
- **Etch depth and sidewall profile** (plasma process variation)
- **Sidewall roughness** (resist stochasticity, etch-induced roughness)
- **Overlay errors** (layer-to-layer misalignment)
- **Contamination / defect density** (particles, process-induced defects)

These variations are measured **inline** (on the wafer, during or after fabrication, using metrology tools like CD-SEM, OCD/scatterometry, ellipsometry, overlay sensors). However:

1. **Inline measurements are noisy** and model-dependent (e.g., OCD reconstruction error, SEM measurement limits).
2. **Not all dies are tested downstream** due to cost and cycle-time constraints.
3. **Downstream tests fail or are invalid more often for degraded devices** (MNAR – missing-not-at-random).

### The Modeling Challenge

Given:
- Table A: **Inline metrology** (features from CD-SEM, OCD, overlay, defect inspection) — abundant, noisy
- Table B: **Downstream wafer test** (optical resonance wavelength, Q factor) — sparse, sometimes missing

**Objective**: Build and evaluate surrogate models that predict downstream performance from inline inputs, understanding how:
- Measurement noise affects prediction quality
- Sampling strategies and missingness bias predictions
- Spatial wafer variations (systematic per-wafer / per-position) impact generalization

### Significance

- **Industrial relevance**: Virtual metrology and predictive modeling are active research areas in semiconductor manufacturing (FOUP-less, real-time optimization).
- **Academic framing**: Combines process variation modeling, uncertainty quantification, missing-data handling, and group-aware cross-validation.
- **Defensible synthetic approach**: Rather than reverse-engineering proprietary fab data, we use published physics (resonance sensitivity coefficients from literature) to generate synthetic benchmark.

## Physical Story: Silicon Photonics Microring Resonators

### Device and Physics

A **microring resonator** is a waveguide bent into a circular or racetrack geometry, supporting resonant optical modes at specific wavelengths. The device is extremely sensitive to:

- **Waveguide width**: controls effective refractive index → resonance wavelength shift
- **Silicon thickness**: controls mode confinement and effective index
- **Roughness and defects**: scatter light, increasing loss and degrading quality factor Q

**Published sensitivity coefficients** (from silicon photonics characterization studies):
- $\frac{d\lambda}{dt} \approx 1.25 \text{ nm/nm}$ (resonance shift per 1 nm thickness change)
- $\frac{d\lambda}{dw} \approx 1.08 \text{ nm/nm}$ (resonance shift per 1 nm width change)
- Q is limited by scattering loss: $\log Q \propto -k_r \cdot \text{roughness} - k_d \cdot \text{defect\_density}$

### Synthetic Data Generator

We implement a generative model that produces:

1. **Latent (true but unmeasured) geometry fields**:
   - Wafer-level drifts: $\Delta w_{\text{wafer}}$, $\Delta t_{\text{wafer}}$ (systematic per-wafer variation)
   - Spatial fields: radial + angular variation + low-frequency random perturbations
   - Die-level residuals: random die-to-die noise

2. **Noisy inline measurements** (with calibrated noise levels):
   - $w_{\text{meas}} = w_{\text{true}} + \eta^{(w)}$ (measurement noise)
   - Similarly for thickness, roughness, overlay, defects

3. **Downstream optical test targets** (computed from true geometry, then noised):
   - $\lambda_{\text{target}} = \lambda_0 + \alpha (t_{\text{true}} - t_0) + \beta (w_{\text{true}} - w_0) + \eta^{(\lambda)}$
   - Log Q modeled as roughness/defect-driven

4. **Realistic missingness**:
   - **MCAR**: only $p_{\text{sample}} \times 100\%$ of dies have downstream tests (planned sampling)
   - **MNAR**: downstream test validity depends on Q (devices with very low Q more likely to fail or be rejected)

## Project Scope

### Included

- Synthetic data generation with physics-based generator
- Two independent data sources (inline metrology, downstream test)
- Baseline regression models (linear, ridge/lasso, tree ensembles, Gaussian Process, MLP)
- Group-aware cross-validation (by wafer, to respect spatial structure)
- Experiments: noise robustness, missingness bias, domain shift, feature importance
- Uncertainty quantification (predictive intervals, calibration analysis)

### Deliberately Excluded (Out of Scope)

- Image-based metrology (SEM image analysis) — keep to tabular features only
- Real fab data — synthetic approach chosen for defensibility and time constraints
- Deep learning — stick to interpretable tabular methods
- Multi-layer photonic devices — focus on single platform (e.g., 220 nm SOI)
- Reliability or thermal effects — linear first-order resonance model only

## Data Schemas

### Inline Metrology Source (`inline_metrology.csv`)

Grain: one row per (wafer_id, die_id)

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| wafer_id | str | `W001`–`W030` | Wafer identifier |
| lot_id | str | `L01`–`L05` | Lot identifier (enables lot-level drift modeling) |
| die_id | str | `D_R000_C000` | Die identifier on wafer |
| x_mm, y_mm | float | ±75 | Spatial coordinates on wafer |
| wg_width_nm_meas | float | 350–550 | Measured waveguide width |
| soi_thickness_nm_meas | float | 210–240 | Measured SOI device-layer thickness |
| etch_depth_nm_meas | float | 50–120 | Measured etch depth (proxy) |
| roughness_rms_nm_meas | float | 0.2–3.0 | Measured RMS roughness proxy |
| overlay_x_nm_meas, overlay_y_nm_meas | float | −10 to 10 | Measured overlay vector components |
| defect_density_cm2_meas | float | 0–20000 | Measured defect density proxy |
| metrology_valid | int | 0 or 1 | Whether die was fully measured |

### Downstream Wafer Test Source (`downstream_wafer_test.csv`)

Grain: one row per die **actually tested** (sparse coverage: 20–60% of dies)

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| wafer_id | str | `W001`–`W030` | Join key |
| die_id | str | `D_R000_C000` | Join key |
| test_station_id | str | `TS1`–`TS3` | Which test station (introduces measurement-system variation) |
| lambda_res_nm | float | 1500–1600 | Measured resonance wavelength (primary target) |
| q_loaded | float | 1e4–1e6 | Measured loaded quality factor |
| insertion_loss_db | float | 0–20 | Measured optical loss |
| test_pass | int | 0 or 1 | Binary pass/fail |
| test_valid | int | 0 or 1 | Whether measurement was valid |

## Deliverables

1. **Main Jupyter Notebook** (`notebooks/analysis.ipynb`):
   - Problem formulation + physical story
   - Data generation and validation
   - EDA and sanity checks
   - Model training and evaluation
   - Experiments (noise, missingness, domain shift, UQ)
   - Discussion of limitations

2. **Python Library** (`src/`):
   - `physics.py`: MRR parameters, resonance/Q models
   - `generator.py`: synthetic data generation (deterministic with seed)
   - `utils.py`: I/O, validation, plotting
   - `models.py`: baseline model wrappers

3. **Tests** (`tests/`):
   - Generator determinism, schema validation, physics consistency

4. **GitHub Repository**:
   - ≥10 meaningful commits
   - Complete README with methods and citations
   - Reproducible from scratch

## How to Run

```bash
# Clone repo
git clone <your-repo-url>
cd semiconductor-surrogates

# Install dependencies
pip install -e .

# Run notebook
jupyter notebook notebooks/analysis.ipynb
```

The notebook is self-contained: executing all cells will regenerate the synthetic dataset (with fixed seed for reproducibility), train models, run experiments, and produce all figures and tables.

## Project Structure

```
semiconductor-surrogates/
├── README.md                          (this file)
├── pyproject.toml                     (dependencies)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── physics.py                     (MRRParameters, resonance/Q models)
│   ├── generator.py                   (SyntheticMRRDataGenerator)
│   ├── utils.py                       (I/O, validation, plotting)
│   └── models.py                      (baseline model wrappers)
├── tests/
│   ├── __init__.py
│   └── test_generator.py              (unit tests)
├── notebooks/
│   └── analysis.ipynb                 (main deliverable)
└── data/
    └── .gitkeep                       (for generated CSV files)
```

## References

### Silicon Photonics & Microring Sensitivity

- Barwicz, T., et al. (2005). "Silicon photonics." *Nature Photonics*.
- Lumerical silicon photonics design resources; resonance wavelength sensitivity studies.
- Process variation characterization: ring resonators on 220 nm SOI platforms.

### Metrology & Manufacturing Analytics

- Semiconductor Industry Association (SIA). IRDS metrology roadmaps (CD-SEM, OCD, overlay).
- SEMI Standards: P18 (overlay definition), E173 (defect classification).
- Virtual metrology: adaptive sampling strategies in semiconductor manufacturing.

### Statistical Methods & Missing Data

- Missing-data mechanisms (MCAR, MAR, MNAR): Rubin framework.
- Group-aware cross-validation: respecting spatial correlations in wafer data.
- Uncertainty quantification: Gaussian Processes, conformal prediction.
