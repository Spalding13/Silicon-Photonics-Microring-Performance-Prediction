# Predicting Silicon Photonics Microring Performance from Inline Metrology

## Overview

This project builds a small synthetic benchmark for a realistic manufacturing question:

> Can we predict downstream optical behavior from cheaper inline metrology measurements?

The setting is silicon photonics. The target is the resonance wavelength of a microring
resonator (MRR). The data is synthetic, but the structure is meant to be physically
plausible, explainable, and easy to inspect.

The code is intentionally centered on a simple idea:

- resonance wavelength depends mostly on waveguide width and silicon thickness
- wafers have smooth spatial variation, not pure random noise
- inline measurements are noisy
- downstream test is available only for part of the wafer
- some sampled downstream tests are invalid, and some valid tests fail spec

## What The Project Does

The project generates two tables:

1. `inline_metrology.csv`
   One row per die with noisy inline measurements.
2. `downstream_wafer_test.csv`
   One row per die that was sampled for downstream test.

Then it trains a few baseline regression models to predict:

- `lambda_res_nm`

from measured inline features such as:

- `wg_width_nm_meas`
- `soi_thickness_nm_meas`
- `etch_depth_nm_meas`
- `roughness_rms_nm_meas`
- `overlay_x_nm_meas`
- `overlay_y_nm_meas`
- `defect_density_cm2_meas`

## Core Idea In Plain Language

If you want the shortest possible explanation of the generator, it is this:

1. Start from a nominal design:
   `lambda0 = 1550 nm`, `w0 = 450 nm`, `t0 = 220 nm`
2. Give each wafer a small width and thickness drift.
3. Add smooth spatial variation across the wafer.
4. Add small die-level randomness.
5. Convert the true width and thickness into a true resonance wavelength with a linear model.
6. Add measurement noise to create the observed inline and downstream tables.
7. Sample only part of the dies for downstream test.
8. Mark some sampled dies as invalid and apply a tighter pass/fail spec to the valid test results.

That is the heart of the project. Everything else is support code around this flow.

## Physical Model

The resonance model is first-order and intentionally simple:

$$
\lambda_i = \lambda_0 + \alpha (t_i - t_0) + \beta (w_i - w_0) + \eta_i^{(\lambda)}
$$

where:

- $\lambda_i$ is the resonance wavelength for device $i$
- $\lambda_0$ is the nominal resonance wavelength
- $t_i$ and $w_i$ are the true thickness and true width
- $t_0$ and $w_0$ are the nominal thickness and width
- $\alpha$ and $\beta$ are sensitivity coefficients
- $\eta_i^{(\lambda)}$ is a noise term

Default coefficients used in the project:

- $\alpha = 1.25\,\text{nm/nm}$
- $\beta = 1.08\,\text{nm/nm}$

The quality factor is modeled separately using a simple log-linear rule:

$$
\log Q_i = \log Q_0 - k_r r_i - k_d d_i + \eta_i^{(Q)}
$$

where:

- $Q_i$ is the loaded quality factor for device $i$
- $Q_0$ is the nominal quality factor
- $r_i$ is a roughness-related degradation term
- $d_i$ is a defect-density-related degradation term
- $k_r$ and $k_d$ are degradation coefficients
- $\eta_i^{(Q)}$ is a noise term

In the current project, `Q` matters in two places:

- as a measured downstream quantity
- as part of the invalid-test and pass/fail logic

## What Is Synthetic And What Is Not

This is not real fab data.

It is a synthetic generator with:

- physically motivated parameter values
- wafer-level and die-level variability
- noisy measurements
- partial downstream sampling
- valid, failed, and invalid downstream outcomes

It is useful as a controlled benchmark, not as a substitute for real process data.

## Data Tables

### Inline metrology

One row per `(wafer_id, die_id)`.

Main public columns:

| Column | Meaning |
| --- | --- |
| `wafer_id` | Wafer identifier |
| `lot_id` | Lot identifier |
| `die_id` | Die identifier |
| `x_mm`, `y_mm` | Die coordinates |
| `r_mm` | Distance from wafer center |
| `wg_width_nm_meas` | Measured waveguide width |
| `soi_thickness_nm_meas` | Measured silicon thickness |
| `etch_depth_nm_meas` | Measured etch depth proxy |
| `roughness_rms_nm_meas` | Measured roughness proxy |
| `overlay_x_nm_meas`, `overlay_y_nm_meas` | Measured overlay components |
| `defect_density_cm2_meas` | Measured defect proxy |
| `metrology_valid` | Inline measurement validity flag |

Important:

The generator internally creates latent "true" quantities such as `w_true`, `t_true`,
`lambda_true`, and `q_true`. These variables represent the hidden physical state of the
synthetic device before measurement noise, test filtering, and downstream sampling are
applied.

They are intentionally **not included in the public inline table**. The modeling workflow
should only use quantities that would realistically be observable through inline metrology.
If latent true variables were exposed as input features, the prediction task would become
unrealistically easy and the workflow would suffer from data leakage.

### Downstream wafer test

One row per die that was sampled for downstream test.

Main columns:

| Column | Meaning |
| --- | --- |
| `wafer_id`, `die_id` | Join keys |
| `test_station_id` | Test station identifier |
| `lambda_res_nm` | Measured resonance wavelength |
| `q_loaded` | Measured loaded Q |
| `insertion_loss_db` | Measured optical loss |
| `test_pass` | Pass/fail flag |
| `test_valid` | Validity flag |

Notes:

- A sampled die can still have `test_valid = 0`. In that case the test was attempted, but there is no usable downstream measurement.
- For invalid rows, `lambda_res_nm`, `q_loaded`, and `insertion_loss_db` are left empty on purpose.
- `test_pass` should be interpreted only when `test_valid = 1`.
- Dies that were never sampled for downstream test do not appear in this table at all.

This means a left join from inline to downstream naturally gives three downstream states:

- `Pass`
- `Fail`
- `Not tested / no valid downstream record`

That is exactly the status split used in the wafer map EDA notebook.

## Models

The project keeps the modeling deliberately small:

- Linear regression
- Ridge regression
- Histogram-based gradient boosting

This is enough to compare:

- a model that matches the simple physical story well
- a regularized linear variant
- a more flexible nonlinear baseline

## Evaluation

The main evaluation uses `GroupKFold` by `wafer_id`.

This matters because dies from the same wafer are correlated. A random row-wise split
would make the task look easier than it really is.

For regression, the merge helper keeps only `test_valid = 1` rows by default so the
target is always defined.

## Repository Structure

```text
src/
  physics.py      parameters and simple physics-inspired equations
  generator.py    synthetic data generation
  utils.py        CSV I/O, schema checks, and a few plotting helpers
  models.py       baseline models

tests/
  test_generator.py

notebooks/
  analysis.ipynb
  wafermap_downstream_status.ipynb
```

## How To Read The Project

If you want to understand the project quickly, this is the best order:

1. Read `src/physics.py`
   This gives you the parameter names and the two core equations.
2. Read `src/generator.py`
   Focus on `_generate_wafer()` and `_apply_downstream_sampling()`.
3. Open `notebooks/analysis.ipynb`
   This shows how the generated tables are used for regression.
4. Open `notebooks/wafermap_downstream_status.ipynb`
   This shows how to inspect pass/fail/not-tested status spatially from saved CSVs.
5. Read `tests/test_generator.py`
   This is a good way to see what behavior the code is supposed to guarantee.

## How To Run

```bash
pip install -e .
jupyter notebook notebooks/analysis.ipynb
```

The main notebook regenerates the synthetic data, runs the models, and produces the figures.
The wafer map notebook loads the saved CSV files and focuses on EDA of downstream status.

## What To Expect From The Results

Because the target is generated from a mostly linear width/thickness relationship:

- linear regression should perform very well
- ridge should usually behave almost the same
- the nonlinear model is useful as a comparison, not necessarily the winner

If you evaluate by wafer groups, performance should be lower than with a random split.
That is expected and is part of the point of the project.

The downstream table should also contain a visible mix of:

- valid pass results
- valid fail results
- sampled but invalid rows

This makes wafer maps and missingness analysis more informative.

## Current Limits

The project is intentionally modest in scope.

It does not try to model:

- image-based metrology
- deep learning pipelines
- advanced uncertainty quantification
- full process physics
- real production data artifacts

The goal is a clear, explainable synthetic benchmark that is easy to inspect.
