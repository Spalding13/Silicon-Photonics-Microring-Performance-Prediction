"""Synthetic data generator for the silicon photonics toy problem.

This module builds two public CSV-style tables used by the notebooks:

1. inline metrology: one row per die, available for all dies
2. downstream optical test: one row per tested die, available only for a subset

Internally, the generator also creates latent physical quantities such as true
width, true thickness, true roughness, true defectivity, true resonance, and true
Q. Those latent values are used only to generate realistic measured columns and
are intentionally excluded from the public inline table to avoid leakage.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .physics import (
    SyntheticMRRProcessConfig,
    combine_spatial_fields,
    compute_log_q,
    compute_resonance_wavelength,
    generate_angular_spatial_field,
    generate_radial_spatial_field,
    generate_random_spatial_field,
)


class SyntheticMRRDataGenerator:
    """Generate synthetic inline metrology and downstream test tables.

    The generator follows a simple hierarchical manufacturing story:

    lot-level variation
        -> wafer-level variation
        -> within-wafer spatial variation
        -> die-level noise
        -> measurement noise

    The output is deliberately split into two public data sources. The inline
    table contains early process/metrology measurements for every die. The
    downstream table contains later optical test results only for sampled dies.
    """

    def __init__(
        self,
        params: Optional[SyntheticMRRProcessConfig] = None,
        seed: int = 42,
    ):
        """Initialize the generator.

        Args:
            params: Synthetic manufacturing/process configuration. If omitted,
                the default SyntheticMRRProcessConfig is used.
            seed: Base random seed used to make the generator deterministic.
        """
        self.params = params or SyntheticMRRProcessConfig()
        self.seed = seed

    def generate_dataset(
        self,
        n_wafers: int = 20,
        n_dies_per_wafer: int = 400,
        p_downstream_sample: float = 0.5,
        mnar_intensity: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate the public inline and downstream tables.

        Args:
            n_wafers: Number of wafers to generate.
            n_dies_per_wafer: Number of die locations generated per wafer.
            p_downstream_sample: Planned downstream sampling probability. This
                is the MCAR-like component: before quality-dependent missingness,
                this fraction of dies is selected for downstream testing.
            mnar_intensity: Strength of quality-dependent downstream missingness.
                Larger values make low-Q devices more likely to be missing from
                the final downstream table.

        Returns:
            A tuple ``(df_inline, df_downstream)``:
            - ``df_inline`` contains one public inline metrology row per die.
            - ``df_downstream`` contains one public optical test row per tested die.
        """
        all_die_records = []

        # Use five synthetic lots. Each lot receives a shared latent process
        # fingerprint before wafer-level and die-level variation are added.
        lot_ids = [f"L{i + 1:02d}" for i in range(5)]
        lot_effects = self._generate_lot_effects(lot_ids)

        for wafer_idx in range(n_wafers):
            wafer_id = f"W{wafer_idx + 1:03d}"
            lot_id = f"L{(wafer_idx % 5) + 1:02d}"  # Five lots cycling.
            wafer_seed = self.seed + wafer_idx  # Reproducible per-wafer randomness.
            lot_effect = lot_effects[lot_id]

            wafer_records = self._generate_wafer(
                wafer_id=wafer_id,
                lot_id=lot_id,
                n_dies=n_dies_per_wafer,
                wafer_seed=wafer_seed,
                lot_effect=lot_effect,
            )
            all_die_records.extend(wafer_records)

        # df_all contains both public measurements and latent internal columns.
        # The public builders below decide which columns are exposed.
        df_all = pd.DataFrame(all_die_records)
        df_downstream = self._apply_downstream_sampling(
            df_all=df_all,
            p_sample=p_downstream_sample,
            mnar_intensity=mnar_intensity,
        )
        df_inline = self._build_public_inline_table(df_all)
        return df_inline, df_downstream

    def _build_public_inline_table(
        self,
        df_all: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return the public inline table without latent or downstream columns.

        The full internal table contains columns such as ``w_true``, ``t_true``,
        ``lambda_true`` and ``q_true``. Those values are the hidden synthetic
        physical state and must not be exposed to the ML workflow.
        """
        public_columns = [
            "wafer_id",
            "lot_id",
            "die_id",
            "x_mm",
            "y_mm",
            "r_mm",
            "wg_width_nm_meas",
            "soi_thickness_nm_meas",
            "etch_depth_nm_meas",
            "roughness_rms_nm_meas",
            "overlay_x_nm_meas",
            "overlay_y_nm_meas",
            "defect_density_cm2_meas",
            "metrology_valid",
        ]

        available_columns = [col for col in public_columns if col in df_all.columns]
        return df_all.loc[:, available_columns].copy()

    def _generate_lot_effects(self, lot_ids: list[str]) -> Dict[str, Dict[str, float]]:
        """Generate latent lot-level process offsets.

        A lot represents a group of wafers processed under a similar synthetic
        manufacturing context. Geometry offsets are additive in nanometers;
        roughness and defectivity use multiplicative log-normal factors because
        they are positive quantities.

        Args:
            lot_ids: List of lot identifiers to generate.

        Returns:
            Mapping from lot_id to latent lot-level process offsets.
        """
        lot_rng = np.random.RandomState(self.seed + 5000)
        lot_effects = {}

        for lot_id in lot_ids:
            lot_effects[lot_id] = {
                "delta_w_lot": lot_rng.normal(0, self.params.sigma_w_lot),
                "delta_t_lot": lot_rng.normal(0, self.params.sigma_t_lot),
                "roughness_lot_multiplier": np.exp(
                    lot_rng.normal(0, self.params.roughness_lot_std)
                ),
                "defect_lot_multiplier": np.exp(
                    lot_rng.normal(0, self.params.defect_density_lot_std)
                ),
            }

        return lot_effects

    def _generate_wafer(
        self,
        wafer_id: str,
        lot_id: str,
        n_dies: int,
        wafer_seed: int,
        lot_effect: Dict[str, float],
    ) -> list:
        """Generate latent and measured records for a single wafer.

        This method creates the internal die-level state. It combines:
        - shared lot-level offsets,
        - wafer-level offsets,
        - smooth spatial fields,
        - local edge/semi-ring degradation effects,
        - die-level random variation,
        - inline measurement noise.

        The returned records still include latent columns; they are filtered out
        later by ``_build_public_inline_table`` before the public inline table is
        exposed.
        """
        wafer_rng = np.random.RandomState(wafer_seed)

        # Lot-level latent process offsets shared by all wafers in this lot.
        delta_w_lot = lot_effect["delta_w_lot"]
        delta_t_lot = lot_effect["delta_t_lot"]
        roughness_lot_base = (
            self.params.roughness_mean * lot_effect["roughness_lot_multiplier"]
        )
        defect_lot_base = (
            self.params.defect_density_mean * lot_effect["defect_lot_multiplier"]
        )

        # Wafer-level offsets create differences between wafers within a lot.
        delta_w_wafer = wafer_rng.normal(0, self.params.sigma_w_wafer)
        delta_t_wafer = wafer_rng.normal(0, self.params.sigma_t_wafer)

        # Wafer-level quality baselines are sampled around the lot-level baseline.
        # This creates a hierarchy: lot baseline -> wafer baseline -> die values.
        roughness_wafer_base = np.exp(
            wafer_rng.normal(np.log(roughness_lot_base), self.params.roughness_std)
        )
        defect_wafer_base = np.exp(
            wafer_rng.normal(np.log(defect_lot_base), self.params.defect_density_std)
        )

        test_station_id = f"TS{wafer_rng.choice([1, 2, 3])}"

        die_lattice = self._generate_die_coordinates(n_dies)
        x_coords = die_lattice["x_mm"].to_numpy()
        y_coords = die_lattice["y_mm"].to_numpy()
        r_coords = die_lattice["r_mm"].to_numpy()
        die_rows = die_lattice["grid_row"].to_numpy()
        die_cols = die_lattice["grid_col"].to_numpy()

        local_effects = self._compute_local_failure_effects(x_coords, y_coords, r_coords)
        edge_q_effect = local_effects["edge_q_effect"]
        semi_ring_effect = local_effects["semi_ring_effect"]

        # Smooth spatial fields create within-wafer structure rather than purely
        # independent die-to-die random noise.
        field_radial = generate_radial_spatial_field(
            x_coords,
            y_coords,
            wafer_radius_mm=75.0,
            amplitude=self.params.spatial_field_scale,
        )
        field_angular = generate_angular_spatial_field(
            x_coords,
            y_coords,
            amplitude=self.params.spatial_field_scale,
        )
        field_random = generate_random_spatial_field(
            x_coords,
            y_coords,
            scale=self.params.spatial_rms,
            rng=wafer_rng,
        )

        field_w = combine_spatial_fields(field_radial, field_angular, field_random)
        field_t = combine_spatial_fields(
            field_radial * 0.5,
            field_angular,
            field_random * 0.8,
        )
        field_roughness = combine_spatial_fields(
            field_radial * 0.3,
            field_angular * 0.2,
            field_random,
        )

        die_records = []
        for die_idx in range(n_dies):
            die_id = f"D_R{int(die_rows[die_idx]):03d}_C{int(die_cols[die_idx]):03d}"
            x_mm = x_coords[die_idx]
            y_mm = y_coords[die_idx]
            r_mm = r_coords[die_idx]

            # Die-level process variation: local deviations not explained by lot,
            # wafer, or smooth spatial fields.
            epsilon_w = wafer_rng.normal(0, self.params.sigma_w_die)
            epsilon_t = wafer_rng.normal(0, self.params.sigma_t_die)
            epsilon_roughness = wafer_rng.normal(0, self.params.sigma_roughness_die)
            epsilon_defect_log = wafer_rng.normal(0, 0.2)

            # Hidden true geometry. These values drive physics but are not exposed
            # as public model features.
            w_true = (
                self.params.w0
                + delta_w_lot
                + delta_w_wafer
                + field_w[die_idx]
                + 1.2 * self.params.ring_failure_strength * semi_ring_effect[die_idx]
                + epsilon_w
            )
            t_true = (
                self.params.t0
                + delta_t_lot
                + delta_t_wafer
                + field_t[die_idx]
                + 0.6 * self.params.ring_failure_strength * semi_ring_effect[die_idx]
                + epsilon_t
            )

            # Hidden true quality drivers. Edge and semi-ring effects mainly act
            # through roughness/defectivity and therefore through q_loaded.
            roughness_true = np.maximum(
                roughness_wafer_base
                + field_roughness[die_idx]
                + 0.9 * self.params.edge_failure_strength * edge_q_effect[die_idx]
                + 0.2 * self.params.ring_failure_strength * semi_ring_effect[die_idx]
                + epsilon_roughness,
                0.1,
            )
            defect_true = np.maximum(
                defect_wafer_base
                * np.exp(
                    epsilon_defect_log
                    + 0.35 * self.params.edge_failure_strength * edge_q_effect[die_idx]
                    + 0.15 * self.params.ring_failure_strength * semi_ring_effect[die_idx]
                ),
                10.0,
            )

            overlay_x_true = wafer_rng.normal(0, 1.0)
            overlay_y_true = wafer_rng.normal(0, 1.0)

            # Public inline metrology: noisy observations of the hidden process state.
            w_meas = w_true + wafer_rng.normal(0, self.params.sigma_w_meas)
            t_meas = t_true + wafer_rng.normal(0, self.params.sigma_t_meas)
            roughness_meas = np.maximum(
                roughness_true + wafer_rng.normal(0, self.params.sigma_roughness_meas),
                0.1,
            )
            defect_meas = np.maximum(
                defect_true
                * np.exp(wafer_rng.normal(0, self.params.sigma_defect_meas / 1000)),
                10.0,
            )
            overlay_x_meas = overlay_x_true + wafer_rng.normal(
                0, self.params.sigma_overlay_meas
            )
            overlay_y_meas = overlay_y_true + wafer_rng.normal(
                0, self.params.sigma_overlay_meas
            )
            etch_depth_meas = 80.0 + field_t[die_idx] * 2 + wafer_rng.normal(0, 5.0)

            lambda_true = compute_resonance_wavelength(
                t_true=np.array([t_true]),
                w_true=np.array([w_true]),
                t0=self.params.t0,
                w0=self.params.w0,
                lambda0=self.params.lambda0,
                alpha=self.params.alpha,
                beta=self.params.beta,
            )[0]

            log_q_true = compute_log_q(
                roughness=np.array([roughness_true]),
                defect_density=np.array([defect_true]),
                q0=self.params.q0,
                k_r=self.params.k_r,
                k_d=self.params.k_d,
            )[0]
            q_true = np.exp(log_q_true)

            die_record = {
                "wafer_id": wafer_id,
                "lot_id": lot_id,
                "die_id": die_id,
                "x_mm": x_mm,
                "y_mm": y_mm,
                "r_mm": r_mm,
                "wg_width_nm_meas": w_meas,
                "soi_thickness_nm_meas": t_meas,
                "etch_depth_nm_meas": etch_depth_meas,
                "roughness_rms_nm_meas": roughness_meas,
                "overlay_x_nm_meas": overlay_x_meas,
                "overlay_y_nm_meas": overlay_y_meas,
                "defect_density_cm2_meas": defect_meas,
                "metrology_valid": 1,
                "w_true": w_true,
                "t_true": t_true,
                "roughness_true": roughness_true,
                "defect_true": defect_true,
                "lambda_true": lambda_true,
                "q_true": q_true,
                "edge_q_effect_true": edge_q_effect[die_idx],
                "semi_ring_effect_true": semi_ring_effect[die_idx],
                "test_station_id": test_station_id,
            }
            die_records.append(die_record)

        return die_records

    def _apply_downstream_sampling(
        self,
        df_all: pd.DataFrame,
        p_sample: float = 0.5,
        mnar_intensity: float = 1.0,
    ) -> pd.DataFrame:
        """Sample downstream tests and return public optical test records.

        The downstream table is generated in two stages:
        1. planned random sampling, controlled by ``p_sample``;
        2. quality-dependent missingness, controlled by ``mnar_intensity``.

        Dies with lower true Q are more likely to be missing from the public
        downstream table. This models the idea that degraded devices can be less
        likely to produce a usable downstream optical test record.
        """
        rng = np.random.RandomState(self.seed + 1000)

        n = len(df_all)
        sampled_indices = rng.choice(n, size=int(np.ceil(n * p_sample)), replace=False)
        df_sampled = df_all.iloc[sampled_indices].copy()

        # Downstream measurements observe the latent optical values with test noise.
        df_sampled["lambda_res_nm"] = df_sampled["lambda_true"] + rng.normal(
            0, self.params.sigma_lambda_meas, size=len(df_sampled)
        )
        df_sampled["q_loaded"] = np.exp(
            np.log(df_sampled["q_true"])
            + rng.normal(0, self.params.sigma_q_meas, size=len(df_sampled))
        )
        df_sampled["insertion_loss_db"] = (
            5.0 + (np.log(self.params.q0) - np.log(df_sampled["q_true"])) * 0.1
        )

        # MNAR-style filtering: low-Q devices are more likely to be missing from
        # the final usable downstream table.
        missing_bias = -2.2
        missing_scale = 8.0 * mnar_intensity
        log_q_true = np.log(df_sampled["q_true"].values)
        log_q_thresh = np.log(self.params.q_mnar_threshold)
        missing_logits = missing_bias + missing_scale * (log_q_thresh - log_q_true)
        missing_probs = 1.0 / (1.0 + np.exp(-missing_logits))
        keep_mask = rng.uniform(0, 1, size=len(df_sampled)) > missing_probs

        df_downstream = df_sampled.loc[keep_mask].copy()
        df_downstream["test_pass"] = (
            df_downstream["lambda_res_nm"].between(
                self.params.lambda_spec_min,
                self.params.lambda_spec_max,
            )
            & (df_downstream["q_loaded"] >= self.params.q_spec_min)
        ).astype(int)

        cols_to_keep = [
            "wafer_id",
            "die_id",
            "test_station_id",
            "lambda_res_nm",
            "q_loaded",
            "insertion_loss_db",
            "test_pass",
        ]
        return df_downstream.loc[:, cols_to_keep].reset_index(drop=True)

    def _generate_die_coordinates(
        self,
        n_dies: int,
    ) -> pd.DataFrame:
        """Generate a regular die lattice clipped to a circular wafer footprint.

        The lattice is created on a square grid and then clipped by an effective
        wafer radius. The most central sites are kept so the final footprint is
        round-ish and has exactly ``n_dies`` selected locations.
        """
        wafer_radius_mm = 75.0
        edge_exclusion_mm = 5.0
        effective_radius = wafer_radius_mm - edge_exclusion_mm

        grid_side = int(np.ceil(np.sqrt(n_dies * 4.0 / np.pi)))
        while True:
            axis = np.linspace(-effective_radius, effective_radius, grid_side)
            x_grid, y_grid = np.meshgrid(axis, axis)
            r_grid = np.sqrt(x_grid**2 + y_grid**2)
            inside_mask = r_grid <= effective_radius + 1e-9
            if int(np.count_nonzero(inside_mask)) >= n_dies:
                break
            grid_side += 1

        center_index = 0.5 * (grid_side - 1)
        candidates = pd.DataFrame(
            {
                "grid_row": np.repeat(np.arange(grid_side), grid_side),
                "grid_col": np.tile(np.arange(grid_side), grid_side),
                "x_mm": x_grid.ravel(),
                "y_mm": y_grid.ravel(),
                "r_mm": r_grid.ravel(),
            }
        )
        candidates = candidates[candidates["r_mm"] <= effective_radius + 1e-9].copy()
        candidates["row_center_dist"] = np.abs(candidates["grid_row"] - center_index)
        candidates["col_center_dist"] = np.abs(candidates["grid_col"] - center_index)

        selected = (
            candidates
            .sort_values(
                ["r_mm", "row_center_dist", "col_center_dist", "grid_row", "grid_col"]
            )
            .head(n_dies)
            .sort_values(["grid_row", "grid_col"])
            .reset_index(drop=True)
        )
        return selected[["grid_row", "grid_col", "x_mm", "y_mm", "r_mm"]]

    def _compute_local_failure_effects(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        r_coords: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build local edge and semi-ring degradation fields.

        ``edge_q_effect`` increases near the wafer perimeter and mainly affects
        roughness/defectivity. ``semi_ring_effect`` creates a localized angular
        ring-like pattern. Both are latent fields used only inside the generator.
        """
        wafer_radius_mm = 75.0
        edge_exclusion_mm = 5.0
        effective_radius = wafer_radius_mm - edge_exclusion_mm

        normalized_r = np.clip(r_coords / effective_radius, 0.0, 1.0)
        theta = np.arctan2(y_coords, x_coords)

        # Logistic edge effect: near zero in the center, increasing toward edge.
        edge_q_effect = 1.0 / (
            1.0
            + np.exp(
                -(normalized_r - self.params.edge_failure_center)
                / self.params.edge_failure_width
            )
        )

        # Semi-ring effect: radial Gaussian ring modulated by angular sector.
        ring_mask = np.exp(
            -0.5
            * ((normalized_r - self.params.ring_failure_radius) / self.params.ring_failure_width)
            ** 2
        )
        ring_angle = np.deg2rad(self.params.ring_failure_angle_deg)
        sector_mask = np.clip(np.cos(theta - ring_angle), 0.0, None) ** 2
        semi_ring_effect = ring_mask * (0.35 + 0.65 * sector_mask)

        return {
            "edge_q_effect": edge_q_effect,
            "semi_ring_effect": semi_ring_effect,
        }

    def validate_and_summarize(
        self,
        df_inline: pd.DataFrame,
        df_downstream: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return summary statistics for the generated public tables.

        The summary is intentionally lightweight and notebook-friendly. It is
        used to check dataset scale, downstream coverage, pass/fail balance,
        resonance distribution, Q distribution, and simple width-resonance signal.
        """
        stats = {
            "n_dies_inline": len(df_inline),
            "n_dies_downstream": len(df_downstream),
            "n_dies_downstream_pass": int((df_downstream["test_pass"] == 1).sum()),
            "n_dies_downstream_fail": int((df_downstream["test_pass"] == 0).sum()),
            "n_dies_not_tested": max(len(df_inline) - len(df_downstream), 0),
            "coverage_pct": 100.0 * len(df_downstream) / len(df_inline)
            if len(df_inline)
            else 0.0,
            "n_wafers": df_inline["wafer_id"].nunique(),
            "n_lots": df_inline["lot_id"].nunique(),
        }
        stats["not_tested_pct"] = (
            100.0 * stats["n_dies_not_tested"] / len(df_inline)
            if len(df_inline)
            else 0.0
        )
        stats["pass_rate_pct"] = (
            100.0 * stats["n_dies_downstream_pass"] / len(df_downstream)
            if len(df_downstream)
            else 0.0
        )

        stats["lambda_min"] = (
            df_downstream["lambda_res_nm"].min() if len(df_downstream) > 0 else np.nan
        )
        stats["lambda_max"] = (
            df_downstream["lambda_res_nm"].max() if len(df_downstream) > 0 else np.nan
        )
        stats["lambda_mean"] = (
            df_downstream["lambda_res_nm"].mean() if len(df_downstream) > 0 else np.nan
        )
        stats["lambda_std"] = (
            df_downstream["lambda_res_nm"].std() if len(df_downstream) > 0 else np.nan
        )

        stats["q_min"] = (
            df_downstream["q_loaded"].min() if len(df_downstream) > 0 else np.nan
        )
        stats["q_max"] = (
            df_downstream["q_loaded"].max() if len(df_downstream) > 0 else np.nan
        )
        stats["q_mean"] = (
            df_downstream["q_loaded"].mean() if len(df_downstream) > 0 else np.nan
        )
        stats["q_std"] = (
            df_downstream["q_loaded"].std() if len(df_downstream) > 0 else np.nan
        )

        stats["width_mean"] = df_inline["wg_width_nm_meas"].mean()
        stats["width_std"] = df_inline["wg_width_nm_meas"].std()
        stats["thickness_mean"] = df_inline["soi_thickness_nm_meas"].mean()
        stats["thickness_std"] = df_inline["soi_thickness_nm_meas"].std()

        if len(df_downstream) > 1:
            df_merged = df_downstream.merge(
                df_inline[["wafer_id", "die_id", "wg_width_nm_meas", "soi_thickness_nm_meas"]],
                on=["wafer_id", "die_id"],
                how="left",
            )
            stats["corr_width_lambda"] = (
                df_merged[["wg_width_nm_meas", "lambda_res_nm"]].corr().iloc[0, 1]
                if len(df_merged) > 1
                else np.nan
            )
        else:
            stats["corr_width_lambda"] = np.nan

        return stats
