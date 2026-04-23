"""Synthetic data generator for the silicon photonics toy problem."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

from .physics import (
    MRRParameters,
    compute_resonance_wavelength,
    compute_log_q,
    generate_radial_spatial_field,
    generate_angular_spatial_field,
    generate_random_spatial_field,
    combine_spatial_fields,
)


class SyntheticMRRDataGenerator:
    """Generate inline metrology and downstream test tables."""
    
    def __init__(
        self,
        params: Optional[MRRParameters] = None,
        seed: int = 42,
    ):
        self.params = params or MRRParameters()
        self.seed = seed
    
    def generate_dataset(
        self,
        n_wafers: int = 20,
        n_dies_per_wafer: int = 400,
        p_downstream_sample: float = 0.5,
        mnar_intensity: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate the public inline table and the downstream test table."""
        all_die_records = []
        
        for wafer_idx in range(n_wafers):
            wafer_id = f"W{wafer_idx + 1:03d}"
            lot_id = f"L{(wafer_idx % 5) + 1:02d}"  # 5 lots cycling
            wafer_seed = self.seed + wafer_idx  # Reproducible wafer-level randomness
            
            wafer_records = self._generate_wafer(
                wafer_id=wafer_id,
                lot_id=lot_id,
                n_dies=n_dies_per_wafer,
                wafer_seed=wafer_seed,
            )
            all_die_records.extend(wafer_records)
        
        df_all = pd.DataFrame(all_die_records)
        df_downstream = self._apply_downstream_sampling(
            df_all=df_all,
            p_sample=p_downstream_sample,
            mnar_intensity=mnar_intensity,
        )
        df_inline = self._build_public_inline_table(df_all)
        return df_inline, df_downstream

    def _build_public_inline_table(self, df_all: pd.DataFrame) -> pd.DataFrame:
        """Return the public inline table without latent columns."""
        public_columns = [
            'wafer_id',
            'lot_id',
            'die_id',
            'x_mm',
            'y_mm',
            'r_mm',
            'wg_width_nm_meas',
            'soi_thickness_nm_meas',
            'etch_depth_nm_meas',
            'roughness_rms_nm_meas',
            'overlay_x_nm_meas',
            'overlay_y_nm_meas',
            'defect_density_cm2_meas',
            'metrology_valid',
        ]
        available_columns = [col for col in public_columns if col in df_all.columns]
        return df_all.loc[:, available_columns].copy()
    
    def _generate_wafer(
        self,
        wafer_id: str,
        lot_id: str,
        n_dies: int,
        wafer_seed: int,
    ) -> list:
        """Generate all dies for a single wafer."""
        wafer_rng = np.random.RandomState(wafer_seed)
        
        delta_w_wafer = wafer_rng.normal(0, self.params.sigma_w_wafer)
        delta_t_wafer = wafer_rng.normal(0, self.params.sigma_t_wafer)
        roughness_wafer_base = np.exp(
            wafer_rng.normal(np.log(self.params.roughness_mean), self.params.roughness_std)
        )
        defect_wafer_base = np.exp(
            wafer_rng.normal(np.log(self.params.defect_density_mean), self.params.defect_density_std)
        )
        
        test_station_id = f"TS{wafer_rng.choice([1, 2, 3])}"
        
        die_lattice = self._generate_die_coordinates(n_dies)
        x_coords = die_lattice['x_mm'].to_numpy()
        y_coords = die_lattice['y_mm'].to_numpy()
        r_coords = die_lattice['r_mm'].to_numpy()
        die_rows = die_lattice['grid_row'].to_numpy()
        die_cols = die_lattice['grid_col'].to_numpy()
        
        field_radial = generate_radial_spatial_field(
            x_coords, y_coords, 
            wafer_radius_mm=75.0,
            amplitude=self.params.spatial_field_scale
        )
        field_angular = generate_angular_spatial_field(
            x_coords, y_coords,
            amplitude=self.params.spatial_field_scale
        )
        field_random = generate_random_spatial_field(
            x_coords, y_coords,
            scale=self.params.spatial_rms,
            rng=wafer_rng,
        )
        
        field_w = combine_spatial_fields(field_radial, field_angular, field_random)
        field_t = combine_spatial_fields(
            field_radial * 0.5, field_angular, field_random * 0.8
        )  # Slightly different weighting for thickness
        field_roughness = combine_spatial_fields(
            field_radial * 0.3, field_angular * 0.2, field_random
        )
        
        die_records = []
        for die_idx in range(n_dies):
            die_id = f"D_R{int(die_rows[die_idx]):03d}_C{int(die_cols[die_idx]):03d}"
            x_mm = x_coords[die_idx]
            y_mm = y_coords[die_idx]
            r_mm = r_coords[die_idx]
            
            epsilon_w = wafer_rng.normal(0, self.params.sigma_w_die)
            epsilon_t = wafer_rng.normal(0, self.params.sigma_t_die)
            epsilon_roughness = wafer_rng.normal(0, self.params.sigma_roughness_die)
            epsilon_defect_log = wafer_rng.normal(0, 0.2)
            
            w_true = (
                self.params.w0 + delta_w_wafer + field_w[die_idx] + epsilon_w
            )
            t_true = (
                self.params.t0 + delta_t_wafer + field_t[die_idx] + epsilon_t
            )
            roughness_true = np.maximum(
                roughness_wafer_base + field_roughness[die_idx] + epsilon_roughness,
                0.1
            )
            defect_true = np.maximum(
                defect_wafer_base * np.exp(epsilon_defect_log),
                10.0
            )
            overlay_x_true = wafer_rng.normal(0, 1.0)
            overlay_y_true = wafer_rng.normal(0, 1.0)
            
            w_meas = w_true + wafer_rng.normal(0, self.params.sigma_w_meas)
            t_meas = t_true + wafer_rng.normal(0, self.params.sigma_t_meas)
            roughness_meas = np.maximum(
                roughness_true + wafer_rng.normal(0, self.params.sigma_roughness_meas),
                0.1
            )
            defect_meas = np.maximum(
                defect_true * np.exp(wafer_rng.normal(0, self.params.sigma_defect_meas / 1000)),
                10.0
            )
            overlay_x_meas = overlay_x_true + wafer_rng.normal(0, self.params.sigma_overlay_meas)
            overlay_y_meas = overlay_y_true + wafer_rng.normal(0, self.params.sigma_overlay_meas)
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
                'wafer_id': wafer_id,
                'lot_id': lot_id,
                'die_id': die_id,
                'x_mm': x_mm,
                'y_mm': y_mm,
                'r_mm': r_mm,
                'wg_width_nm_meas': w_meas,
                'soi_thickness_nm_meas': t_meas,
                'etch_depth_nm_meas': etch_depth_meas,
                'roughness_rms_nm_meas': roughness_meas,
                'overlay_x_nm_meas': overlay_x_meas,
                'overlay_y_nm_meas': overlay_y_meas,
                'defect_density_cm2_meas': defect_meas,
                'metrology_valid': 1,
                'w_true': w_true,
                't_true': t_true,
                'roughness_true': roughness_true,
                'defect_true': defect_true,
                'lambda_true': lambda_true,
                'q_true': q_true,
                'test_station_id': test_station_id,
            }
            die_records.append(die_record)
        
        return die_records
    
    def _apply_downstream_sampling(
        self,
        df_all: pd.DataFrame,
        p_sample: float = 0.5,
        mnar_intensity: float = 1.0,
    ) -> pd.DataFrame:
        """Sample downstream tests and return only usable public test records."""
        rng = np.random.RandomState(self.seed + 1000)
        
        n = len(df_all)
        
        sampled_indices = rng.choice(n, size=int(np.ceil(n * p_sample)), replace=False)
        df_sampled = df_all.iloc[sampled_indices].copy()

        df_sampled['lambda_res_nm'] = (
            df_sampled['lambda_true']
            + rng.normal(0, self.params.sigma_lambda_meas, size=len(df_sampled))
        )
        df_sampled['q_loaded'] = np.exp(
            np.log(df_sampled['q_true'])
            + rng.normal(0, self.params.sigma_q_meas, size=len(df_sampled))
        )
        df_sampled['insertion_loss_db'] = (
            5.0 + (np.log(self.params.q0) - np.log(df_sampled['q_true'])) * 0.1
        )

        missing_bias = -2.2
        missing_scale = 8.0 * mnar_intensity
        log_q_true = np.log(df_sampled['q_true'].values)
        log_q_thresh = np.log(self.params.q_mnar_threshold)
        missing_logits = missing_bias + missing_scale * (log_q_thresh - log_q_true)
        missing_probs = 1.0 / (1.0 + np.exp(-missing_logits))
        keep_mask = rng.uniform(0, 1, size=len(df_sampled)) > missing_probs

        df_downstream = df_sampled.loc[keep_mask].copy()
        df_downstream['test_valid'] = 1
        df_downstream['test_pass'] = (
            df_downstream['lambda_res_nm'].between(
                self.params.lambda_spec_min,
                self.params.lambda_spec_max,
            )
            & (df_downstream['q_loaded'] >= self.params.q_spec_min)
        ).astype(int)

        cols_to_keep = [
            'wafer_id',
            'die_id',
            'test_station_id',
            'lambda_res_nm',
            'q_loaded',
            'insertion_loss_db',
            'test_pass',
            'test_valid',
        ]
        return df_downstream.loc[:, cols_to_keep].reset_index(drop=True)
    
    def _generate_die_coordinates(
        self,
        n_dies: int,
    ) -> pd.DataFrame:
        """Generate a regular die lattice clipped to a circular wafer."""
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
        candidates = pd.DataFrame({
            'grid_row': np.repeat(np.arange(grid_side), grid_side),
            'grid_col': np.tile(np.arange(grid_side), grid_side),
            'x_mm': x_grid.ravel(),
            'y_mm': y_grid.ravel(),
            'r_mm': r_grid.ravel(),
        })
        candidates = candidates[candidates['r_mm'] <= effective_radius + 1e-9].copy()
        candidates['row_center_dist'] = np.abs(candidates['grid_row'] - center_index)
        candidates['col_center_dist'] = np.abs(candidates['grid_col'] - center_index)

        # Keep the most central lattice sites so the footprint stays round-ish.
        selected = (
            candidates
            .sort_values(
                ['r_mm', 'row_center_dist', 'col_center_dist', 'grid_row', 'grid_col']
            )
            .head(n_dies)
            .sort_values(['grid_row', 'grid_col'])
            .reset_index(drop=True)
        )
        return selected[['grid_row', 'grid_col', 'x_mm', 'y_mm', 'r_mm']]
    
    def validate_and_summarize(
        self,
        df_inline: pd.DataFrame,
        df_downstream: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Return a small set of summary statistics for the two tables."""
        valid_downstream = df_downstream.copy()

        stats = {
            'n_dies_inline': len(df_inline),
            'n_dies_downstream': len(df_downstream),
            'n_dies_downstream_valid': len(valid_downstream),
            'n_dies_downstream_invalid': 0,
            'n_dies_downstream_pass': int((valid_downstream['test_pass'] == 1).sum()),
            'n_dies_downstream_fail': int((valid_downstream['test_pass'] == 0).sum()),
            'n_dies_not_tested': max(len(df_inline) - len(df_downstream), 0),
            'coverage_pct': 100.0 * len(df_downstream) / len(df_inline) if len(df_inline) else 0.0,
            'valid_coverage_pct': 100.0 * len(valid_downstream) / len(df_inline) if len(df_inline) else 0.0,
            'n_wafers': df_inline['wafer_id'].nunique(),
            'n_lots': df_inline['lot_id'].nunique(),
        }
        stats['not_tested_pct'] = (
            100.0 * stats['n_dies_not_tested'] / len(df_inline)
            if len(df_inline)
            else 0.0
        )
        stats['pass_rate_valid_pct'] = (
            100.0 * stats['n_dies_downstream_pass'] / len(valid_downstream)
            if len(valid_downstream)
            else 0.0
        )
        
        stats['lambda_min'] = valid_downstream['lambda_res_nm'].min() if len(valid_downstream) > 0 else np.nan
        stats['lambda_max'] = valid_downstream['lambda_res_nm'].max() if len(valid_downstream) > 0 else np.nan
        stats['lambda_mean'] = valid_downstream['lambda_res_nm'].mean() if len(valid_downstream) > 0 else np.nan
        stats['lambda_std'] = valid_downstream['lambda_res_nm'].std() if len(valid_downstream) > 0 else np.nan
        
        stats['q_min'] = valid_downstream['q_loaded'].min() if len(valid_downstream) > 0 else np.nan
        stats['q_max'] = valid_downstream['q_loaded'].max() if len(valid_downstream) > 0 else np.nan
        stats['q_mean'] = valid_downstream['q_loaded'].mean() if len(valid_downstream) > 0 else np.nan
        stats['q_std'] = valid_downstream['q_loaded'].std() if len(valid_downstream) > 0 else np.nan
        
        stats['width_mean'] = df_inline['wg_width_nm_meas'].mean()
        stats['width_std'] = df_inline['wg_width_nm_meas'].std()
        stats['thickness_mean'] = df_inline['soi_thickness_nm_meas'].mean()
        stats['thickness_std'] = df_inline['soi_thickness_nm_meas'].std()
        
        if len(valid_downstream) > 1:
            df_merged = valid_downstream.merge(
                df_inline[['wafer_id', 'die_id', 'wg_width_nm_meas', 'soi_thickness_nm_meas']],
                on=['wafer_id', 'die_id'],
                how='left'
            )
            stats['corr_width_lambda'] = (
                df_merged[['wg_width_nm_meas', 'lambda_res_nm']].corr().iloc[0, 1]
                if len(df_merged) > 1 else np.nan
            )
        else:
            stats['corr_width_lambda'] = np.nan
        
        return stats
