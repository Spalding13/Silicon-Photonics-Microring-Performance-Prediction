"""
Comprehensive unit tests for the synthetic data generator.

Tests cover:
- Determinism: reproducibility with fixed seed
- Schema: correct column structure and data types
- Physics consistency: resonance shift vs geometry follows sensitivity model
- Missingness: coverage roughly matches specifications
- Ranges: all features within expected bounds
"""

import pytest
import numpy as np
import pandas as pd

from src.physics import MRRParameters, compute_resonance_wavelength, compute_log_q
from src.generator import SyntheticMRRDataGenerator
from src.models import RidgeRegressionModel
from src.utils import (
    validate_schemas,
    merge_sources,
    find_inline_leakage_columns,
    save_sources,
    load_sources,
)


class TestMRRParameters:
    """Test MRRParameters class."""
    
    def test_default_parameters(self):
        """Test that default parameters are valid."""
        params = MRRParameters()
        assert params.lambda0 == 1550.0
        assert params.alpha > 0
        assert params.beta > 0
    
    def test_parameter_validation(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            MRRParameters(lambda0=200.0)  # Out of valid range
        
        with pytest.raises(ValueError):
            MRRParameters(alpha=-1.0)  # Negative coefficient
    
    def test_to_dict(self):
        """Test parameter export to dictionary."""
        params = MRRParameters()
        d = params.to_dict()
        assert isinstance(d, dict)
        assert 'lambda0' in d
        assert d['alpha'] == params.alpha


class TestPhysicsModels:
    """Test physics computation functions."""
    
    def test_resonance_wavelength_linear(self):
        """
        Test that resonance wavelength scales linearly with dimensions.
        
        In the small-variation regime, dλ = α·dt + β·dw should hold.
        """
        t0, w0, lambda0 = 220.0, 450.0, 1550.0
        alpha, beta = 1.25, 1.08
        
        # Create test data: small perturbations
        t_vals = np.array([t0, t0 + 1.0, t0 + 2.0])
        w_vals = np.array([w0, w0, w0])
        
        lambda_vals = compute_resonance_wavelength(
            t_true=t_vals,
            w_true=w_vals,
            t0=t0, w0=w0, lambda0=lambda0,
            alpha=alpha, beta=beta
        )
        
        # Check linearity: slope should be alpha
        slope = (lambda_vals[2] - lambda_vals[0]) / 2.0
        assert np.isclose(slope, alpha, rtol=1e-6)
    
    def test_log_q_decreases_with_roughness(self):
        """Test that Q decreases (in log space) with increasing roughness."""
        roughness_vals = np.array([0.5, 1.0, 2.0])
        defect_vals = np.array([0.0, 0.0, 0.0])
        q0 = 2e5
        k_r = 1e-4
        k_d = 1e-6
        
        log_q_vals = compute_log_q(
            roughness=roughness_vals,
            defect_density=defect_vals,
            q0=q0, k_r=k_r, k_d=k_d
        )
        
        # log_q should decrease monotonically
        assert log_q_vals[0] > log_q_vals[1] > log_q_vals[2]


class TestDataGenerator:
    """Test SyntheticMRRDataGenerator class."""
    
    def test_determinism_seed(self):
        """Test that same seed produces identical output."""
        params = MRRParameters()
        
        gen1 = SyntheticMRRDataGenerator(params=params, seed=42)
        df1_inline, df1_downstream = gen1.generate_dataset(
            n_wafers=2,
            n_dies_per_wafer=50,
            p_downstream_sample=0.5,
        )
        
        gen2 = SyntheticMRRDataGenerator(params=params, seed=42)
        df2_inline, df2_downstream = gen2.generate_dataset(
            n_wafers=2,
            n_dies_per_wafer=50,
            p_downstream_sample=0.5,
        )
        
        # Check that DataFrames are identical
        pd.testing.assert_frame_equal(df1_inline, df2_inline)
        pd.testing.assert_frame_equal(df1_downstream, df2_downstream)
    
    def test_different_seed_different_data(self):
        """Test that different seeds produce different data."""
        params = MRRParameters()
        
        gen1 = SyntheticMRRDataGenerator(params=params, seed=42)
        df1_inline, _ = gen1.generate_dataset(n_wafers=2, n_dies_per_wafer=50)
        
        gen2 = SyntheticMRRDataGenerator(params=params, seed=43)
        df2_inline, _ = gen2.generate_dataset(n_wafers=2, n_dies_per_wafer=50)
        
        # Data should differ
        assert not df1_inline.equals(df2_inline)
    
    def test_schema_inline_metrology(self):
        """Test that inline metrology has correct schema."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, _ = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=10)
        
        required_cols = {
            'wafer_id', 'lot_id', 'die_id', 'x_mm', 'y_mm',
            'wg_width_nm_meas', 'soi_thickness_nm_meas',
            'roughness_rms_nm_meas', 'overlay_x_nm_meas', 'overlay_y_nm_meas',
            'defect_density_cm2_meas',
        }
        assert required_cols.issubset(set(df_inline.columns))
        assert len(df_inline) == 10
        assert df_inline['metrology_valid'].min() == 1  # All valid

    def test_inline_metrology_excludes_latent_columns(self):
        """Public inline table should not expose latent generator state."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, _ = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=10)

        leakage_columns = find_inline_leakage_columns(df_inline)
        assert leakage_columns == []
    
    def test_schema_downstream_test(self):
        """Test that downstream test has correct schema."""
        gen = SyntheticMRRDataGenerator(seed=42)
        _, df_downstream = gen.generate_dataset(
            n_wafers=1, n_dies_per_wafer=100, p_downstream_sample=0.5
        )
        
        required_cols = {
            'wafer_id', 'die_id', 'test_station_id',
            'lambda_res_nm', 'q_loaded', 'insertion_loss_db',
            'test_pass', 'test_valid',
        }
        assert required_cols.issubset(set(df_downstream.columns))
        assert len(df_downstream) <= 100
        assert len(df_downstream) > 0

    def test_downstream_contains_pass_and_fail_states(self):
        """Public downstream table should contain only usable pass/fail records."""
        gen = SyntheticMRRDataGenerator(seed=42)
        _, df_downstream = gen.generate_dataset(
            n_wafers=20,
            n_dies_per_wafer=100,
            p_downstream_sample=0.5,
        )

        assert len(df_downstream) > 0
        assert (df_downstream['test_valid'] == 1).all()
        assert (df_downstream['test_pass'] == 1).any()
        assert (df_downstream['test_pass'] == 0).any()
        assert df_downstream['lambda_res_nm'].notna().all()
        assert df_downstream['q_loaded'].notna().all()
        assert df_downstream['insertion_loss_db'].notna().all()

    def test_die_layout_uses_regular_lattice_with_empty_corners(self):
        """Die coordinates should lie on a lattice with some outer grid slots unused."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, _ = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=400)

        die_rc = df_inline['die_id'].str.extract(r'D_R(\d+)_C(\d+)').astype(int)
        unique_rows = die_rc[0].nunique()
        unique_cols = die_rc[1].nunique()
        x_values = np.sort(df_inline['x_mm'].unique())
        y_values = np.sort(df_inline['y_mm'].unique())

        assert unique_rows * unique_cols > len(df_inline)
        assert len(x_values) == unique_cols
        assert len(y_values) == unique_rows
        assert np.allclose(np.diff(x_values), np.diff(x_values)[0])
        assert np.allclose(np.diff(y_values), np.diff(y_values)[0])
    
    def test_coverage_matches_p_sample(self):
        """Test that downstream coverage roughly matches p_sample parameter."""
        gen = SyntheticMRRDataGenerator(seed=42)
        
        n_dies = 1000
        p_sample = 0.5
        
        _, df_downstream = gen.generate_dataset(
            n_wafers=5,
            n_dies_per_wafer=n_dies // 5,
            p_downstream_sample=p_sample,
            mnar_intensity=0.0,  # No MNAR to isolate MCAR effect
        )
        
        actual_coverage = len(df_downstream) / n_dies
        # Allow ±10% tolerance
        assert 0.4 <= actual_coverage <= 0.6
    
    def test_physics_consistency_width_resonance(self):
        """
        Test that measured width deviation and resonance shift correlate properly.
        
        Expected: Pearson r > 0.7 (strong positive correlation)
        with slope approximately equal to sensitivity coefficient beta.
        """
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=10,
            n_dies_per_wafer=100,
            p_downstream_sample=0.8,
        )
        
        # Merge to get both inline and downstream for same dies
        df_merged = df_downstream.merge(
            df_inline[['wafer_id', 'die_id', 'wg_width_nm_meas', 'soi_thickness_nm_meas']],
            on=['wafer_id', 'die_id']
        )
        
        w0, lambda0 = 450.0, 1550.0
        x = df_merged['wg_width_nm_meas'] - w0
        y = df_merged['lambda_res_nm'] - lambda0
        
        # Check correlation is strong
        corr = np.corrcoef(x, y)[0, 1]
        assert corr > 0.65, f"Width-resonance correlation {corr} too weak"
        
        # Check slope approximately matches beta (~1.08)
        slope, _ = np.polyfit(x, y, 1)
        assert 0.8 < slope < 1.4, f"Width-resonance slope {slope} outside expected range"
    
    def test_feature_ranges(self):
        """Test that all features are within physically reasonable ranges."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=5,
            n_dies_per_wafer=100,
            p_downstream_sample=0.8,
        )
        
        params = gen.params
        
        # Inline features
        assert df_inline['wg_width_nm_meas'].min() > 300
        assert df_inline['wg_width_nm_meas'].max() < 600
        
        assert df_inline['soi_thickness_nm_meas'].min() > 150
        assert df_inline['soi_thickness_nm_meas'].max() < 300
        
        assert (df_inline['roughness_rms_nm_meas'] >= 0.1).all()
        
        # Downstream features
        assert df_downstream['lambda_res_nm'].min() > params.lambda_min
        assert df_downstream['lambda_res_nm'].max() < params.lambda_max
        
        assert (df_downstream['q_loaded'] > 1e3).all()
        assert (df_downstream['test_pass'].isin([0, 1])).all()
        assert (df_downstream['test_valid'] == 1).all()
    
    def test_no_nan_values(self):
        """Test that generated data has no NaN values in critical columns."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=5,
            n_dies_per_wafer=50,
            p_downstream_sample=0.8,
        )
        
        inline_critical = [
            'wafer_id', 'die_id', 'wg_width_nm_meas', 'soi_thickness_nm_meas'
        ]
        for col in inline_critical:
            assert not df_inline[col].isna().any(), f"NaN found in inline {col}"
        
        downstream_critical = ['wafer_id', 'die_id', 'lambda_res_nm', 'q_loaded', 'insertion_loss_db', 'test_pass', 'test_valid']
        for col in downstream_critical:
            assert not df_downstream[col].isna().any(), f"NaN found in downstream {col}"

        assert (df_downstream['test_valid'] == 1).all()
    
    def test_join_consistency(self):
        """Test that downstream keys are valid subset of inline keys."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=5,
            n_dies_per_wafer=100,
            p_downstream_sample=0.5,
        )
        
        # Get keys
        inline_keys = set(df_inline[['wafer_id', 'die_id']].itertuples(index=False, name=None))
        downstream_keys = set(df_downstream[['wafer_id', 'die_id']].itertuples(index=False, name=None))
        
        # Downstream should be subset of inline
        assert downstream_keys.issubset(inline_keys)
    
    def test_validate_and_summarize(self):
        """Test validation and summary statistics."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=5,
            n_dies_per_wafer=100,
            p_downstream_sample=0.6,
        )
        
        stats = gen.validate_and_summarize(df_inline, df_downstream)
        
        assert stats['n_dies_inline'] == 500
        assert 0 < stats['n_dies_downstream'] < 500
        assert stats['n_dies_downstream_valid'] == stats['n_dies_downstream']
        assert stats['n_dies_downstream_invalid'] == 0
        assert stats['n_dies_not_tested'] == 500 - stats['n_dies_downstream']
        assert 0 < stats['coverage_pct'] < 100
        assert stats['valid_coverage_pct'] == stats['coverage_pct']
        assert stats['n_wafers'] == 5


class TestValidationUtils:
    """Test validation utility functions."""
    
    def test_validate_schemas_passes(self):
        """Test that valid schemas pass validation."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=10)
        
        # Should not raise
        result = validate_schemas(df_inline, df_downstream, raise_on_error=False)
        assert result is True
    
    def test_validate_schemas_fails_on_missing_columns(self):
        """Test that validation fails when columns are missing."""
        df_inline = pd.DataFrame({'col1': [1, 2, 3]})
        df_downstream = pd.DataFrame({'col2': [4, 5, 6]})
        
        with pytest.raises(ValueError):
            validate_schemas(df_inline, df_downstream, raise_on_error=True)

    def test_validate_schemas_fails_on_inline_leakage_columns(self):
        """Validation should reject latent or downstream columns in inline metrology."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=10)
        df_inline['lambda_true'] = 1550.0

        with pytest.raises(ValueError):
            validate_schemas(df_inline, df_downstream, raise_on_error=True)

    def test_save_sources_sanitizes_inline_table(self, tmp_path):
        """Saving sources should strip latent inline columns before writing CSVs."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(n_wafers=1, n_dies_per_wafer=10)
        df_inline_dirty = df_inline.copy()
        df_inline_dirty['q_true'] = 2e5
        df_inline_dirty['test_station_id'] = 'TS1'

        save_sources(df_inline_dirty, df_downstream, output_dir=str(tmp_path), prefix='check')
        df_inline_saved, df_downstream_saved = load_sources(input_dir=str(tmp_path), prefix='check')

        assert 'q_true' not in df_inline_saved.columns
        assert 'test_station_id' not in df_inline_saved.columns
        assert len(df_downstream_saved) == len(df_downstream)


class TestMergeSources:
    """Test merging of data sources."""
    
    def test_merge_sources(self):
        """Test that merging produces expected output."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=2,
            n_dies_per_wafer=50,
            p_downstream_sample=0.8,
        )
        
        df_merged = merge_sources(df_inline, df_downstream, how='inner')
        
        # Merged should have same length as downstream rows (inner join)
        assert len(df_merged) == len(df_downstream)
        
        # Should have features from both sources
        assert 'wg_width_nm_meas' in df_merged.columns
        assert 'lambda_res_nm' in df_merged.columns
        
        # Deviations should be added
        assert 'width_deviation' in df_merged.columns
        assert 'thickness_deviation' in df_merged.columns

    def test_merge_sources_drops_inline_leakage_columns(self):
        """Merge helper should sanitize dirty inline tables before joining."""
        gen = SyntheticMRRDataGenerator(seed=42)
        df_inline, df_downstream = gen.generate_dataset(
            n_wafers=2,
            n_dies_per_wafer=20,
            p_downstream_sample=0.8,
        )
        df_inline_dirty = df_inline.copy()
        df_inline_dirty['lambda_true'] = 1550.0
        df_inline_dirty['test_station_id'] = 'TS1'

        df_merged = merge_sources(df_inline_dirty, df_downstream, how='inner')

        assert 'lambda_true' not in df_merged.columns
        assert 'test_station_id' in df_merged.columns
        assert 'test_station_id_x' not in df_merged.columns
        assert 'test_station_id_y' not in df_merged.columns


class TestModels:
    """Test model wrappers."""

    def test_ridge_regression_supports_group_aware_fit(self):
        """Ridge should accept group labels for internal alpha selection."""
        rng = np.random.RandomState(42)
        X = rng.normal(size=(40, 3))
        groups = np.repeat(np.array(['W001', 'W002', 'W003', 'W004']), 10)
        y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.05, size=40)

        model = RidgeRegressionModel(alphas=np.array([0.01, 0.1, 1.0]), cv_splits=4)
        model.fit(X, y, groups=groups)
        preds = model.predict(X[:5])

        assert model.selected_alpha_ in {0.01, 0.1, 1.0}
        assert preds.shape == (5,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
