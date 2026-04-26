"""I/O, validation, and a few plotting helpers used by the notebook."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


INLINE_REQUIRED_COLUMNS = {
    'wafer_id', 'lot_id', 'die_id', 'x_mm', 'y_mm',
    'wg_width_nm_meas', 'soi_thickness_nm_meas', 'etch_depth_nm_meas',
    'roughness_rms_nm_meas', 'overlay_x_nm_meas', 'overlay_y_nm_meas',
    'defect_density_cm2_meas', 'metrology_valid',
}

DOWNSTREAM_REQUIRED_COLUMNS = {
    'wafer_id', 'die_id', 'test_station_id',
    'lambda_res_nm', 'q_loaded', 'insertion_loss_db',
    'test_pass',
}

FORBIDDEN_INLINE_COLUMNS = {
    'test_station_id',
    'lambda_true',
    'q_true',
    'w_true',
    't_true',
    'roughness_true',
    'defect_true',
    'lambda_res_nm',
    'q_loaded',
    'insertion_loss_db',
    'test_pass',
    'lambda_res_nm_meas',
    'q_loaded_meas',
    'insertion_loss_db_meas',
}


def find_inline_leakage_columns(df_inline: pd.DataFrame) -> list[str]:
    """Return columns that should not appear in the public inline table."""
    leakage_columns = []
    for col in df_inline.columns:
        if col in FORBIDDEN_INLINE_COLUMNS or col.endswith('_true'):
            leakage_columns.append(col)
    return sorted(leakage_columns)


def sanitize_inline_metrology(df_inline: pd.DataFrame) -> pd.DataFrame:
    """Drop latent columns before saving or merging inline data."""
    leakage_columns = find_inline_leakage_columns(df_inline)
    if not leakage_columns:
        return df_inline.copy()
    return df_inline.drop(columns=leakage_columns, errors='ignore').copy()


def save_sources(
    df_inline: pd.DataFrame,
    df_downstream: pd.DataFrame,
    output_dir: str = './data',
    prefix: str = 'synthetic',
) -> None:
    """Save the two public tables to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_inline_public = sanitize_inline_metrology(df_inline)
    
    inline_path = output_path / f"{prefix}_inline_metrology.csv"
    downstream_path = output_path / f"{prefix}_downstream_wafer_test.csv"
    
    df_inline_public.to_csv(inline_path, index=False)
    df_downstream.to_csv(downstream_path, index=False)
    
    print(f"Saved inline metrology: {inline_path}")
    print(f"Saved downstream wafer test: {downstream_path}")


def load_sources(
    input_dir: str = './data',
    prefix: str = 'synthetic',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved CSV files."""
    input_path = Path(input_dir)
    
    inline_path = input_path / f"{prefix}_inline_metrology.csv"
    downstream_path = input_path / f"{prefix}_downstream_wafer_test.csv"
    
    if not inline_path.exists():
        raise FileNotFoundError(f"Inline metrology file not found: {inline_path}")
    if not downstream_path.exists():
        raise FileNotFoundError(f"Downstream wafer test file not found: {downstream_path}")
    
    df_inline = pd.read_csv(inline_path)
    df_downstream = pd.read_csv(downstream_path)
    
    print(f"Loaded inline metrology: {inline_path}")
    print(f"Loaded downstream wafer test: {downstream_path}")
    
    return df_inline, df_downstream


def validate_schemas(
    df_inline: pd.DataFrame,
    df_downstream: pd.DataFrame,
    raise_on_error: bool = True,
) -> bool:
    """Validate the public inline and downstream tables."""
    errors = []
    
    missing_inline = INLINE_REQUIRED_COLUMNS - set(df_inline.columns)
    if missing_inline:
        errors.append(f"Inline metrology missing columns: {missing_inline}")

    leakage_columns = find_inline_leakage_columns(df_inline)
    if leakage_columns:
        errors.append(
            "Inline metrology contains latent or downstream-only columns: "
            f"{set(leakage_columns)}"
        )
    
    if not df_inline.empty:
        # Check for NaNs in key columns
        critical_cols = ['wafer_id', 'die_id', 'wg_width_nm_meas', 'soi_thickness_nm_meas']
        for col in critical_cols:
            if col in df_inline.columns and df_inline[col].isna().any():
                errors.append(f"Inline metrology column '{col}' contains NaN values")
    
    missing_downstream = DOWNSTREAM_REQUIRED_COLUMNS - set(df_downstream.columns)
    if missing_downstream:
        errors.append(f"Downstream wafer test missing columns: {missing_downstream}")
    
    if not df_downstream.empty:
        critical_cols = ['wafer_id', 'die_id', 'test_pass']
        for col in critical_cols:
            if col in df_downstream.columns and df_downstream[col].isna().any():
                errors.append(f"Downstream test column '{col}' contains NaN values")

        if 'test_pass' in df_downstream.columns and not df_downstream['test_pass'].isin([0, 1]).all():
            errors.append("Downstream test column 'test_pass' must contain only 0/1 values")

        measured_cols = ['lambda_res_nm', 'q_loaded', 'insertion_loss_db']
        for col in measured_cols:
            if col in df_downstream.columns and df_downstream[col].isna().any():
                errors.append(f"Downstream test column '{col}' contains NaN values")
    
    if not df_inline.empty and not df_downstream.empty:
        if 'wafer_id' not in df_inline.columns or 'die_id' not in df_inline.columns:
            errors.append("Inline metrology missing required join keys: wafer_id, die_id")
        elif 'wafer_id' not in df_downstream.columns or 'die_id' not in df_downstream.columns:
            errors.append("Downstream test missing required join keys: wafer_id, die_id")
        else:
            inline_keys = set(df_inline[['wafer_id', 'die_id']].itertuples(index=False, name=None))
            downstream_keys = set(df_downstream[['wafer_id', 'die_id']].itertuples(index=False, name=None))
            bad_keys = downstream_keys - inline_keys
            if bad_keys:
                errors.append(f"Downstream test contains {len(bad_keys)} key(s) not in inline metrology")
    
    if errors:
        error_msg = "Schema validation failed:\n" + "\n".join(errors)
        if raise_on_error:
            raise ValueError(error_msg)
        else:
            print(error_msg)
            return False
    
    return True


def merge_sources(
    df_inline: pd.DataFrame,
    df_downstream: pd.DataFrame,
    how: str = 'inner',
) -> pd.DataFrame:
    """Merge inline metrology with downstream test data."""
    df_inline_public = sanitize_inline_metrology(df_inline)
    df_downstream_public = df_downstream.copy()

    df_merged = df_downstream_public.merge(
        df_inline_public,
        on=['wafer_id', 'die_id'],
        how=how,
    )
    
    if 'wg_width_nm_meas' in df_merged.columns and 'soi_thickness_nm_meas' in df_merged.columns:
        w0, t0, lambda0 = 450.0, 220.0, 1550.0
        df_merged['width_deviation'] = df_merged['wg_width_nm_meas'] - w0
        df_merged['thickness_deviation'] = df_merged['soi_thickness_nm_meas'] - t0
        df_merged['lambda_deviation'] = df_merged['lambda_res_nm'] - lambda0
    
    return df_merged

def plot_feature_distributions(
    df_inline: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """Plot the main continuous inline features."""
    continuous_cols = [
        'wg_width_nm_meas',
        'soi_thickness_nm_meas',
        'etch_depth_nm_meas',
        'roughness_rms_nm_meas',
        'overlay_x_nm_meas',
        'overlay_y_nm_meas',
        'defect_density_cm2_meas',
    ]
    
    available_cols = [col for col in continuous_cols if col in df_inline.columns]
    n_cols = 3
    n_rows = (len(available_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(available_cols):
        ax = axes[idx]
        df_inline[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of {col}')
        ax.grid(True, alpha=0.3)
    
    for idx in range(len(available_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_inline_vs_resonance(
    df_merged: pd.DataFrame,
    w0: float = 450.0,
    lambda0: float = 1550.0,
    figsize: Tuple[int, int] = (11, 6),
) -> None:
    """Plot measured inline width deviation against downstream resonance shift.

    Points are colored by wafer_id when wafer_id is available.
    """
    required_cols = {"wg_width_nm_meas", "lambda_res_nm"}

    if not required_cols.issubset(df_merged.columns):
        print("Warning: Required columns not found for plot_inline_vs_resonance")
        return

    x = df_merged["wg_width_nm_meas"] - w0
    y = df_merged["lambda_res_nm"] - lambda0

    valid = x.notna() & y.notna()

    fig, ax = plt.subplots(figsize=figsize)

    if "wafer_id" in df_merged.columns:
        wafer_ids = sorted(df_merged.loc[valid, "wafer_id"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(wafer_ids)))

        for wafer_id, color in zip(wafer_ids, colors):
            mask = valid & (df_merged["wafer_id"] == wafer_id)

            ax.scatter(
                x[mask],
                y[mask],
                s=20,
                alpha=0.6,
                color=color,
                label=f"Wafer {wafer_id}",
            )
    else:
        wafer_ids = []
        ax.scatter(
            x[valid],
            y[valid],
            s=20,
            alpha=0.6,
            label="Tested dies",
        )

    # Linear fit over all tested dies
    z = np.polyfit(x[valid], y[valid], 1)
    p = np.poly1d(z)

    x_line = np.linspace(x[valid].min(), x[valid].max(), 100)

    ax.plot(
        x_line,
        p(x_line),
        "r--",
        linewidth=2,
        label=f"Linear fit: y = {z[0]:.3f}x + {z[1]:.3f}",
    )

    ax.set_xlabel("Measured width deviation from nominal (nm)", fontsize=12)
    ax.set_ylabel("Measured resonance shift from nominal (nm)", fontsize=12)
    ax.set_title(
        "Measured Width Deviation vs Downstream Resonance Shift",
        fontsize=13,
    )

    ax.grid(True, alpha=0.3)

    # Legend outside the plot
    if "wafer_id" in df_merged.columns:
        ax.legend(
            title="Color = wafer_id",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=8,
            title_fontsize=9,
            ncol=1,
            frameon=True,
        )
    else:
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()
