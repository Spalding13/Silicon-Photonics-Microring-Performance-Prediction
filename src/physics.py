"""Core configuration and physics-inspired helper functions.

The functions in this module are intentionally simple. They are not meant to be
full electromagnetic or process simulators. Instead, they provide a compact
physical backbone for generating a synthetic, inspectable silicon photonics
microring dataset.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SyntheticMRRProcessConfig:
    """Configuration for the synthetic MRR manufacturing process.

    This dataclass groups the parameters used by the synthetic generator:

    - nominal microring design values,
    - first-order resonance sensitivity coefficients,
    - lot-level, wafer-level, die-level, and measurement variation scales,
    - spatial effect parameters,
    - downstream pass/fail specification limits.

    The values are chosen to make a realistic toy dataset, not to calibrate a
    specific confidential foundry process.
    """

    # Nominal design point.
    lambda0: float = 1550.0
    w0: float = 450.0
    t0: float = 220.0

    # First-order sensitivity coefficients for resonance wavelength.
    # Units are approximately nm wavelength shift per nm geometry shift.
    alpha: float = 1.25
    beta: float = 1.08

    # Lot-level process variation.
    # These offsets are shared by all wafers in the same synthetic lot.
    # They model slow shifts such as batch material, tool condition, or recipe drift.
    sigma_w_lot: float = 0.25
    sigma_t_lot: float = 0.20
    roughness_lot_std: float = 0.05
    defect_density_lot_std: float = 0.10

    # Wafer-level and die-level process variation.
    sigma_w_wafer: float = 2.0
    sigma_t_wafer: float = 1.5
    sigma_w_die: float = 0.3
    sigma_t_die: float = 0.2
    sigma_roughness_die: float = 0.1

    # Inline metrology measurement noise.
    sigma_w_meas: float = 0.5
    sigma_t_meas: float = 0.5
    sigma_roughness_meas: float = 0.15
    sigma_defect_meas: float = 100.0
    sigma_overlay_meas: float = 2.0

    # Downstream optical test measurement noise.
    sigma_lambda_meas: float = 0.05

    # Nominal Q model and noise.
    q0: float = 2e5
    k_r: float = 0.08
    k_d: float = 1.2e-4
    sigma_q_meas: float = 0.1

    # Smooth within-wafer spatial field parameters.
    spatial_field_scale: float = 1.0
    spatial_rms: float = 0.5

    # Baseline roughness and defectivity distributions.
    roughness_mean: float = 1.0
    roughness_std: float = 0.3
    defect_density_mean: float = 1000.0
    defect_density_std: float = 0.5

    # Broad sanity bounds used by validation/tests.
    lambda_min: float = 1500.0
    lambda_max: float = 1600.0
    w_min: float = 350.0
    w_max: float = 550.0
    t_min: float = 200.0
    t_max: float = 260.0

    # Downstream specification limits.
    # lambda is checked against a window; Q is checked against a minimum threshold.
    lambda_spec_min: float = 1545.0
    lambda_spec_max: float = 1553.0
    q_spec_min: float = 145000.0
    q_mnar_threshold: float = 140000.0

    # Local spatial degradation parameters.
    # Edge effect mainly degrades Q through roughness and defectivity.
    edge_failure_strength: float = 1.0
    edge_failure_center: float = 0.78
    edge_failure_width: float = 0.06

    # Semi-ring effect creates a localized angular/radial pattern.
    ring_failure_strength: float = 1.0
    ring_failure_radius: float = 0.58
    ring_failure_width: float = 0.08
    ring_failure_angle_deg: float = 30.0

    def __post_init__(self) -> None:
        """Validate parameter ranges immediately after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Raise ValueError if configuration values are clearly invalid."""
        if not (1500 <= self.lambda0 <= 1600):
            raise ValueError(f"lambda0 out of valid range: {self.lambda0}")
        if not (300 <= self.w0 <= 600):
            raise ValueError(f"w0 out of valid range: {self.w0}")
        if not (150 <= self.t0 <= 300):
            raise ValueError(f"t0 out of valid range: {self.t0}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Sensitivity coefficients must be non-negative")
        if self.lambda_spec_min >= self.lambda_spec_max:
            raise ValueError("lambda_spec_min must be smaller than lambda_spec_max")
        if self.q_spec_min <= 0 or self.q_mnar_threshold <= 0:
            raise ValueError("Q thresholds must be positive")
        if self.edge_failure_width <= 0 or self.ring_failure_width <= 0:
            raise ValueError("Local failure widths must be positive")
        if not (0.0 <= self.edge_failure_center <= 1.0):
            raise ValueError("edge_failure_center must be between 0 and 1")
        if not (0.0 <= self.ring_failure_radius <= 1.0):
            raise ValueError("ring_failure_radius must be between 0 and 1")
        if self.edge_failure_strength < 0 or self.ring_failure_strength < 0:
            raise ValueError("Local failure strengths must be non-negative")
        if any(
            s < 0
            for s in [
                self.sigma_w_lot,
                self.sigma_t_lot,
                self.roughness_lot_std,
                self.defect_density_lot_std,
                self.sigma_w_wafer,
                self.sigma_t_wafer,
                self.sigma_w_die,
                self.sigma_t_die,
                self.sigma_w_meas,
                self.sigma_t_meas,
            ]
        ):
            raise ValueError("Noise scales must be non-negative")

    def to_dict(self) -> dict:
        """Export the full configuration as a plain dictionary."""
        return asdict(self)


def compute_resonance_wavelength(
    t_true: np.ndarray,
    w_true: np.ndarray,
    t0: float,
    w0: float,
    lambda0: float,
    alpha: float,
    beta: float,
    eta_lambda: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute resonance wavelength from a first-order sensitivity model.

    The model is a local linear approximation around the nominal design point:

        lambda = lambda0 + alpha * (t_true - t0) + beta * (w_true - w0) + noise

    Args:
        t_true: True silicon thickness values.
        w_true: True waveguide width values.
        t0: Nominal silicon thickness.
        w0: Nominal waveguide width.
        lambda0: Nominal resonance wavelength.
        alpha: Sensitivity to silicon thickness.
        beta: Sensitivity to waveguide width.
        eta_lambda: Optional additive residual/noise term.

    Returns:
        Array of generated resonance wavelengths.
    """
    n = len(t_true)
    if eta_lambda is None:
        eta_lambda = np.zeros(n)
    return lambda0 + alpha * (t_true - t0) + beta * (w_true - w0) + eta_lambda


def compute_log_q(
    roughness: np.ndarray,
    defect_density: np.ndarray,
    q0: float,
    k_r: float,
    k_d: float,
    eta_q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute log(Q) from a simple degradation model.

    Roughness and defect density reduce optical quality. The model is written in
    log space so multiplicative Q degradation becomes additive.

    Args:
        roughness: Roughness-related degradation proxy.
        defect_density: Defect-density-related degradation proxy.
        q0: Nominal high-quality loaded Q.
        k_r: Roughness penalty coefficient.
        k_d: Defect-density penalty coefficient.
        eta_q: Optional additive residual/noise term in log-Q space.

    Returns:
        Array of log-Q values.
    """
    n = len(roughness)
    if eta_q is None:
        eta_q = np.zeros(n)
    return np.log(q0) - k_r * roughness - k_d * defect_density + eta_q


def generate_radial_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    wafer_radius_mm: float = 75.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a smooth radial field that grows from center to edge.

    Args:
        x: Die x-positions in millimeters.
        y: Die y-positions in millimeters.
        wafer_radius_mm: Radius used to normalize radial distance.
        amplitude: Overall field scale.

    Returns:
        Smooth radial field evaluated at each die position.
    """
    r = np.sqrt(x**2 + y**2)
    normalized_r = r / wafer_radius_mm
    return amplitude * (0.5 * normalized_r + 0.3 * normalized_r**2)


def generate_angular_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Generate a smooth angular asymmetry field.

    The field creates non-radial variation so the wafer maps do not look like a
    perfectly symmetric center-to-edge gradient.
    """
    theta = np.arctan2(y, x)
    return amplitude * (0.4 * np.sin(theta) + 0.3 * np.cos(2 * theta))


def generate_random_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    scale: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Generate a low-frequency random spatial field.

    The field is a small sum of sinusoidal components with random orientation and
    phase. It creates spatially correlated process variation without using a
    full Gaussian-process model.
    """
    if rng is None:
        rng = np.random.RandomState()

    field = np.zeros(len(x))
    for freq_idx in range(3):
        kx = rng.randn() * 0.1 * (freq_idx + 1)
        ky = rng.randn() * 0.1 * (freq_idx + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = scale / (freq_idx + 1) ** 1.5
        field += amplitude * np.sin(kx * x + ky * y + phase)
    return field


def combine_spatial_fields(
    radial: np.ndarray,
    angular: np.ndarray,
    random: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> np.ndarray:
    """Blend radial, angular, and random fields into one spatial field.

    Args:
        radial: Center-to-edge spatial component.
        angular: Smooth angular spatial component.
        random: Low-frequency random spatial component.
        weights: Relative weights for the three components.

    Returns:
        Weighted sum of the input spatial fields.
    """
    w_radial, w_angular, w_random = weights
    return w_radial * radial + w_angular * angular + w_random * random
