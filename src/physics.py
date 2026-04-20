"""
Physics module for silicon photonics microring resonators.

Encapsulates physical parameters, sensitivity coefficients, and mathematical models
for computing resonance wavelength and quality factor from geometric variations.

References:
- Barwicz, T., et al. "Silicon photonics." Nature Photonics 2005+
- Silicon photonics process variation characterization studies
- Ring resonator resonance wavelength sensitivity: ~1.25 nm/nm (thickness), ~1.08 nm/nm (width)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class MRRParameters:
    """
    Physical and design parameters for silicon photonics microring resonators.
    
    This class encapsulates:
    - Nominal design values (geometry and resonance)
    - Sensitivity coefficients (resonance shift per geometry change)
    - Measurement noise scales
    - Spatial variation scales (wafer-level drifts)
    - Q degradation parameters (roughness, defects)
    
    All parameters are configurable for ablation studies; defaults are physically motivated.
    """
    
    # ===== Nominal Design Values =====
    lambda0: float = 1550.0
    """Nominal resonance wavelength (nm). Telecom band common in silicon photonics."""
    
    w0: float = 450.0
    """Nominal waveguide width (nm). Typical single-mode-like width for 220 nm SOI."""
    
    t0: float = 220.0
    """Nominal SOI device-layer thickness (nm). 220 nm is standard platform."""
    
    # ===== Sensitivity Coefficients =====
    alpha: float = 1.25
    """Resonance shift per nm of thickness change: dλ/dt (nm/nm).
    
    Published value for silicon photonics ring resonators on 220 nm SOI.
    Physical basis: thickness controls mode confinement and effective refractive index."""
    
    beta: float = 1.08
    """Resonance shift per nm of width change: dλ/dw (nm/nm).
    
    Published value for silicon photonics ring resonators.
    Physical basis: width controls effective index of propagation mode."""
    
    # ===== Wafer-Level Spatial Variation (Latent) =====
    sigma_w_wafer: float = 2.0
    """Std. dev. of wafer-level width drift (nm). Represents systematic per-wafer uniformity."""
    
    sigma_t_wafer: float = 1.5
    """Std. dev. of wafer-level thickness drift (nm)."""
    
    sigma_w_die: float = 0.3
    """Std. dev. of die-level width residuals (nm). Within-die, random variation."""
    
    sigma_t_die: float = 0.2
    """Std. dev. of die-level thickness residuals (nm)."""
    
    sigma_roughness_die: float = 0.1
    """Std. dev. of die-level roughness residuals (nm)."""
    
    # ===== Measurement Noise (Inline Metrology) =====
    sigma_w_meas: float = 0.5
    """Std. dev. of width measurement noise (nm). Typical for CD-SEM or OCD."""
    
    sigma_t_meas: float = 0.5
    """Std. dev. of thickness measurement noise (nm). Typical for ellipsometry."""
    
    sigma_roughness_meas: float = 0.15
    """Std. dev. of roughness measurement noise (nm)."""
    
    sigma_defect_meas: float = 100.0
    """Std. dev. of defect density measurement noise (defects/cm²)."""
    
    sigma_overlay_meas: float = 2.0
    """Std. dev. of overlay measurement noise (nm)."""
    
    # ===== Downstream Optical Test Measurement Noise =====
    sigma_lambda_meas: float = 0.05
    """Std. dev. of resonance wavelength measurement noise in downstream test (nm).
    
    Optical test precision (e.g., tunable laser interrogation)."""
    
    # ===== Q Factor Parameters =====
    q0: float = 2e5
    """Nominal (ideal) loaded Q factor. Typical for MRRs in silicon photonics."""
    
    k_r: float = 1e-4
    """Roughness-to-Q degradation coefficient. Scale: log(Q) penalty per nm of roughness."""
    
    k_d: float = 1e-6
    """Defect-density-to-Q degradation coefficient. Scale: log(Q) penalty per defect/cm²."""
    
    sigma_q_meas: float = 0.1
    """Std. dev. of log(Q) measurement noise in downstream test."""
    
    # ===== Spatial Field Parameters =====
    spatial_field_scale: float = 1.0
    """Global scaling factor for amplitude of spatial variation fields (radial + angular)."""
    
    spatial_rms: float = 0.5
    """RMS amplitude of spatial random field perturbations as fraction of wafer radius."""
    
    # ===== Roughness & Defect Latent Parameters =====
    roughness_mean: float = 1.0
    """Mean RMS roughness (nm) on wafer (log-normally distributed)."""
    
    roughness_std: float = 0.3
    """Std. dev. of log(roughness) (log-space)."""
    
    defect_density_mean: float = 1000.0
    """Mean defect density (defects/cm²) on wafer (log-normally distributed)."""
    
    defect_density_std: float = 0.5
    """Std. dev. of log(defect_density) (log-space)."""
    
    # ===== Validation Bounds =====
    lambda_min: float = 1500.0
    """Minimum valid resonance wavelength (nm)."""
    
    lambda_max: float = 1600.0
    """Maximum valid resonance wavelength (nm)."""
    
    w_min: float = 350.0
    """Minimum valid waveguide width (nm)."""
    
    w_max: float = 550.0
    """Maximum valid waveguide width (nm)."""
    
    t_min: float = 200.0
    """Minimum valid thickness (nm)."""
    
    t_max: float = 260.0
    """Maximum valid thickness (nm)."""
    
    def __post_init__(self):
        """Validate parameter ranges after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Check that all parameters are within reasonable ranges."""
        if not (1500 <= self.lambda0 <= 1600):
            raise ValueError(f"lambda0 out of valid range: {self.lambda0}")
        if not (300 <= self.w0 <= 600):
            raise ValueError(f"w0 out of valid range: {self.w0}")
        if not (150 <= self.t0 <= 300):
            raise ValueError(f"t0 out of valid range: {self.t0}")
        if self.alpha < 0 or self.beta < 0:
            raise ValueError("Sensitivity coefficients must be non-negative")
        if any(s < 0 for s in [self.sigma_w_wafer, self.sigma_t_wafer, 
                                self.sigma_w_die, self.sigma_t_die,
                                self.sigma_w_meas, self.sigma_t_meas]):
            raise ValueError("Noise scales must be non-negative")
    
    def to_dict(self) -> dict:
        """Export parameters as dictionary (useful for logging, reproducibility)."""
        return {
            'lambda0': self.lambda0,
            'w0': self.w0,
            't0': self.t0,
            'alpha': self.alpha,
            'beta': self.beta,
            'sigma_w_wafer': self.sigma_w_wafer,
            'sigma_t_wafer': self.sigma_t_wafer,
            'sigma_w_die': self.sigma_w_die,
            'sigma_t_die': self.sigma_t_die,
            'sigma_w_meas': self.sigma_w_meas,
            'sigma_t_meas': self.sigma_t_meas,
            'sigma_lambda_meas': self.sigma_lambda_meas,
            'q0': self.q0,
            'k_r': self.k_r,
            'k_d': self.k_d,
        }


# ============================================================================
# Physical Models (Functions)
# ============================================================================

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
    """
    Compute resonance wavelength from true geometry using published sensitivity model.
    
    Linear first-order model (valid in small-deviation regime):
        λ = λ₀ + α(t - t₀) + β(w - w₀) + η^(λ)
    
    Physical basis:
    - Both thickness (t) and width (w) control modal effective refractive index
    - Effective index change → resonance wavelength shift
    - Sensitivity coefficients (α, β) from silicon photonics literature
    
    Args:
        t_true: True thickness values (nm), shape (n_dies,)
        w_true: True width values (nm), shape (n_dies,)
        t0: Nominal thickness (nm)
        w0: Nominal width (nm)
        lambda0: Nominal resonance wavelength (nm)
        alpha: Thickness sensitivity coefficient (nm/nm)
        beta: Width sensitivity coefficient (nm/nm)
        eta_lambda: Optional measurement noise, shape (n_dies,). If None, zeros used.
    
    Returns:
        lambda_true: Computed resonance wavelengths (nm), shape (n_dies,)
    """
    n = len(t_true)
    if eta_lambda is None:
        eta_lambda = np.zeros(n)
    
    lambda_computed = lambda0 + alpha * (t_true - t0) + beta * (w_true - w0) + eta_lambda
    
    return lambda_computed


def compute_log_q(
    roughness: np.ndarray,
    defect_density: np.ndarray,
    q0: float,
    k_r: float,
    k_d: float,
    eta_q: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute logged quality factor from roughness and defect density.
    
    Model:
        log(Q) = log(Q₀) - k_r · roughness - k_d · defect_density + η^(Q)
    
    Physical basis:
    - Roughness and defects introduce optical scattering losses
    - Scattering reduces cavity photon lifetime → lower Q
    - Log-linear model captures dominant loss mechanisms
    
    Args:
        roughness: RMS surface roughness (nm), shape (n_dies,)
        defect_density: Defect density (defects/cm²), shape (n_dies,)
        q0: Nominal ideal Q (dimensionless)
        k_r: Roughness-to-Q degradation coefficient (log(Q)/nm)
        k_d: Defect-to-Q degradation coefficient (log(Q)/(defects/cm²))
        eta_q: Optional log(Q) measurement noise, shape (n_dies,). If None, zeros used.
    
    Returns:
        log_q: Computed log(Q), shape (n_dies,)
    """
    n = len(roughness)
    if eta_q is None:
        eta_q = np.zeros(n)
    
    log_q = np.log(q0) - k_r * roughness - k_d * defect_density + eta_q
    
    return log_q


def generate_radial_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    wafer_radius_mm: float = 75.0,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Generate radial (r-dependent) component of spatial variation field.
    
    Models radial drift typical in deposition/CMP processes.
    
    Args:
        x: X-coordinates on wafer (mm), shape (n_dies,)
        y: Y-coordinates on wafer (mm), shape (n_dies,)
        wafer_radius_mm: Wafer radius in mm
        amplitude: Scaling amplitude of radial variation
    
    Returns:
        field: Radial spatial field values, shape (n_dies,)
    """
    r = np.sqrt(x**2 + y**2)
    normalized_r = r / wafer_radius_mm
    # Low-order radial term: quadratic
    field = amplitude * (0.5 * normalized_r + 0.3 * normalized_r**2)
    return field


def generate_angular_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    Generate angular (θ-dependent) component of spatial variation field.
    
    Models angular asymmetries typical in lithography/etch patterns.
    
    Args:
        x: X-coordinates on wafer (mm), shape (n_dies,)
        y: Y-coordinates on wafer (mm), shape (n_dies,)
        amplitude: Scaling amplitude of angular variation
    
    Returns:
        field: Angular spatial field values, shape (n_dies,)
    """
    theta = np.arctan2(y, x)
    # Sinusoidal angular term
    field = amplitude * (0.4 * np.sin(theta) + 0.3 * np.cos(2 * theta))
    return field


def generate_random_spatial_field(
    x: np.ndarray,
    y: np.ndarray,
    scale: float = 1.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate low-frequency random spatial field (Gaussian random field proxy).
    
    Simple implementation: random phase sinusoids with decreasing amplitudes.
    For production, consider scipy.spatial.distance.pdist + covariance matrix.
    
    Args:
        x: X-coordinates on wafer (mm), shape (n_dies,)
        y: Y-coordinates on wafer (mm), shape (n_dies,)
        scale: Overall scale of random field
        rng: Random state for reproducibility
    
    Returns:
        field: Random spatial field values, shape (n_dies,)
    """
    if rng is None:
        rng = np.random.RandomState()
    
    n = len(x)
    # Generate a few random "frequencies" and sum them to create low-freq field
    field = np.zeros(n)
    for freq_idx in range(3):
        kx = rng.randn() * 0.1 * (freq_idx + 1)
        ky = rng.randn() * 0.1 * (freq_idx + 1)
        phase = rng.uniform(0, 2 * np.pi)
        amplitude = scale / (freq_idx + 1)**1.5  # Decreasing amplitude for higher freqs
        field += amplitude * np.sin(kx * x + ky * y + phase)
    
    return field


def combine_spatial_fields(
    radial: np.ndarray,
    angular: np.ndarray,
    random: np.ndarray,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> np.ndarray:
    """
    Combine radial, angular, and random spatial field components.
    
    Args:
        radial: Radial field component, shape (n_dies,)
        angular: Angular field component, shape (n_dies,)
        random: Random field component, shape (n_dies,)
        weights: Tuple of (w_radial, w_angular, w_random) summing to ~1
    
    Returns:
        combined: Combined spatial field, shape (n_dies,)
    """
    w_r, w_a, w_rnd = weights
    combined = w_r * radial + w_a * angular + w_rnd * random
    return combined
