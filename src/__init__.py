"""
Semiconductor Surrogate Modeling Package

Synthetic data generation and surrogate modeling for silicon photonics 
microring resonator performance prediction from inline metrology.
"""

__version__ = "0.1.0"
__author__ = "Data Science Project Team"

from .physics import MRRParameters
from .generator import SyntheticMRRDataGenerator

__all__ = [
    "MRRParameters",
    "SyntheticMRRDataGenerator",
]
