"""FaithSR research and scale-aware satellite inference package."""

from .model import FaithSR3X, FaithSRConfig, count_parameters

__all__ = ["FaithSR3X", "FaithSRConfig", "count_parameters"]
__version__ = "0.3.0"
