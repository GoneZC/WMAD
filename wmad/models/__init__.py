"""WMAD models module."""

from .mead import MEAD, MEADLoss
from .base_model import BaseDeepAD

# Alias for external use
WMAD = MEAD
WMADLoss = MEADLoss

__all__ = ["MEAD", "MEADLoss", "WMAD", "WMADLoss", "BaseDeepAD"]


