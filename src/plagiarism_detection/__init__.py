"""Evidence-first pairwise text similarity."""

from .analyzer import AnalysisConfig, PairwiseAnalyzer
from .models import AnalysisResult, EvidenceMatch

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "EvidenceMatch",
    "PairwiseAnalyzer",
]

__version__ = "2.0.0"
