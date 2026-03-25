"""
Helper functions (normalization, tensor conversions)
"""

def normalize_score(score: float) -> float:
    return max(0.0, min(1.0, score))
