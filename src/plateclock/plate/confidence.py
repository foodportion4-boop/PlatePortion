"""Ranking scores; these are not calibrated correctness probabilities."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Decision:
    score: float
    action: str

def decide(probability, area_fraction, fit_quality, config):
    if not all(np.isfinite(v) and 0 <= v <= 1 for v in (probability, area_fraction, fit_quality)):
        raise ValueError('Confidence factors must be finite and in [0,1]')
    low, high = config['omit_below'], config['keep_at']
    full = config['full_size_fraction']
    if not 0 <= low <= high <= 1 or not 0 < full <= 1:
        raise ValueError('Invalid confidence thresholds')
    score = float(probability * min(1., area_fraction / full) * fit_quality)
    return Decision(score, 'keep' if score >= high else 'hedge' if score >= low else 'omit')
