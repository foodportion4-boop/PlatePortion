"""Build a visibly synthetic RGB-only presentation depth map.

No metric depth is inferred from colour. The assumptions below exist solely to
exercise the UI and geometric arithmetic in a clearly identified demo mode.
"""
from __future__ import annotations

import cv2
import numpy as np

from n5k.geometric.volume import CameraModel


def synthesize(labels: np.ndarray, plate: np.ndarray, shape=(480, 640)) -> tuple[np.ndarray, dict]:
    """Return raw-style uint16 PNG pixels and illustrative whole-plate values."""
    if labels.shape != plate.shape or labels.ndim != 2:
        raise ValueError('A matching label map and plate mask are required.')
    width, height = shape[1], shape[0]
    plate_s = cv2.resize(plate.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    food = cv2.resize(((labels > 0) & (labels != 255) & plate).astype(np.uint8),
                      (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    if not plate_s.any() or food.sum() < 50:
        raise ValueError('No usable plate and food mask for the demo surface.')
    # Assumed horizontal plate at 40 cm; 25 mm high rounded food regions.
    distance = cv2.distanceTransform(food.astype(np.uint8), cv2.DIST_L2, 5)
    height_m = np.where(food, 0.025 * np.minimum(distance / 18.0, 1.0), 0.0)
    depth_m = np.full((height, width), 0.4, dtype=np.float32) - height_m
    raw = np.rint(depth_m * 10000).astype(np.uint16)
    pixel_area = CameraModel().pixel_area_m2(depth_m)
    volume_cm3 = float(np.sum(height_m[food] * pixel_area[food]) * 1e6)
    mass_g = volume_cm3 * 0.757
    return raw, {
        'method': 'RGB mask extrusion with assumed 40 cm camera distance and 2.5 cm food height',
        'status': 'illustrative', 'volume_cm3': round(volume_cm3, 1),
        'container': 'assumed flat plate', 'filled_fraction': 0,
        'nutrition': {'mass': round(mass_g, 1), 'calories': round(mass_g * 1.07, 1)},
        'direct': None,
        'reason': 'Synthetic depth and assumed densities. These figures are for a presentation demo only; they are not measured portions or nutritional advice.'
    }
