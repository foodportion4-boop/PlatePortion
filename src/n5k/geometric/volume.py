"""Integrate food volume above the plate surface.

Volume is a per-pixel prism sum::

    V = sum_over_food_pixels  h(u,v) * pixel_area(z(u,v))

where ``h`` is height above the fitted plate surface and the pixel footprint of
a pinhole camera at depth ``z`` is ``(z/fx) * (z/fy)`` square metres.

Nutrition5k does **not** publish camera intrinsics, so ``fx``/``fy`` are
configuration, not fact. The defaults below are nominal Intel RealSense D435
values at 640x480. Rather than trust them blindly, the calibration step
(``n5k.geometric.calibrate``) fits food density on the *training split* and
reports it -- if the intrinsics are roughly right the fitted density lands in a
physically plausible 0.3-1.2 g/cm3 band, which is a genuine check on the
assumption. A wildly out-of-band density means the intrinsics are wrong, and the
number to report is that diagnosis, not a silently-rescaled result.

Dropouts matter here. 11-33% of pixels per frame have no depth return, and they
cluster on exactly the specular and dark surfaces food presents. Integrating
only over valid pixels systematically *under*-counts volume, so we fill holes
inside the plate region before integrating and record the filled fraction as a
per-dish confidence signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from n5k.data.dataset import DEPTH_FAR_M, DEPTH_NEAR_M
from n5k.geometric.surfaces import Surfaces, estimate_surfaces

#: Nominal RealSense D435 colour-stream focal lengths at 640x480, in pixels.
#: Treat as a prior to be validated by calibration, not as ground truth.
DEFAULT_FX = 615.0
DEFAULT_FY = 615.0


@dataclass
class CameraModel:
    fx: float = DEFAULT_FX
    fy: float = DEFAULT_FY

    def pixel_area_m2(self, z: np.ndarray) -> np.ndarray:
        """Square metres subtended by one pixel at depth ``z`` metres."""
        return (z / self.fx) * (z / self.fy)


@dataclass
class VolumeResult:
    dish_id: str
    volume_cm3: float
    food_pixels: int
    filled_fraction: float   # share of integrated pixels that were interpolated
    plate_height_cm: float
    mean_food_height_cm: float
    max_food_height_cm: float
    tilt_deg: float
    container: str           # "plate" | "bowl" | "unknown"
    concavity_cm: float
    ok: bool

    @property
    def reliable(self) -> bool:
        """Whether this estimate rests on assumptions that actually hold.

        A planar datum is only meaningful for a flat container, and a mostly
        interpolated food region is mostly guesswork. Use this to filter before
        reporting geometric-branch metrics -- and report the excluded fraction.
        """
        return self.ok and self.container == "plate" and self.filled_fraction < 0.35


def _fill_dropouts(
    height: np.ndarray, known: np.ndarray, region: np.ndarray
) -> tuple[np.ndarray, float]:
    """Nearest-neighbour fill of unknown heights inside ``region``.

    Returns the filled height map and the fraction of ``region`` that had to be
    filled. Nearest-neighbour rather than smooth interpolation: dropouts sit on
    steep specular edges where a smooth fit overshoots badly.
    """
    to_fill = region & ~known
    if not to_fill.any():
        return height, 0.0
    if not (known & region).any():
        return np.zeros_like(height), 1.0
    # distance_transform_edt with return_indices gives, for each pixel, the
    # coordinates of the nearest known pixel.
    _, (iy, ix) = ndimage.distance_transform_edt(
        ~(known & region), return_indices=True
    )
    filled = height.copy()
    filled[to_fill] = height[iy[to_fill], ix[to_fill]]
    return filled, float(to_fill.sum() / max(region.sum(), 1))


def estimate_volume(
    depth_m: np.ndarray,
    valid: np.ndarray,
    dish_id: str = "",
    camera: CameraModel | None = None,
    min_food_height_m: float = 0.004,
    max_food_height_m: float = 0.15,
    surfaces: Surfaces | None = None,
    **surface_kwargs,
) -> VolumeResult:
    """Estimate food volume in cm3 for one overhead RGB-D capture.

    Args:
        min_food_height_m: heights below this are plate texture and sensor
            noise, not food. 4 mm is a little above the RealSense noise floor at
            this range.
        max_food_height_m: guards against a failed plate fit turning the whole
            tray into "food".
    """
    camera = camera or CameraModel()
    surf = surfaces or estimate_surfaces(depth_m, valid, **surface_kwargs)

    if not surf.ok:
        return VolumeResult(
            dish_id, 0.0, 0, 1.0, 0.0, 0.0, 0.0, surf.tilt_deg,
            surf.container, surf.concavity_m * 100, False,
        )

    in_range = valid & (depth_m > DEPTH_NEAR_M) & (depth_m < DEPTH_FAR_M)
    height = np.where(in_range, surf.plate_plane - depth_m, np.nan)

    # Integrate over the rim-stripped interior, never the full container: the
    # raised rim sits above the floor plane and would be counted as food.
    food = (
        surf.interior_mask
        & in_range
        & (height > min_food_height_m)
        & (height < max_food_height_m)
    )
    # Drop isolated speckle, then close pinholes within food blobs.
    food = ndimage.binary_opening(food, structure=np.ones((3, 3)))
    food = ndimage.binary_closing(food, structure=np.ones((5, 5)))

    if food.sum() < 50:
        return VolumeResult(
            dish_id, 0.0, 0, 0.0, surf.plate_height_m * 100, 0.0, 0.0,
            surf.tilt_deg, surf.container, surf.concavity_m * 100, True,
        )

    # Integrate over the food region including its dropout holes.
    integrate = ndimage.binary_fill_holes(food)
    known = in_range & np.isfinite(height)
    h_filled, filled_frac = _fill_dropouts(np.nan_to_num(height), known, integrate)
    h_filled = np.clip(h_filled, 0.0, max_food_height_m)

    z = np.where(in_range, depth_m, surf.plate_plane)
    area = camera.pixel_area_m2(z)
    volume_m3 = float((h_filled * area)[integrate].sum())

    hf = h_filled[integrate]
    return VolumeResult(
        dish_id=dish_id,
        volume_cm3=volume_m3 * 1e6,           # m3 -> cm3
        food_pixels=int(integrate.sum()),
        filled_fraction=filled_frac,
        plate_height_cm=surf.plate_height_m * 100,
        mean_food_height_cm=float(hf.mean()) * 100,
        max_food_height_cm=float(hf.max()) * 100,
        tilt_deg=surf.tilt_deg,
        container=surf.container,
        concavity_cm=surf.concavity_m * 100,
        ok=True,
    )
