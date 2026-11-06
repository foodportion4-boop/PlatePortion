"""Recover the two reference surfaces in an overhead Nutrition5k capture.

The scene has three stacked surfaces, seen from ~41 cm above:

    table / tray   ~0.41 m from camera   (farthest)
    plate surface  ~0.35 m               (plate stands ~5-7 cm proud of table)
    food           ~0.29-0.34 m          (nearest)

Measuring food volume against the **table** would count the entire plate as
food. The plate surface is the correct datum, so we estimate both:

1. Fit the table plane by RANSAC over the far pixels. A plane (not a scalar)
   because the tray is not perfectly perpendicular to the optical axis, and a
   1-2 degree tilt across 640 px is millimetres of systematic volume error.
2. Find the plate as the large central region standing proud of that plane, and
   take its own surface as a second plane fit.

A naive histogram mode of the depth image does *not* work for step 1: on dishes
where food covers most of the frame the mode lands on the food, not the table
(e.g. dish_1560442226 modes at 0.357 m against a true table of ~0.41 m). We use
a high percentile of the in-range depth to seed RANSAC instead, since the table
is by construction the farthest real surface in the scene.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from n5k.data.dataset import DEPTH_FAR_M, DEPTH_NEAR_M


@dataclass
class Surfaces:
    """Geometry recovered from one depth frame. All depths in metres."""

    table_plane: np.ndarray      # (H, W) fitted table depth per pixel
    plate_plane: np.ndarray      # (H, W) fitted plate-surface depth per pixel
    plate_mask: np.ndarray       # (H, W) bool, the container region
    interior_mask: np.ndarray    # (H, W) bool, container minus its rim
    plate_height_m: float        # median container stand-off above the table
    tilt_deg: float              # table plane tilt from the image plane
    concavity_m: float           # rim-to-centre drop; > 0 means a bowl
    container: str               # "plate" | "bowl" | "unknown"
    ok: bool                     # False if no container could be located


def _fit_plane(
    depth: np.ndarray, mask: np.ndarray, iters: int = 60, tol: float = 0.004,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC-fit ``z = a*u + b*v + c`` over ``mask`` pixels.

    Returns (plane_depth_full_image, inlier_mask). Falls back to a constant
    plane at the median depth when there are too few points to fit.
    """
    rng = rng or np.random.default_rng(0)
    h, w = depth.shape
    vv, uu = np.mgrid[0:h, 0:w]
    pts_u, pts_v, pts_z = uu[mask], vv[mask], depth[mask]
    n = len(pts_z)

    if n < 50:
        const = float(np.median(pts_z)) if n else float(np.median(depth))
        return np.full((h, w), const, np.float32), np.zeros_like(mask)

    A_all = np.stack([pts_u, pts_v, np.ones(n)], axis=1).astype(np.float64)
    best_inliers, best_coef = None, None
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        A, z = A_all[idx], pts_z[idx]
        try:
            coef = np.linalg.solve(A, z)
        except np.linalg.LinAlgError:
            continue  # degenerate (collinear) sample
        inliers = np.abs(A_all @ coef - pts_z) < tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_coef = inliers, coef

    # Refit least-squares on the consensus set for a lower-variance estimate.
    if best_inliers is not None and best_inliers.sum() >= 3:
        coef, *_ = np.linalg.lstsq(A_all[best_inliers], pts_z[best_inliers], rcond=None)
    else:
        coef = best_coef if best_coef is not None else np.array(
            [0.0, 0.0, float(np.median(pts_z))]
        )

    plane = (coef[0] * uu + coef[1] * vv + coef[2]).astype(np.float32)
    full_inliers = np.zeros_like(mask)
    if best_inliers is not None:
        full_inliers[mask] = best_inliers
    return plane, full_inliers


def estimate_surfaces(
    depth_m: np.ndarray,
    valid: np.ndarray,
    plate_min_height_m: float = 0.015,
    plate_max_height_m: float = 0.11,
    min_plate_frac: float = 0.03,
    floor_method: str = "low_percentile",
    floor_percentile: float = 15.0,
    rng: np.random.Generator | None = None,
) -> Surfaces:
    """Estimate the table plane and container floor for one frame.

    Args:
        floor_method: ``"low_percentile"`` (default) or ``"mode"``. See the
            note at step 4 -- the mode fails on heavily loaded plates.
        floor_percentile: percentile of interior height used as the floor when
            ``floor_method="low_percentile"``.
    """
    h, w = depth_m.shape
    in_range = valid & (depth_m > DEPTH_NEAR_M) & (depth_m < DEPTH_FAR_M)

    def _fail(table=None, mask=None, tilt=0.0) -> Surfaces:
        flat = np.full((h, w), DEPTH_FAR_M, np.float32)
        empty = np.zeros((h, w), bool)
        return Surfaces(
            table_plane=flat if table is None else table,
            plate_plane=flat,
            plate_mask=empty if mask is None else mask,
            interior_mask=empty,
            plate_height_m=0.0,
            tilt_deg=tilt,
            concavity_m=0.0,
            container="unknown",
            ok=False,
        )

    if in_range.sum() < 500:
        return _fail()

    # --- 1. table plane -------------------------------------------------
    # Seed on the far tail: the table is the farthest real surface, so pixels
    # beyond the 75th percentile are overwhelmingly table.
    far_thresh = np.percentile(depth_m[in_range], 75)
    table_seed = in_range & (depth_m >= far_thresh)
    table_plane, _ = _fit_plane(depth_m, table_seed, rng=rng)

    # Plane is z = a*u + b*v + c with a, b in metres/pixel; its normal is
    # (-a, -b, 1), so the tilt off the image plane is atan(hypot(a, b)).
    grad_u = float(table_plane[0, 1] - table_plane[0, 0])
    grad_v = float(table_plane[1, 0] - table_plane[0, 0])
    tilt_deg = float(np.degrees(np.arctan(np.hypot(grad_u, grad_v))))

    # --- 2. plate region ------------------------------------------------
    height_above_table = np.where(in_range, table_plane - depth_m, 0.0)
    candidate = (
        in_range
        & (height_above_table > plate_min_height_m)
        & (height_above_table < plate_max_height_m)
    )
    # Close over dropout speckle so the plate does not fragment into pieces.
    candidate = ndimage.binary_closing(candidate, structure=np.ones((7, 7)))
    candidate = ndimage.binary_fill_holes(candidate)

    labels, n_lab = ndimage.label(candidate)
    if n_lab == 0:
        return _fail(table_plane, tilt=tilt_deg)

    # The plate is the largest blob overlapping the image centre; scoring by
    # size alone can latch onto a bright tray edge running down the frame.
    cy, cx = h // 2, w // 2
    centre = np.zeros((h, w), bool)
    centre[cy - h // 6 : cy + h // 6, cx - w // 6 : cx + w // 6] = True
    sizes = ndimage.sum(candidate, labels, index=np.arange(1, n_lab + 1))
    centre_hits = ndimage.sum(centre, labels, index=np.arange(1, n_lab + 1))
    score = np.where(centre_hits > 0, sizes, sizes * 0.05)
    plate_label = int(np.argmax(score)) + 1
    plate_mask = labels == plate_label

    if plate_mask.sum() < min_plate_frac * h * w:
        return _fail(table_plane, plate_mask, tilt_deg)

    # --- 3. strip the rim ------------------------------------------------
    # Every container here has a raised rim. Including it makes the rim ring
    # read as "above the floor plane" and it gets integrated as food on every
    # single dish -- a large systematic over-count. Erode inward proportionally
    # to the container's own size so it works for both a dinner plate and a
    # small ramekin.
    radius_px = np.sqrt(plate_mask.sum() / np.pi)
    erode_px = int(np.clip(round(0.16 * radius_px), 4, 40))
    interior = ndimage.binary_erosion(
        plate_mask, structure=np.ones((3, 3)), iterations=erode_px
    )
    if interior.sum() < 200:
        interior = plate_mask.copy()

    # --- 4. floor plane from the interior only --------------------------
    # The container floor is the *farthest* surface inside the interior: food
    # only ever adds height, never removes it. So a low percentile of "height
    # above table" recovers the floor, while the modal height does not -- on a
    # full plate the mode is the food surface, and measuring food relative to
    # its own top collapses the volume toward zero.
    #
    # Swept on 299 train dishes (scripts/tune_floor.py), Spearman rho of volume
    # against ground-truth mass:
    #     mode  0.270 | p5  0.750 | p15 0.726 | p30 0.620 | p50 0.410
    # Monotonic decay with percentile is exactly the predicted failure. The
    # optimum is flat over p1-p7 (rho 0.748-0.755); p5 is chosen over p1 as the
    # less noise-sensitive point on that plateau.
    h_table_int = (table_plane - depth_m)[interior & in_range]
    if h_table_int.size < 100:
        return _fail(table_plane, plate_mask, tilt_deg)
    if floor_method == "mode":
        hist, edges = np.histogram(h_table_int, bins=120)
        floor_h = float(edges[int(np.argmax(hist))] + (edges[1] - edges[0]) / 2)
    else:
        floor_h = float(np.percentile(h_table_int, floor_percentile))
    floor_seed = (
        interior
        & in_range
        & (np.abs((table_plane - depth_m) - floor_h) < 0.004)
    )
    if floor_seed.sum() < 50:
        floor_seed = interior & in_range
    plate_plane, _ = _fit_plane(depth_m, floor_seed, rng=rng)

    plate_height_m = float(np.median(table_plane[interior] - plate_plane[interior]))

    # --- 5. plate vs bowl -------------------------------------------------
    # A planar datum is only valid for a flat plate. In a bowl the floor sits
    # well below the rim, so every empty pixel of the cavity integrates as
    # food. Detect it by comparing the container's centre against its edge and
    # let callers down-weight or drop these rather than emit a wrong number.
    dist = ndimage.distance_transform_edt(plate_mask)
    core = plate_mask & (dist > 0.65 * dist.max())
    ring = interior & (dist < 0.35 * dist.max())
    if core.sum() > 50 and ring.sum() > 50:
        core_depth = float(np.median(depth_m[core & in_range])) if (core & in_range).any() else np.nan
        ring_depth = float(np.median(depth_m[ring & in_range])) if (ring & in_range).any() else np.nan
        concavity_m = float(core_depth - ring_depth) if np.isfinite([core_depth, ring_depth]).all() else 0.0
    else:
        concavity_m = 0.0
    container = "bowl" if concavity_m > 0.012 else "plate"

    return Surfaces(
        table_plane=table_plane,
        plate_plane=plate_plane,
        plate_mask=plate_mask,
        interior_mask=interior,
        plate_height_m=plate_height_m,
        tilt_deg=tilt_deg,
        concavity_m=concavity_m,
        container=container,
        ok=True,
    )
