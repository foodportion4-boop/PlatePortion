"""Tests for the geometric branch, on synthetic scenes with known volume.

Synthetic rather than real captures: with a made-up scene the true volume is
known in closed form, so these check the integration is actually correct rather
than merely stable.
"""

from __future__ import annotations

import numpy as np
import pytest

from n5k.geometric.calibrate import Calibration, fit_calibration
from n5k.geometric.surfaces import estimate_surfaces
from n5k.geometric.volume import CameraModel, estimate_volume
from n5k.data.metadata import Dish, Ingredient

H = W = 240
TABLE_Z = 0.41
PLATE_Z = 0.355


def make_scene(food_h_m: float = 0.0, food_r: int = 30, bowl: bool = False):
    """Overhead scene: table, a square plate, optional food disc on the plate."""
    depth = np.full((H, W), TABLE_Z, np.float32)
    cy = cx = H // 2
    plate = slice(cy - 70, cy + 70), slice(cx - 70, cx + 70)
    depth[plate] = PLATE_Z

    if bowl:
        # Concave interior: centre sits well below the rim.
        yy, xx = np.mgrid[0:H, 0:W]
        r = np.hypot(yy - cy, xx - cx)
        inside = r < 60
        depth[inside] = PLATE_Z + 0.03 * (1 - (r[inside] / 60) ** 2)

    if food_h_m > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        disc = np.hypot(yy - cy, xx - cx) < food_r
        depth[disc] = PLATE_Z - food_h_m

    return depth, np.ones((H, W), bool)


def test_finds_table_and_plate_surfaces():
    depth, valid = make_scene(food_h_m=0.02)
    s = estimate_surfaces(depth, valid)
    assert s.ok
    assert s.plate_mask.sum() > 1000
    assert float(np.median(s.table_plane)) == pytest.approx(TABLE_Z, abs=0.005)
    assert float(np.median(s.plate_plane)) == pytest.approx(PLATE_Z, abs=0.005)
    assert s.plate_height_m == pytest.approx(TABLE_Z - PLATE_Z, abs=0.005)


def test_empty_plate_has_near_zero_volume():
    """The rim must not be integrated as food -- this was a real bug."""
    depth, valid = make_scene(food_h_m=0.0)
    r = estimate_volume(depth, valid, "empty")
    assert r.ok
    assert r.volume_cm3 < 5.0, f"empty plate reported {r.volume_cm3:.1f} cm3"


def test_volume_matches_analytic_cylinder():
    """A disc of known radius and height integrates to its true volume."""
    food_h, food_r = 0.02, 30
    depth, valid = make_scene(food_h_m=food_h, food_r=food_r)
    cam = CameraModel(fx=615.0, fy=615.0)
    r = estimate_volume(depth, valid, "cyl", camera=cam)

    px_m = PLATE_Z / cam.fx                      # metres per pixel at food depth
    expected_cm3 = (np.pi * (food_r * px_m) ** 2 * food_h) * 1e6
    assert r.volume_cm3 == pytest.approx(expected_cm3, rel=0.15)


def test_volume_scales_with_food_height():
    cam = CameraModel()
    v1 = estimate_volume(*make_scene(food_h_m=0.01), "a", camera=cam).volume_cm3
    v2 = estimate_volume(*make_scene(food_h_m=0.02), "b", camera=cam).volume_cm3
    assert v2 == pytest.approx(2 * v1, rel=0.2)


def test_bowl_is_flagged_not_silently_measured():
    depth, valid = make_scene(food_h_m=0.0, bowl=True)
    r = estimate_volume(depth, valid, "bowl")
    assert r.container == "bowl" or not r.reliable, (
        f"concave container passed as reliable (container={r.container}, "
        f"concavity={r.concavity_cm:.2f} cm)"
    )


def test_enclosed_dropouts_are_filled_not_dropped():
    """A dropout inside a food blob must be interpolated, not silently skipped.

    Skipping it would under-count volume exactly where food is most specular --
    the systematic bias this fill exists to prevent.
    """
    depth, valid = make_scene(food_h_m=0.02, food_r=30)
    valid[114:126, 114:126] = False   # 12x12 hole, comfortably inside r=30 disc
    depth[114:126, 114:126] = 0.0

    r = estimate_volume(depth, valid, "holes")
    assert r.ok
    assert r.filled_fraction > 0, "enclosed dropout was not detected as filled"

    full = estimate_volume(*make_scene(food_h_m=0.02, food_r=30), "full").volume_cm3
    assert r.volume_cm3 == pytest.approx(full, rel=0.15)


def test_dropout_spanning_the_whole_blob_is_reported_as_lost():
    """Known limitation: a dropout wide enough to sever the food blob cannot be
    filled -- the region stops being enclosed. It must under-count rather than
    invent geometry, and the shortfall must be visible to the caller."""
    depth, valid = make_scene(food_h_m=0.02, food_r=30)
    valid[100:140, 100:140] = False   # nearly as wide as the disc itself
    depth[100:140, 100:140] = 0.0

    r = estimate_volume(depth, valid, "severed")
    full = estimate_volume(*make_scene(food_h_m=0.02, food_r=30), "full").volume_cm3
    assert r.volume_cm3 < full          # under-counts, never over-counts
    assert r.food_pixels < 0.6 * (np.pi * 30**2)


def test_calibration_recovers_known_density():
    dishes, volumes, ids = {}, {}, []
    for i in range(50):
        vol = 100.0 + i
        dishes[f"d{i}"] = Dish(f"d{i}", calories=vol * 0.8 * 1.5, mass=vol * 0.8,
                               fat=vol * 0.8 * 0.05, carb=vol * 0.8 * 0.2,
                               protein=vol * 0.8 * 0.1)
        volumes[f"d{i}"] = vol
        ids.append(f"d{i}")
    calib = fit_calibration(volumes, dishes, ids)
    assert calib.density_g_per_cm3 == pytest.approx(0.8, rel=0.02)
    assert calib.kcal_per_g == pytest.approx(1.5, rel=0.02)

    pred = calib.predict(200.0)
    assert pred["mass"] == pytest.approx(160.0, rel=0.02)
    assert pred["calories"] == pytest.approx(240.0, rel=0.03)


def _mixed_dishes(n=400):
    """Synthetic dishes from two ingredients with very different densities."""
    import random

    rng = random.Random(0)
    dishes, volumes, ids = {}, {}, []
    truth = {"lettuce": 0.2, "chicken": 1.0}
    for i in range(n):
        name = "lettuce" if i % 2 == 0 else "chicken"
        rho = truth[name]
        vol = rng.uniform(80, 400)
        mass = vol * rho
        kcal_g = 0.15 if name == "lettuce" else 1.65
        dishes[f"d{i}"] = Dish(
            f"d{i}", calories=mass * kcal_g, mass=mass, fat=mass * 0.02,
            carb=mass * 0.05, protein=mass * 0.2,
            ingredients=[Ingredient(f"ingr_{i}", name, mass, mass * kcal_g, 0, 0, 0)],
        )
        volumes[f"d{i}"] = vol
        ids.append(f"d{i}")
    return dishes, volumes, ids, truth


def test_ingredient_calibration_recovers_per_ingredient_densities():
    from n5k.geometric.calibrate import fit_ingredient_calibration

    dishes, volumes, ids, truth = _mixed_dishes()
    calib = fit_ingredient_calibration(volumes, dishes, ids, alpha=0.01)

    table = calib.density_table()
    for name, expected in truth.items():
        assert table[name] == pytest.approx(expected, abs=0.12), (
            f"{name}: fitted {table[name]:.2f}, expected {expected:.2f}"
        )

    # A lettuce dish and a chicken dish of equal volume must differ in mass.
    light = calib.predict(200.0, ["lettuce"])
    heavy = calib.predict(200.0, ["chicken"])
    assert heavy["mass"] > 3 * light["mass"]
    assert heavy["calories"] > light["calories"]


def test_ingredient_calibration_beats_global_density_on_mixed_foods():
    from n5k.geometric.calibrate import fit_calibration, fit_ingredient_calibration

    dishes, volumes, ids, _ = _mixed_dishes()
    glob = fit_calibration(volumes, dishes, ids)
    ing = fit_ingredient_calibration(volumes, dishes, ids, alpha=0.01)

    def mae(predict):
        return sum(abs(predict(d) - dishes[d].mass) for d in ids) / len(ids)

    glob_mae = mae(lambda d: glob.predict(volumes[d])["mass"])
    ing_mae = mae(
        lambda d: ing.predict(volumes[d], [i.name for i in dishes[d].ingredients])["mass"]
    )
    assert ing_mae < 0.5 * glob_mae, f"ingredient {ing_mae:.1f} vs global {glob_mae:.1f}"


def test_ingredient_calibration_falls_back_on_unknown_ingredients():
    from n5k.geometric.calibrate import fit_ingredient_calibration

    dishes, volumes, ids, _ = _mixed_dishes()
    calib = fit_ingredient_calibration(volumes, dishes, ids, alpha=0.01)
    pred = calib.predict(200.0, ["something never seen"])
    fallback = calib.fallback.predict(200.0)
    assert pred["mass"] == pytest.approx(fallback["mass"])


def test_ingredient_calibration_never_returns_negative_nutrition():
    from n5k.geometric.calibrate import fit_ingredient_calibration

    dishes, volumes, ids, _ = _mixed_dishes()
    calib = fit_ingredient_calibration(volumes, dishes, ids, alpha=0.01)
    for names in (["lettuce"], ["chicken"], ["lettuce", "chicken"], []):
        pred = calib.predict(150.0, names)
        assert all(v >= 0 for v in pred.values()), (names, pred)
    assert calib.predict(0.0, ["chicken"])["mass"] == pytest.approx(0.0)


def test_ingredient_calibration_roundtrips_through_disk(tmp_path):
    from n5k.geometric.calibrate import IngredientCalibration, fit_ingredient_calibration

    dishes, volumes, ids, _ = _mixed_dishes()
    calib = fit_ingredient_calibration(volumes, dishes, ids, alpha=0.01)
    calib.save(tmp_path / "ing.json")
    back = IngredientCalibration.load(tmp_path / "ing.json")
    assert back.ingredients == calib.ingredients
    assert back.predict(200.0, ["chicken"])["mass"] == pytest.approx(
        calib.predict(200.0, ["chicken"])["mass"]
    )


def test_calibration_roundtrips_through_disk(tmp_path):
    c = Calibration(0.7, 1.4, 0.05, 0.2, 0.1, n_train=99, notes="x")
    c.save(tmp_path / "calib.json")
    back = Calibration.load(tmp_path / "calib.json")
    assert back.density_g_per_cm3 == pytest.approx(0.7)
    assert back.n_train == 99
