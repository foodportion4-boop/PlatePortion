"""Isolate where ingredient knowledge helps the geometric branch.

Conditioning the geometric branch on ingredient identity improves it by ~10.7
points of mean PMAE. That gain could come from either half of the pipeline:

    volume --[density]--> mass --[composition]--> nutrition

This 2x2 ablation swaps each half between a single global constant and an
ingredient-conditioned model, holding the geometry fixed. The answer is not
what the project's own roadmap originally predicted -- see README.

Requires ``runs/volume_cache.json`` (written by ``n5k.eval.compare``), so the
geometry is computed once and every variant is scored in seconds:

    PYTHONPATH=src python3 scripts/ablate_calibration.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from n5k.data.metadata import TARGETS, load_dishes
from n5k.data.splits import load_split
from n5k.eval.metrics import FIELDS, compute_metrics
from n5k.geometric.calibrate import fit_calibration, fit_ingredient_calibration


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    ap.add_argument("--cache", type=Path, default=Path("runs/volume_cache.json"))
    args = ap.parse_args()

    if not args.cache.exists():
        raise SystemExit(
            f"{args.cache} not found. Run `python3 -m n5k.eval.compare` first "
            f"to compute and cache the geometry."
        )

    dishes = load_dishes(args.data_root / "metadata", cafes=("cafe1",))
    cache = json.loads(args.cache.read_text())
    train_vol = {k: v["volume_cm3"] for k, v in cache["train"].items()}
    test_vol = {k: v["volume_cm3"] for k, v in cache["test"].items()}
    splits = args.data_root / "dish_ids/splits"
    train_ids = [d for d in load_split(splits, "depth", "train") if d in train_vol and d in dishes]
    test_ids = [d for d in load_split(splits, "depth", "test") if d in test_vol and d in dishes]

    glob = fit_calibration(train_vol, dishes, train_ids)
    ing = fit_ingredient_calibration(train_vol, dishes, train_ids)
    groundtruth = {d: {t: getattr(dishes[d], t) for t in TARGETS} for d in test_ids}

    def names(d):
        return [i.name for i in dishes[d].ingredients]

    def global_comp(mass):
        return {
            "mass": mass,
            "calories": mass * glob.kcal_per_g,
            "fat": mass * glob.fat_per_g,
            "carb": mass * glob.carb_per_g,
            "protein": mass * glob.protein_per_g,
        }

    def ingr_ratios(d):
        """Per-gram composition implied by the ingredient model (mass divided out)."""
        p = ing.predict(test_vol[d], names(d))
        if p["mass"] <= 0:
            return None
        return {k: p[k] / p["mass"] for k in ("calories", "fat", "carb", "protein")}

    variants = {
        "global rho + global comp": lambda d: glob.predict(test_vol[d]),
        "ingr rho  + global comp": lambda d: global_comp(
            ing.predict(test_vol[d], names(d))["mass"]
        ),
        "global rho + ingr comp": lambda d: _apply(
            test_vol[d] * glob.density_g_per_cm3, ingr_ratios(d), glob, test_vol[d]
        ),
        "ingr rho  + ingr comp": lambda d: ing.predict(test_vol[d], names(d)),
    }

    print(f"{'variant':26} " + " ".join(f"{f:>9}" for f in FIELDS) + "     mean")
    print("-" * 96)
    rows = {}
    for label, fn in variants.items():
        metrics = compute_metrics({d: fn(d) for d in test_ids}, groundtruth)
        pcts = [metrics[f"{f}_MAE_%"] for f in FIELDS]
        rows[label] = pcts
        print(f"{label:26} " + " ".join(f"{p:8.1f}%" for p in pcts) + f"  {np.mean(pcts):7.1f}%")

    base = np.mean(rows["global rho + global comp"])
    d_rho = base - np.mean(rows["ingr rho  + global comp"])
    d_comp = base - np.mean(rows["global rho + ingr comp"])
    print(
        f"\nmean PMAE improvement over baseline:\n"
        f"  from ingredient DENSITY     {d_rho:+6.1f} points\n"
        f"  from ingredient COMPOSITION {d_comp:+6.1f} points\n"
        f"\nDensity conditioning is nearly free of effect: volume->mass is limited\n"
        f"by how well the geometry measures volume, not by knowing the density."
    )
    return 0


def _apply(mass, ratios, glob, volume):
    if ratios is None:
        return glob.predict(volume)
    return {"mass": mass, **{k: mass * v for k, v in ratios.items()}}


if __name__ == "__main__":
    raise SystemExit(main())
