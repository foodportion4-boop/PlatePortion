"""Run both branches on the official test split and print one comparison table.

    PYTHONPATH=src python3 -m n5k.eval.compare --checkpoint runs/direct/best.pt

The geometric branch needs no checkpoint -- it fits its calibration constants on
the train split at run time. The direct branch needs a checkpoint from
``n5k.train``; omit ``--checkpoint`` to report the geometric branch alone.

Both branches are scored on the **same dish set** so the comparison is like for
like. The geometric branch flags dishes whose assumptions do not hold (bowls,
mostly-interpolated food regions); ``--geometric-reliable-only`` restricts to
those it trusts and reports the excluded fraction, which is a fairer read of the
method but no longer a like-for-like set. Both are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm as _tqdm


def tqdm(iterable, **kwargs):
    """tqdm that goes quiet when stderr is not a terminal.

    Without this a ``nohup``/redirected run writes one progress line per dish
    into the log, burying the results table under thousands of lines.
    """
    kwargs.setdefault("disable", not sys.stderr.isatty())
    kwargs.setdefault("mininterval", 1.0)
    return _tqdm(iterable, **kwargs)

from n5k.data.dataset import (
    Nutrition5kOverhead,
    collate,
    decode_depth_png,
    overhead_paths,
    usable_dish_ids,
)
from n5k.data.metadata import TARGETS, load_dishes
from n5k.data.splits import load_split
from n5k.eval.metrics import compute_metrics, format_table, write_predictions_csv
from n5k.geometric.calibrate import (
    MIN_INGREDIENT_COUNT,
    fit_calibration,
    fit_ingredient_calibration,
)
from n5k.geometric.volume import VolumeResult, estimate_volume

#: Fields of VolumeResult persisted in the volume cache. Recomputing volumes
#: takes ~20 minutes over the full splits; caching lets calibration variants be
#: compared in seconds without re-running the geometry.
_VOL_FIELDS = (
    "dish_id", "volume_cm3", "food_pixels", "filled_fraction", "plate_height_cm",
    "mean_food_height_cm", "max_food_height_cm", "tilt_deg", "container",
    "concavity_cm", "ok",
)


def _vol_fields(v: VolumeResult) -> dict:
    return {f: getattr(v, f) for f in _VOL_FIELDS}


class _VolStub(VolumeResult):
    """VolumeResult rehydrated from the cache (same fields, same .reliable)."""


def compute_volumes(data_root: Path, dish_ids: list[str], desc: str) -> dict:
    """Run the geometric pipeline over a list of dishes."""
    out = {}
    for did in tqdm(dish_ids, desc=desc, unit="dish"):
        depth_m, valid = decode_depth_png(overhead_paths(data_root, did)[1])
        out[did] = estimate_volume(depth_m, valid, did)
    return out


def run_geometric(data_root: Path, dishes, train_ids, test_ids, cache: Path | None = None):
    if cache and cache.exists():
        blob = json.loads(cache.read_text())
        train_vols = {k: _VolStub(**v) for k, v in blob["train"].items()}
        test_vols = {k: _VolStub(**v) for k, v in blob["test"].items()}
        print(f"loaded cached volumes from {cache}")
    else:
        train_vols = compute_volumes(data_root, train_ids, "geometric/train")
        test_vols = compute_volumes(data_root, test_ids, "geometric/test")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "train": {k: _vol_fields(v) for k, v in train_vols.items()},
                "test": {k: _vol_fields(v) for k, v in test_vols.items()},
            }))
            print(f"cached volumes to {cache}")

    vol_map = {k: v.volume_cm3 for k, v in train_vols.items()}
    calib = fit_calibration(vol_map, dishes, train_ids)
    print(
        f"\ncalibration (fitted on {calib.n_train} train dishes): "
        f"density={calib.density_g_per_cm3:.3f} g/cm3, "
        f"{calib.kcal_per_g:.2f} kcal/g"
    )
    preds = {d: calib.predict(r.volume_cm3) for d, r in test_vols.items()}

    # Oracle variant: same geometry, but density and composition conditioned on
    # which ingredients are present. Bounds what a perfect classifier would buy.
    ing_calib = fit_ingredient_calibration(vol_map, dishes, train_ids)
    print(
        f"ingredient calibration: {len(ing_calib.ingredients)} ingredients with "
        f">={MIN_INGREDIENT_COUNT} train occurrences, fitted on {ing_calib.n_train} dishes"
    )
    ing_preds = {
        d: ing_calib.predict(r.volume_cm3, [i.name for i in dishes[d].ingredients])
        for d, r in test_vols.items()
        if d in dishes
    }
    return preds, ing_preds, test_vols, calib, ing_calib


@torch.no_grad()
def predict_ingredients(checkpoint: Path, data_root: Path) -> dict[str, list[str]]:
    """Predict the ingredient set for every test dish.

    This is what turns the ``geometric+ingr`` oracle into a deployable number:
    the calibration takes the same interface either way, so swapping predicted
    names for ground-truth ones isolates exactly what the classifier costs.
    """
    from n5k.models.direct import pick_device
    from n5k.models.ingredients import IngredientClassifier

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab = ckpt["vocab"]
    threshold = ckpt.get("threshold", 0.5)
    device = pick_device()

    model = IngredientClassifier(len(vocab), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    ds = Nutrition5kOverhead(
        data_root, "test", "depth", ckpt.get("args", {}).get("image_size", 224),
        augment=False, ingredient_vocab=vocab,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate)

    out: dict[str, list[str]] = {}
    for batch in tqdm(loader, desc="ingredients/test", unit="batch"):
        probs = torch.sigmoid(
            model(batch["rgb"].to(device), batch["depth"].to(device),
                  batch["valid"].to(device))
        ).cpu().numpy()
        for did, row in zip(batch["dish_id"], probs):
            picked = [vocab[j] for j in range(len(vocab)) if row[j] >= threshold]
            if not picked:
                # Never hand back an empty set -- the calibration would fall
                # back to global constants for a dish the model does have an
                # opinion about. Take its single most confident ingredient.
                picked = [vocab[int(row.argmax())]]
            out[did] = picked
    print(f"predicted ingredients at threshold {threshold:.2f}: "
          f"{sum(len(v) for v in out.values()) / max(len(out), 1):.1f} per dish")
    return out


@torch.no_grad()
def run_direct(checkpoint: Path, data_root: Path, dish_subset: set[str] | None):
    from n5k.models.direct import DirectNutritionNet, TargetScaler, pick_device

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = pick_device()
    saved = ckpt.get("args", {})
    model = DirectNutritionNet(backbone=saved.get("backbone", "resnet18"), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    scaler = TargetScaler.from_state_dict(ckpt["scaler"]).to(device)

    ds = Nutrition5kOverhead(
        data_root, "test", "depth", saved.get("image_size", 224), augment=False
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=16, collate_fn=collate)
    preds = {}
    for batch in tqdm(loader, desc="direct/test", unit="batch"):
        out = model(
            batch["rgb"].to(device), batch["depth"].to(device), batch["valid"].to(device)
        )
        for did, row in zip(batch["dish_id"], scaler.decode(out).cpu()):
            if dish_subset is None or did in dish_subset:
                preds[did] = {t: float(v) for t, v in zip(TARGETS, row)}
    return preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="direct regression checkpoint from n5k.train")
    ap.add_argument("--ingredient-checkpoint", type=Path, default=None,
                    help="classifier from n5k.train_ingredients; adds the "
                         "non-oracle 'geometric+predicted' row")
    ap.add_argument("--out", type=Path, default=Path("runs/compare"))
    ap.add_argument("--limit", type=int, default=None, help="cap dishes per split (smoke test)")
    ap.add_argument(
        "--volume-cache",
        type=Path,
        default=Path("runs/volume_cache.json"),
        help="reuse geometry across calibration variants (~20 min to recompute)",
    )
    ap.add_argument("--no-cache", action="store_true", help="ignore the volume cache")
    args = ap.parse_args(argv)
    if args.no_cache or args.limit:
        args.volume_cache = None  # a limited run must not poison the full cache
    args.out.mkdir(parents=True, exist_ok=True)

    dishes = load_dishes(args.data_root / "metadata", cafes=("cafe1",))
    splits = args.data_root / "dish_ids/splits"

    def avail(ids: list[str]) -> list[str]:
        labelled = [d for d in ids if d in dishes]
        return usable_dish_ids(args.data_root, labelled)

    train_ids = avail(load_split(splits, "depth", "train"))
    test_ids = avail(load_split(splits, "depth", "test"))
    if args.limit:
        train_ids, test_ids = train_ids[: args.limit], test_ids[: args.limit]
    print(f"usable dishes on disk -- train: {len(train_ids)}, test: {len(test_ids)}")
    if not test_ids:
        raise SystemExit("no test imagery on disk; run n5k.data.download first")

    geo_preds, ing_preds, geo_vols, calib, ing_calib = run_geometric(
        args.data_root, dishes, train_ids, test_ids, cache=args.volume_cache
    )
    groundtruth = {d: {t: getattr(dishes[d], t) for t in TARGETS} for d in test_ids}

    results = {}
    common = set(geo_preds) & set(ing_preds)
    if args.checkpoint:
        direct_preds = run_direct(args.checkpoint, args.data_root, None)
        common &= set(direct_preds)
        results["direct (RGB-D)"] = compute_metrics(
            {d: direct_preds[d] for d in common}, groundtruth
        )
    results["geometric"] = compute_metrics({d: geo_preds[d] for d in common}, groundtruth)

    if args.ingredient_checkpoint:
        predicted = predict_ingredients(args.ingredient_checkpoint, args.data_root)
        pred_ing_preds = {
            d: ing_calib.predict(geo_vols[d].volume_cm3, predicted.get(d, []))
            for d in common
        }
        results["geometric+predicted"] = compute_metrics(
            {d: pred_ing_preds[d] for d in common}, groundtruth
        )

    results["geometric+ingr*"] = compute_metrics(
        {d: ing_preds[d] for d in common}, groundtruth
    )

    print("\n" + "=" * 100)
    print("MATCHED DISH SET -- identical dishes for every column")
    print("=" * 100)
    print(format_table(results))

    reliable = {d for d in common if geo_vols[d].reliable}
    if reliable and len(reliable) < len(common):
        pct = 100 * len(reliable) / len(common)
        print("\n" + "=" * 100)
        print(
            f"GEOMETRIC, RELIABLE SUBSET ONLY -- {len(reliable)}/{len(common)} "
            f"dishes ({pct:.0f}%); excludes bowls and heavily interpolated regions"
        )
        print("=" * 100)
        sub = {
            "geometric (reliable)": compute_metrics(
                {d: geo_preds[d] for d in reliable}, groundtruth
            ),
        }
        if args.ingredient_checkpoint:
            sub["geometric+predicted"] = compute_metrics(
                {d: pred_ing_preds[d] for d in reliable}, groundtruth
            )
        sub["geometric+ingr*"] = compute_metrics(
            {d: ing_preds[d] for d in reliable}, groundtruth
        )
        if args.checkpoint:
            sub["direct (same subset)"] = compute_metrics(
                {d: direct_preds[d] for d in reliable}, groundtruth
            )
        print(format_table(sub))

    containers = {}
    for d in common:
        containers[geo_vols[d].container] = containers.get(geo_vols[d].container, 0) + 1
    vols = np.array([geo_vols[d].volume_cm3 for d in common])
    masses = np.array([dishes[d].mass for d in common])
    ok = vols > 0
    print(
        f"\ngeometric diagnostics: containers={containers}  "
        f"volume-vs-mass pearson r={np.corrcoef(masses[ok], vols[ok])[0, 1]:.3f}"
    )

    print(
        "\n* geometric+ingr is an ORACLE: it uses ground-truth ingredient "
        "identity (never quantities) at inference, so it bounds what a perfect\n"
        "  ingredient classifier would buy. It is not a deployable predictor."
    )

    write_predictions_csv(args.out / "geometric_predictions.csv", geo_preds)
    write_predictions_csv(args.out / "geometric_ingr_predictions.csv", ing_preds)
    calib.save(args.out / "calibration.json")
    ing_calib.save(args.out / "ingredient_calibration.json")
    (args.out / "results.json").write_text(json.dumps(results, indent=2))

    table = ing_calib.density_table()
    ranked = sorted(table.items(), key=lambda kv: kv[1])
    print("\nfitted densities (g/cm3), lightest and densest:")
    for name, rho in ranked[:5]:
        print(f"    {rho:5.2f}  {name}")
    print("    ...")
    for name, rho in ranked[-5:]:
        print(f"    {rho:5.2f}  {name}")

    print(f"\nwrote {args.out}/results.json, predictions, calibration.json, "
          f"ingredient_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
