"""Sweep the container-floor estimator for the geometric branch.

The floor datum is the single most consequential choice in the geometric
pipeline: get it wrong and volume either collapses toward zero (floor estimated
at the food surface) or absorbs the plate (floor estimated at the table).

This script sweeps ``floor_method`` / ``floor_percentile`` and reports, per
setting, the Spearman correlation between estimated volume and ground-truth
mass, plus the spread of implied density. **Correlation, not MAE**, because the
volume->mass constant is fitted afterwards -- a setting that ranks dishes well
is what we want, and any consistent scale error is absorbed by calibration.

Run on the TRAIN split only. Selecting this hyper-parameter on test would leak.

    PYTHONPATH=src python3 scripts/tune_floor.py --limit 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from n5k.data.dataset import decode_depth_png
from n5k.data.metadata import load_dishes
from n5k.data.splits import load_split
from n5k.geometric.volume import estimate_volume


def evaluate(dish_ids, dishes, data_root: Path, **kwargs):
    masses, volumes = [], []
    n_fail = 0
    for did in dish_ids:
        depth_path = data_root / "imagery/realsense_overhead" / did / "depth_raw.png"
        if not depth_path.exists():
            continue
        depth_m, valid = decode_depth_png(depth_path)
        res = estimate_volume(depth_m, valid, did, **kwargs)
        if not res.ok or res.volume_cm3 <= 0:
            n_fail += 1
            continue
        masses.append(dishes[did].mass)
        volumes.append(res.volume_cm3)
    if len(masses) < 10:
        return None
    m, v = np.array(masses), np.array(volumes)
    dens = m / v
    return {
        "n": len(m),
        "fail": n_fail,
        "pearson": float(np.corrcoef(m, v)[0, 1]),
        "spearman": float(stats.spearmanr(m, v).statistic),
        "dens_med": float(np.median(dens)),
        "dens_iqr": float(np.percentile(dens, 75) - np.percentile(dens, 25)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    dishes = load_dishes(args.data_root / "metadata", cafes=("cafe1",))
    train_ids = load_split(args.data_root / "dish_ids/splits", "depth", "train")
    ids = [d for d in train_ids if d in dishes][: args.limit]

    settings = [("mode", None)] + [("low_percentile", p) for p in (5, 10, 15, 20, 30, 40, 50)]

    print(f"{'method':>16} {'pct':>5} | {'n':>4} {'fail':>4} {'pearson':>8} "
          f"{'spearman':>9} {'dens_med':>9} {'dens_iqr':>9}")
    print("-" * 78)
    best = None
    for method, pct in settings:
        kw = {"floor_method": method}
        if pct is not None:
            kw["floor_percentile"] = pct
        r = evaluate(ids, dishes, args.data_root, **kw)
        if r is None:
            print(f"{method:>16} {str(pct):>5} | too few usable dishes")
            continue
        print(f"{method:>16} {str(pct):>5} | {r['n']:4d} {r['fail']:4d} "
              f"{r['pearson']:8.3f} {r['spearman']:9.3f} {r['dens_med']:9.3f} "
              f"{r['dens_iqr']:9.3f}")
        if best is None or r["spearman"] > best[1]["spearman"]:
            best = ((method, pct), r)

    if best:
        (method, pct), r = best
        print(f"\nBest by Spearman: floor_method={method} floor_percentile={pct} "
              f"(rho={r['spearman']:.3f}, implied density {r['dens_med']:.3f} g/cm3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
