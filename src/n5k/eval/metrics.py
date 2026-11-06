"""Evaluation metrics, matching the official Nutrition5k eval script.

Reimplemented from ``scripts/compute_eval_statistics.py`` in the dataset
release so results are directly comparable to the CVPR 2021 paper.

Two details are easy to get wrong, and both are taken from the script's *code*
rather than its docstring:

1. **Field order is** ``dish_id, calories, mass, fat, carb, protein``. The
   script's own docstring omits ``fat`` and lists only four fields; its
   ``DATA_FIELDNAMES`` constant includes it. The constant is authoritative and
   matches the metadata CSV column order.

2. **MAE%% is normalised by the mean of ground truth over the set**, i.e.

       MAE_pct = 100 * mean(|pred - gt|) / mean(gt)

   *not* the mean of per-sample percentage errors ``mean(|pred-gt|/gt)``. The
   two differ substantially on a long-tailed target like calories, and the
   per-sample form explodes on near-zero dishes. Reimplementations routinely
   use the wrong one and report numbers that look better than the paper's.

Published baselines are in :data:`BASELINES`, transcribed from Han et al.,
"DPF-Nutrition" (arXiv:2310.11702), Table 2, which restates both Thames et al.
(CVPR 2021) variants under matched splits and preprocessing. Sourced from that
comparison rather than quoted from memory -- the CVPR PDF is not directly
fetchable, and the two Thames rows are easy to conflate.

Compare against ``google_nutrition_depth``: it is the RGB-D row, and this repo's
models consume RGB-D. Treat it as a target to approach rather than beat -- it
comes from a far larger training budget than an 8 GB laptop run.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: Official field order. Do not reorder -- the CSV contract depends on it.
FIELDS = ("calories", "mass", "fat", "carb", "protein")

#: Published test-set baselines as (MAE, PMAE%) per field.
#: Source: Han et al., DPF-Nutrition, arXiv:2310.11702, Table 2.
BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    # Thames et al. CVPR 2021, RGB-D variant -- the row to compare against.
    "google_nutrition_depth": {
        "calories": (47.6, 18.8),
        "mass": (40.7, 18.9),
        "fat": (2.27, 18.1),
        "carb": (4.6, 23.8),
        "protein": (3.7, 20.9),
    },
    # Thames et al. CVPR 2021, monocular RGB variant.
    "google_nutrition_monocular": {
        "calories": (70.6, 26.1),
        "mass": (40.4, 18.8),
        "fat": (5.0, 34.2),
        "carb": (6.1, 31.9),
        "protein": (5.5, 29.5),
    },
    # Depth-only ablation (DPF-Nutrition Table 3b). Useful context for how far
    # geometry alone can go without RGB: even a trained depth CNN lands here.
    "depth_stream_only": {
        "calories": (83.5, 32.5),
        "mass": (44.7, 22.3),
        "fat": (6.29, 48.8),
        "carb": (8.08, 40.8),
        "protein": (8.53, 48.5),
    },
}

#: Default comparison column.
PAPER_BASELINE = BASELINES["google_nutrition_depth"]


def compute_metrics(
    predictions: dict[str, dict[str, float]],
    groundtruth: dict[str, dict[str, float]],
) -> dict[str, float]:
    """MAE and MAE%% per field, over the dishes present in ``predictions``.

    Mirrors the official script: it iterates over predictions and looks each up
    in ground truth, so a prediction with no matching label is an error rather
    than a silent skip.
    """
    missing = [d for d in predictions if d not in groundtruth]
    if missing:
        raise KeyError(
            f"{len(missing)} predicted dishes have no ground truth, "
            f"e.g. {missing[:5]}"
        )
    if not predictions:
        raise ValueError("no predictions to evaluate")

    out: dict[str, float] = {}
    for field in FIELDS:
        errs, gts = [], []
        for dish_id, pred in predictions.items():
            gt = groundtruth[dish_id]
            errs.append(abs(float(pred[field]) - float(gt[field])))
            gts.append(float(gt[field]))
        mae = float(np.mean(errs))
        mean_gt = float(np.mean(gts))
        out[f"{field}_MAE"] = mae
        # Guard the degenerate case the official script would divide by zero on.
        out[f"{field}_MAE_%"] = 100.0 * mae / mean_gt if mean_gt > 0 else float("nan")
    out["n_dishes"] = len(predictions)
    return out


def write_predictions_csv(
    path: str | Path, predictions: dict[str, dict[str, float]]
) -> None:
    """Write predictions in the exact format the official script expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        for dish_id in sorted(predictions):
            p = predictions[dish_id]
            writer.writerow([dish_id] + [f"{float(p[f]):.6f}" for f in FIELDS])


def format_table(
    results: dict[str, dict[str, float]],
    baseline: str | None = "google_nutrition_depth",
) -> str:
    """Render named result sets as a text comparison table, MAE (MAE%) per cell."""
    names = list(results)
    cols = list(names) + ([baseline] if baseline else [])
    width = max(18, max(len(c) for c in cols) + 1)

    header = f"{'field':>9} | " + " | ".join(f"{c:>{width}}" for c in cols)
    lines = [header, "-" * len(header)]
    for field in FIELDS:
        cells = [
            f"{results[n][f'{field}_MAE']:7.1f} ({results[n][f'{field}_MAE_%']:5.1f}%)"
            for n in names
        ]
        if baseline:
            mae, pct = BASELINES[baseline][field]
            cells.append(f"{mae:7.1f} ({pct:5.1f}%)")
        lines.append(f"{field:>9} | " + " | ".join(f"{c:>{width}}" for c in cells))

    means = [
        f"{np.mean([results[n][f'{f}_MAE_%'] for f in FIELDS]):>{width - 8}.1f}% mean"
        for n in names
    ]
    if baseline:
        means.append(
            f"{np.mean([BASELINES[baseline][f][1] for f in FIELDS]):>{width - 8}.1f}% mean"
        )
    lines.append(f"{'PMAE':>9} | " + " | ".join(f"{m:>{width}}" for m in means))

    counts = ", ".join(f"{n}: n={int(results[n]['n_dishes'])}" for n in names)
    lines.append(f"\ndishes evaluated -- {counts}")
    if baseline:
        lines.append(
            f"baseline '{baseline}' from DPF-Nutrition (arXiv:2310.11702) Table 2, "
            f"evaluated on the full official test split."
        )
    return "\n".join(lines)
