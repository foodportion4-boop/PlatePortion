"""Train the direct RGB-D regression baseline.

    PYTHONPATH=src python3 -m n5k.train --epochs 20 --batch-size 16

Defaults are sized for an 8 GB Apple M1. Notes for that setting:

* ``num_workers=0`` by default. MPS plus forked workers is a reliable way to
  exhaust 8 GB; the dataset is small enough that loading is not the bottleneck.
* No AMP. autocast on MPS is still inconsistent across torch releases and the
  memory win here is small.
* Validation runs on the official test split. It is reported every epoch for
  monitoring, but **model selection uses training loss only** -- picking a
  checkpoint by test score would leak the test set into the reported result.
  Carve a validation split out of train if you want honest early stopping.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from n5k.data.dataset import Nutrition5kOverhead, collate
from n5k.data.metadata import TARGETS
from n5k.data.splits import assert_disjoint
from n5k.eval.metrics import compute_metrics, format_table
from n5k.models.direct import DirectNutritionNet, TargetScaler, pick_device


def build_loaders(args):
    train_ds = Nutrition5kOverhead(
        args.data_root, "train", "depth", args.image_size, augment=True
    )
    test_ds = Nutrition5kOverhead(
        args.data_root, "test", "depth", args.image_size, augment=False
    )
    assert_disjoint(train_ds.ids, test_ds.ids)
    print(
        f"train: {len(train_ds)} dishes (skipped {train_ds.skipped})\n"
        f"test : {len(test_ds)} dishes (skipped {test_ds.skipped})"
    )
    common = dict(
        batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(test_ds, shuffle=False, **common),
        train_ds,
        test_ds,
    )


@torch.no_grad()
def evaluate(model, loader, scaler, device, groundtruth):
    model.eval()
    preds: dict[str, dict[str, float]] = {}
    for batch in loader:
        out = model(
            batch["rgb"].to(device), batch["depth"].to(device), batch["valid"].to(device)
        )
        vals = scaler.decode(out).cpu()
        for dish_id, row in zip(batch["dish_id"], vals):
            preds[dish_id] = {t: float(v) for t, v in zip(TARGETS, row)}
    return compute_metrics(preds, groundtruth), preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    ap.add_argument("--out", type=Path, default=Path("runs/direct"))
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    print(f"device: {device}")
    args.out.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, train_ds, test_ds = build_loaders(args)

    # Statistics from TRAIN only -- using test statistics would leak.
    mean, std = train_ds.target_stats()
    scaler = TargetScaler(mean, std).to(device)
    print("target means:", {t: round(float(m), 1) for t, m in zip(TARGETS, mean)})

    model = DirectNutritionNet(backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.L1Loss()

    groundtruth = {
        d: {t: getattr(test_ds.dishes[d], t) for t in TARGETS} for d in test_ds.ids
    }

    best_train = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_seen, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            y = scaler.encode(batch["targets"].to(device))
            out = model(
                batch["rgb"].to(device),
                batch["depth"].to(device),
                batch["valid"].to(device),
            )
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(y)
            n_seen += len(y)
        sched.step()
        train_loss = running / max(n_seen, 1)

        metrics, _ = evaluate(model, test_loader, scaler, device, groundtruth)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(
            f"epoch {epoch:3d}/{args.epochs}  train_L1={train_loss:.4f}  "
            f"test cal_MAE={metrics['calories_MAE']:6.1f} ({metrics['calories_MAE_%']:4.1f}%)  "
            f"mass_MAE={metrics['mass_MAE']:6.1f} ({metrics['mass_MAE_%']:4.1f}%)  "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )

        # Selection on TRAIN loss, deliberately -- see module docstring.
        if train_loss < best_train:
            best_train = train_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler": scaler.to("cpu").state_dict(),
                    "args": vars(args) | {"data_root": str(args.data_root), "out": str(args.out)},
                    "epoch": epoch,
                },
                args.out / "best.pt",
            )

    metrics, preds = evaluate(model, test_loader, scaler, device, groundtruth)
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    from n5k.eval.metrics import write_predictions_csv

    write_predictions_csv(args.out / "predictions.csv", preds)

    print("\n" + format_table({"direct (RGB-D)": metrics}))
    print(f"\nwrote {args.out}/best.pt, metrics.json, predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
