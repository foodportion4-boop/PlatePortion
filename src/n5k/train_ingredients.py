"""Train the multi-label ingredient classifier.

    PYTHONPATH=src python3 -m n5k.train_ingredients --epochs 15

Writes ``runs/ingredients/best.pt`` plus the vocabulary and the decision
threshold, which ``n5k.eval.compare --ingredient-checkpoint`` then uses to
replace oracle ingredients with predicted ones.

Same M1 sizing notes as :mod:`n5k.train`: ResNet-18 at 224px, batch 16,
``num_workers=0``, no AMP. Run it on its own -- training concurrently with the
geometric evaluation drives this machine into swap and costs far more than it
saves.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from n5k.data.dataset import Nutrition5kOverhead, collate
from n5k.data.metadata import build_ingredient_vocab, load_dishes
from n5k.data.splits import assert_disjoint, load_split
from n5k.models.direct import pick_device
from n5k.models.ingredients import (
    IngredientClassifier,
    multilabel_metrics,
    tune_threshold,
)


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels, ids = [], [], []
    for batch in loader:
        logits = model(
            batch["rgb"].to(device), batch["depth"].to(device), batch["valid"].to(device)
        )
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch["ingredients"].numpy())
        ids.extend(batch["dish_id"])
    return np.concatenate(probs), np.concatenate(labels), ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/nutrition5k"))
    ap.add_argument("--out", type=Path, default=Path("runs/ingredients"))
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-count", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Vocabulary from the TRAIN split only.
    dishes = load_dishes(args.data_root / "metadata", cafes=("cafe1",))
    train_split = load_split(args.data_root / "dish_ids/splits", "depth", "train")
    vocab = build_ingredient_vocab(dishes, train_split, min_count=args.min_count)
    print(f"device: {device} | vocabulary: {len(vocab)} ingredients "
          f"(>={args.min_count} train occurrences)")

    common = dict(
        data_root=args.data_root, modality="depth", image_size=args.image_size,
        ingredient_vocab=vocab,
    )
    train_ds = Nutrition5kOverhead(subset="train", augment=True, **common)
    test_ds = Nutrition5kOverhead(subset="test", augment=False, **common)
    assert_disjoint(train_ds.ids, test_ds.ids)
    print(f"train: {len(train_ds)} dishes | test: {len(test_ds)} dishes")

    loader_args = dict(
        batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers
    )
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_args)
    train_eval_loader = DataLoader(train_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model = IngredientClassifier(len(vocab), backbone=args.backbone).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_weight = train_ds.ingredient_pos_weight().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"pos_weight: median={pos_weight.median():.1f} max={pos_weight.max():.1f}")

    best_train, history = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for batch in train_loader:
            y = batch["ingredients"].to(device)
            logits = model(
                batch["rgb"].to(device),
                batch["depth"].to(device),
                batch["valid"].to(device),
            )
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(y)
            seen += len(y)
        sched.step()
        train_loss = running / max(seen, 1)

        probs, labels, _ = collect(model, test_loader, device)
        m = multilabel_metrics(probs, labels, 0.5)
        history.append({"epoch": epoch, "train_loss": train_loss, **m})
        print(
            f"epoch {epoch:3d}/{args.epochs}  BCE={train_loss:.4f}  "
            f"test microF1={m['micro_f1']:.3f} macroF1={m['macro_f1']:.3f} "
            f"mAP={m['mAP']:.3f}  [{time.time() - t0:.0f}s]",
            flush=True,
        )

        # Selection on train loss, as in n5k.train -- see that module's note.
        if train_loss < best_train:
            best_train = train_loss
            torch.save(
                {"model": model.state_dict(), "vocab": vocab, "epoch": epoch,
                 "args": vars(args) | {k: str(v) for k, v in vars(args).items()
                                       if isinstance(v, Path)}},
                args.out / "best.pt",
            )

    # Threshold is a fitted parameter: tune on TRAIN predictions, never test.
    tr_probs, tr_labels, _ = collect(model, train_eval_loader, device)
    threshold, tr_f1 = tune_threshold(tr_probs, tr_labels)
    print(f"\ntuned threshold on train: {threshold:.2f} (train microF1 {tr_f1:.3f})")

    probs, labels, _ = collect(model, test_loader, device)
    final = multilabel_metrics(probs, labels, threshold)
    print("test @ tuned threshold: " + "  ".join(f"{k}={v:.3f}" for k, v in final.items()))

    ckpt = torch.load(args.out / "best.pt", map_location="cpu", weights_only=False)
    ckpt["threshold"] = threshold
    torch.save(ckpt, args.out / "best.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "metrics.json").write_text(
        json.dumps({"threshold": threshold, "test": final}, indent=2)
    )
    print(f"\nwrote {args.out}/best.pt (vocab + threshold), metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
