"""Tests for the evaluation metrics.

Pinned against the official ``compute_eval_statistics.py`` semantics. The MAE%%
normalisation is the subtle one and is worth a dedicated test: the official
script divides by the **mean of ground truth over the set**, not by each
sample's own value.
"""

from __future__ import annotations

import pytest

from n5k.eval.metrics import BASELINES, FIELDS, compute_metrics


def _row(calories, mass, fat, carb, protein):
    return {
        "calories": calories, "mass": mass, "fat": fat,
        "carb": carb, "protein": protein,
    }


def test_mae_is_mean_absolute_error():
    gt = {"a": _row(100, 100, 10, 10, 10), "b": _row(200, 200, 20, 20, 20)}
    pred = {"a": _row(110, 90, 12, 8, 10), "b": _row(180, 220, 20, 24, 15)}
    m = compute_metrics(pred, gt)
    assert m["calories_MAE"] == pytest.approx((10 + 20) / 2)
    assert m["mass_MAE"] == pytest.approx((10 + 20) / 2)
    assert m["protein_MAE"] == pytest.approx((0 + 5) / 2)


def test_mae_pct_normalises_by_mean_groundtruth_not_per_sample():
    """The distinction that reimplementations get wrong.

    Errors of 10 on gt=10 and 10 on gt=190.
      official  : 100 * mean(|err|) / mean(gt) = 100 * 10 / 100 = 10%
      per-sample: mean(|err|/gt) = mean(100%, 5.26%) = 52.6%
    """
    gt = {"a": _row(10, 1, 1, 1, 1), "b": _row(190, 1, 1, 1, 1)}
    pred = {"a": _row(20, 1, 1, 1, 1), "b": _row(180, 1, 1, 1, 1)}
    m = compute_metrics(pred, gt)
    assert m["calories_MAE"] == pytest.approx(10.0)
    assert m["calories_MAE_%"] == pytest.approx(10.0)
    assert m["calories_MAE_%"] != pytest.approx(52.6, abs=1.0)


def test_perfect_prediction_is_zero_error():
    gt = {"a": _row(100, 50, 5, 5, 5)}
    m = compute_metrics(dict(gt), gt)
    for f in FIELDS:
        assert m[f"{f}_MAE"] == pytest.approx(0.0)
        assert m[f"{f}_MAE_%"] == pytest.approx(0.0)


def test_missing_groundtruth_raises_rather_than_silently_skipping():
    gt = {"a": _row(100, 50, 5, 5, 5)}
    pred = {"a": _row(100, 50, 5, 5, 5), "ghost": _row(1, 1, 1, 1, 1)}
    with pytest.raises(KeyError, match="no ground truth"):
        compute_metrics(pred, gt)


def test_only_predicted_dishes_are_scored():
    """Official script iterates predictions, so extra ground truth is ignored."""
    gt = {"a": _row(100, 50, 5, 5, 5), "b": _row(999, 999, 99, 99, 99)}
    pred = {"a": _row(100, 50, 5, 5, 5)}
    m = compute_metrics(pred, gt)
    assert m["n_dishes"] == 1
    assert m["calories_MAE"] == pytest.approx(0.0)


def test_baselines_cover_all_fields():
    for name, table in BASELINES.items():
        assert set(table) == set(FIELDS), f"{name} missing fields"
        for field, (mae, pct) in table.items():
            assert mae > 0 and pct > 0
