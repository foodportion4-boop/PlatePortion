"""Tests for the ingredient vocabulary, multi-hot targets, and classifier metrics."""

from __future__ import annotations

import numpy as np
import pytest

from n5k.data.metadata import Dish, Ingredient, build_ingredient_vocab
from n5k.models.ingredients import (
    _average_precision,
    multilabel_metrics,
    tune_threshold,
)


def _dish(did, *names):
    return Dish(
        did, calories=100, mass=100, fat=1, carb=1, protein=1,
        ingredients=[Ingredient(f"ingr_{n}", n, 10, 10, 1, 1, 1) for n in names],
    )


def test_ingredient_nutrition_parses_name_first_schema(tmp_path):
    """ingredients_metadata.csv has the NAME first and a plain integer id.

    A parser keying on an ``ingr_`` prefix (as the dish metadata uses) matches
    nothing and returns an empty dict silently -- the exact bug this replaced.
    """
    from n5k.data.metadata import load_ingredient_nutrition

    path = tmp_path / "ingredients_metadata.csv"
    path.write_text(
        "ingr,id,cal/g,fat(g),carb(g),protein(g)\n"
        "cottage cheese,1,0.980,0.043,0.034,0.110\n"
        "olive oil,2,8.840,1.000,0.000,0.000\n"
    )
    table = load_ingredient_nutrition(tmp_path)

    assert set(table) == {"cottage cheese", "olive oil"}
    assert table["olive oil"]["kcal_per_g"] == pytest.approx(8.84)
    assert table["olive oil"]["fat_per_g"] == pytest.approx(1.0)
    assert table["cottage cheese"]["protein_per_g"] == pytest.approx(0.110)


def test_ingredient_nutrition_tolerates_missing_header(tmp_path):
    from n5k.data.metadata import load_ingredient_nutrition

    path = tmp_path / "ingredients_metadata.csv"
    path.write_text("cottage cheese,1,0.980,0.043,0.034,0.110\n")
    table = load_ingredient_nutrition(tmp_path)
    assert table["cottage cheese"]["kcal_per_g"] == pytest.approx(0.98)


def test_ingredient_nutrition_reports_missing_file(tmp_path):
    from n5k.data.metadata import load_ingredient_nutrition

    with pytest.raises(FileNotFoundError, match="n5k.data.download"):
        load_ingredient_nutrition(tmp_path)


def test_vocab_drops_the_long_tail():
    dishes, ids = {}, []
    for i in range(30):
        dishes[f"d{i}"] = _dish(f"d{i}", "rice")          # 30 occurrences
        ids.append(f"d{i}")
    for i in range(30, 35):
        dishes[f"d{i}"] = _dish(f"d{i}", "rice", "saffron")  # saffron: 5
        ids.append(f"d{i}")

    assert build_ingredient_vocab(dishes, ids, min_count=20) == ["rice"]
    assert build_ingredient_vocab(dishes, ids, min_count=5) == ["rice", "saffron"]


def test_vocab_is_sorted_and_deduplicated():
    dishes = {"a": _dish("a", "beans", "apple", "beans")}
    vocab = build_ingredient_vocab(dishes, ["a"], min_count=1)
    assert vocab == sorted(set(vocab))
    assert vocab == ["apple", "beans"]


def test_vocab_ignores_ids_with_no_metadata():
    dishes = {"a": _dish("a", "rice")}
    assert build_ingredient_vocab(dishes, ["a", "missing"], min_count=1) == ["rice"]


def test_perfect_predictions_score_one():
    labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    m = multilabel_metrics(labels.copy(), labels, threshold=0.5)
    assert m["micro_f1"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["mAP"] == pytest.approx(1.0)


def test_all_negative_predictions_score_zero_not_nan():
    """The failure mode unweighted BCE falls into -- must not read as success."""
    labels = np.array([[1, 0, 1], [0, 1, 0]], dtype=float)
    probs = np.zeros_like(labels)
    m = multilabel_metrics(probs, labels, threshold=0.5)
    assert m["micro_f1"] == pytest.approx(0.0)
    assert m["micro_recall"] == pytest.approx(0.0)
    assert np.isfinite(m["macro_f1"])
    assert m["pred_per_dish"] == pytest.approx(0.0)


def test_micro_and_macro_f1_diverge_when_rare_classes_fail():
    """A model that nails the common class and misses the rare one should show
    healthy micro-F1 and poor macro-F1 -- the reason both are reported."""
    n = 100
    labels = np.zeros((n, 2))
    labels[:, 0] = 1.0          # common: always present
    labels[:5, 1] = 1.0         # rare: 5 of 100
    probs = np.zeros((n, 2))
    probs[:, 0] = 0.9           # common predicted perfectly
    probs[:, 1] = 0.1           # rare never predicted

    m = multilabel_metrics(probs, labels, threshold=0.5)
    assert m["micro_f1"] > 0.9
    assert m["macro_f1"] < 0.6


def test_average_precision_rewards_ranking():
    y = np.array([1, 0, 1, 0], dtype=float)
    good = np.array([0.9, 0.2, 0.8, 0.1])   # positives ranked first
    bad = np.array([0.1, 0.8, 0.2, 0.9])    # positives ranked last
    assert _average_precision(good, y > 0.5) == pytest.approx(1.0)
    assert _average_precision(bad, y > 0.5) < 0.6


def test_average_precision_handles_class_with_no_positives():
    assert _average_precision(np.array([0.5, 0.4]), np.array([False, False])) == 0.0


def test_tune_threshold_finds_the_separating_value():
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    probs = np.array([[0.8, 0.1], [0.7, 0.2], [0.2, 0.9], [0.1, 0.8]])
    threshold, f1 = tune_threshold(probs, labels)
    assert f1 == pytest.approx(1.0)
    assert 0.2 < threshold <= 0.7


def test_dataset_builds_multi_hot_and_pos_weight(tmp_path):
    """Multi-hot targets must align with the vocabulary, and pos_weight must
    grow for rarer ingredients."""
    from PIL import Image
    from n5k.data.dataset import Nutrition5kOverhead

    root = tmp_path / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    (root / "dish_ids" / "splits").mkdir(parents=True)

    rows, ids = [], []
    for i in range(10):
        # "rice" in every dish; "beans" in only the first two.
        ings = "ingr_1,rice,10.0,10.0,1.0,1.0,1.0"
        if i < 2:
            ings += ",ingr_2,beans,10.0,10.0,1.0,1.0,1.0"
        rows.append(f"dish_{i},100.0,100.0,1.0,1.0,1.0,{ings}")
        ids.append(f"dish_{i}")
    (root / "metadata" / "dish_metadata_cafe1.csv").write_text("\n".join(rows) + "\n")
    (root / "dish_ids" / "splits" / "depth_train_ids.txt").write_text("\n".join(ids) + "\n")

    for did in ids:
        d = root / "imagery" / "realsense_overhead" / did
        d.mkdir(parents=True)
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "rgb.png")
        Image.fromarray(np.full((8, 8), 3500, np.uint16)).save(d / "depth_raw.png")

    vocab = ["beans", "rice"]
    ds = Nutrition5kOverhead(root, "train", "depth", image_size=8, ingredient_vocab=vocab)

    with_beans = ds[ids.index("dish_0")]
    without = ds[ids.index("dish_5")]
    assert with_beans.ingredients.tolist() == [1.0, 1.0]
    assert without.ingredients.tolist() == [0.0, 1.0]

    pw = ds.ingredient_pos_weight()
    assert pw.shape == (2,)
    assert pw[0] > pw[1], "rarer ingredient must carry the larger positive weight"


def test_dataset_without_vocab_yields_empty_ingredient_vector(tmp_path):
    """The direct-regression path must keep working with no vocabulary set."""
    from PIL import Image
    from n5k.data.dataset import Nutrition5kOverhead

    root = tmp_path / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    (root / "dish_ids" / "splits").mkdir(parents=True)
    (root / "metadata" / "dish_metadata_cafe1.csv").write_text(
        "dish_0,100.0,100.0,1.0,1.0,1.0,ingr_1,rice,10.0,10.0,1.0,1.0,1.0\n"
    )
    (root / "dish_ids" / "splits" / "depth_train_ids.txt").write_text("dish_0\n")
    d = root / "imagery" / "realsense_overhead" / "dish_0"
    d.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "rgb.png")
    Image.fromarray(np.full((8, 8), 3500, np.uint16)).save(d / "depth_raw.png")

    ds = Nutrition5kOverhead(root, "train", "depth", image_size=8)
    assert ds[0].ingredients.numel() == 0
    assert ds.ingredient_pos_weight().numel() == 0
