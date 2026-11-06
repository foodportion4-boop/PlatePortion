"""Tests for the data layer.

The metadata-schema test is the important one: the dataset README documents a
``num_ingrs`` column that does not exist in the released CSVs, and silently
following the docs shifts every ingredient field by one.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from n5k.data.dataset import DEPTH_UNITS_PER_METRE, decode_depth_png
from n5k.data.metadata import parse_dish_metadata
from n5k.data.splits import assert_disjoint


def test_parses_documented_schema_without_num_ingrs(tmp_path):
    csv = tmp_path / "dish_metadata_cafe1.csv"
    csv.write_text(
        "dish_1,300.5,193.0,12.4,28.2,18.6,"
        "ingr_0000000508,soy sauce,3.4,1.8,0.02,0.17,0.28,"
        "ingr_0000000026,white rice,8.5,11.0,0.03,2.4,0.23\n"
    )
    dishes = parse_dish_metadata(csv)
    dish = dishes["dish_1"]

    assert dish.calories == pytest.approx(300.5)
    assert dish.mass == pytest.approx(193.0)
    assert dish.fat == pytest.approx(12.4)
    assert dish.carb == pytest.approx(28.2)
    assert dish.protein == pytest.approx(18.6)

    # Two ingredients, correctly aligned -- a one-field shift would put the
    # ingredient id into `name` and turn `grams` into a non-numeric string.
    assert len(dish.ingredients) == 2
    assert dish.ingredients[0].name == "soy sauce"
    assert dish.ingredients[0].grams == pytest.approx(3.4)
    assert dish.ingredients[1].name == "white rice"
    assert dish.ingredients[1].grams == pytest.approx(8.5)


def test_rejects_row_with_unexpected_field_count(tmp_path):
    csv = tmp_path / "dish_metadata_cafe1.csv"
    # 6 header + 3 trailing: not a whole number of 7-field ingredient blocks.
    csv.write_text("dish_1,300.5,193.0,12.4,28.2,18.6,a,b,c\n")
    with pytest.raises(ValueError, match="unexpected schema"):
        parse_dish_metadata(csv)


def test_targets_follow_official_field_order(tmp_path):
    csv = tmp_path / "dish_metadata_cafe1.csv"
    csv.write_text("dish_1,300.5,193.0,12.4,28.2,18.6\n")
    dish = parse_dish_metadata(csv)["dish_1"]
    # calories, mass, fat, carb, protein -- matches compute_eval_statistics.py
    assert dish.targets == pytest.approx([300.5, 193.0, 12.4, 28.2, 18.6])


def test_unlabelled_dish_detected(tmp_path):
    csv = tmp_path / "dish_metadata_cafe1.csv"
    csv.write_text("dish_ok,300.5,193.0,12.4,28.2,18.6\ndish_bad,0.0,74.0,0.0,0.0,0.0\n")
    dishes = parse_dish_metadata(csv)
    assert dishes["dish_ok"].is_labelled
    assert not dishes["dish_bad"].is_labelled


def test_depth_decode_scale_and_validity(tmp_path):
    # 10,000 raw units == 1 metre.
    raw = np.array([[0, 4000], [3500, 65535]], dtype=np.uint16)
    path = tmp_path / "depth_raw.png"
    Image.fromarray(raw).save(path)

    depth_m, valid = decode_depth_png(path)
    assert depth_m[0, 1] == pytest.approx(4000 / DEPTH_UNITS_PER_METRE)
    assert depth_m[1, 0] == pytest.approx(0.35)
    # Zero is a sensor dropout, not "0 metres away".
    assert not valid[0, 0]
    assert valid[0, 1] and valid[1, 0]


def test_depth_is_not_clipped_at_40cm(tmp_path):
    """The README claims a 0.4 m ceiling; real files exceed it. Decode must not
    silently clamp, or the table plane and background become indistinguishable."""
    raw = np.array([[65000]], dtype=np.uint16)  # 6.5 m
    path = tmp_path / "depth_raw.png"
    Image.fromarray(raw).save(path)
    depth_m, _ = decode_depth_png(path)
    assert depth_m[0, 0] == pytest.approx(6.5)


def test_zero_byte_image_is_skipped_at_construction(tmp_path):
    """dish_1564159636's depth_raw.png is 0 bytes in the bucket itself.

    It must be filtered when the dataset is built, not blow up mid-epoch.
    """
    root = tmp_path / "nutrition5k"
    (root / "metadata").mkdir(parents=True)
    (root / "dish_ids" / "splits").mkdir(parents=True)
    (root / "metadata" / "dish_metadata_cafe1.csv").write_text(
        "dish_good,300.0,193.0,12.0,28.0,18.0\ndish_empty,300.0,193.0,12.0,28.0,18.0\n"
    )
    (root / "dish_ids" / "splits" / "depth_train_ids.txt").write_text(
        "dish_good\ndish_empty\n"
    )
    for did, size in (("dish_good", 1), ("dish_empty", 0)):
        d = root / "imagery" / "realsense_overhead" / did
        d.mkdir(parents=True)
        if size:
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "rgb.png")
            Image.fromarray(np.full((8, 8), 3500, np.uint16)).save(d / "depth_raw.png")
        else:
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(d / "rgb.png")
            (d / "depth_raw.png").write_bytes(b"")   # 0 bytes, as upstream

    from n5k.data.dataset import Nutrition5kOverhead

    ds = Nutrition5kOverhead(root, "train", "depth", image_size=8)
    assert ds.ids == ["dish_good"]
    assert ds.skipped["empty_file"] == 1
    ds[0]  # must not raise


def test_assert_disjoint_catches_leakage():
    assert_disjoint(["a", "b"], ["c"])
    with pytest.raises(AssertionError, match="both train and test"):
        assert_disjoint(["a", "b"], ["b", "c"])
