"""Create a synthetic depth PNG from a plate photo for presentation only.

Run from project root: python scripts/make_demo_depth.py photo.jpg demo_depth.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from plateclock.pipeline import run
from plateclock.plate.segment_food import FoodSegmenter
from plateclock.demo_depth import synthesize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--checkpoint',default='checkpoints/foodseg-b0')
    args=parser.parse_args()
    with Image.open(args.photo) as image:
        rgb=np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    result=run(rgb,FoodSegmenter(args.checkpoint))
    if result['status']!='ok':
        raise SystemExit(f"Plate analysis failed: {result['reason'] or result['status']}")
    raw, info=synthesize(result['labels'],result['plate'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(raw).save(args.output)
    print(f"Saved {args.output} ({raw.shape[1]}x{raw.shape[0]}, 16-bit).")
    print(info['reason'])
    print(f"Illustrative volume: {info['volume_cm3']} cm3; mass: {info['nutrition']['mass']} g; energy: {info['nutrition']['calories']} kcal.")


if __name__=='__main__':main()
