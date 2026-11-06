"""Download FoodSeg103 through datasets and export lossless images/masks locally."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument('--output', default='data/raw/foodseg103')
p.add_argument('--split', choices=['train','validation','both'], default='both')
a = p.parse_args()
root = Path(a.output)
root.mkdir(parents=True, exist_ok=True)
for split in (['train','validation'] if a.split == 'both' else [a.split]):
    ds = load_dataset('EduardoPacheco/FoodSeg103', split=split)
    from plateclock.io.class_names import CLASS_NAMES
    (root / 'classes.json').write_text(json.dumps(CLASS_NAMES, indent=2))
    rows = []
    (root / split / 'images').mkdir(parents=True, exist_ok=True)
    (root / split / 'masks').mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(ds):
        ident = str(row.get('id', idx))
        im = row['image'].convert('RGB')
        lab = row['label']
        arr = np.asarray(lab)
        if arr.ndim != 2 or arr.max() > 255:
            raise ValueError(f'Expected one-channel 8-bit class-ID mask for {ident}: {arr.shape}/{arr.max()}')
        image_path = f'{split}/images/{idx:05d}.png'
        mask_path = f'{split}/masks/{idx:05d}.png'
        im.save(root / image_path)
        from PIL import Image
        Image.fromarray(arr.astype('uint8')).save(root / mask_path)
        rows.append(dict(id=ident, image_path=image_path, mask_path=mask_path))
    with (root / f'{split}.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id','image_path','mask_path'])
        w.writeheader(); w.writerows(rows)
    print(split, len(rows), 'images exported to', root)
