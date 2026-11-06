"""Qualitative own-photo review. No ground-truth or accuracy claim."""
import argparse
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image, ImageOps
from plateclock.pipeline import run, load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--photos', default='data/processed/own_photos')
    p.add_argument('--out', default='results/week2/stress')
    p.add_argument('--config', default='configs/week2.yaml')
    a = p.parse_args()
    cfg = load_config(a.config)
    files = sorted(f for f in Path(a.photos).iterdir() if f.suffix.lower() in {'.jpg','.jpeg','.png'})
    if len(files) != 30:
        raise ValueError(f'Expected 30 photos, found {len(files)}. Convert HEIC to JPEG first.')
    from plateclock.plate.segment_food import FoodSegmenter
    model = FoodSegmenter(cfg['model']['checkpoint'])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(files):
        with Image.open(path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        start = perf_counter(); result = run(rgb, model, cfg)
        elapsed = perf_counter()-start
        Image.fromarray(result['overlay']).save(out/f'{i:02d}_overlay.png')
        rows.append(dict(file=path.name, status=result['status'], reason=result['reason'],
                         text=result['text'], seconds=elapsed, visual_correct='pending',
                         failure_type='pending', notes=''))
    with (out/'failures.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('Review all 30 originals and overlays; fill human columns:', out/'failures.csv')

if __name__ == '__main__':
    main()
