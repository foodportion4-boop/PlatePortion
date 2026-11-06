"""Read locally exported FoodSeg103 RGB images and integer masks."""
from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image

@dataclass
class FoodSample:
    image: np.ndarray
    mask: np.ndarray
    class_names: dict[int, str]
    sample_id: str

def load_record(row, root='data/raw/foodseg103'):
    root = Path(root)
    image = np.asarray(Image.open(root / row['image_path']).convert('RGB'))
    mask = np.asarray(Image.open(root / row['mask_path']))
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError(f"Bad mask format/shape for {row['id']}: {mask.shape}, {image.shape}")
    names = {int(k): v for k, v in json.loads((root / 'classes.json').read_text()).items()}
    unknown = set(np.unique(mask).tolist()) - set(names) - {255}
    if unknown:
        raise ValueError(f'Unknown label IDs: {unknown}')
    return FoodSample(image, mask, names, str(row['id']))

def load_subset(csv_path='data/processed/plate_subset.csv', root='data/raw/foodseg103'):
    with open(csv_path, newline='') as stream:
        for row in csv.DictReader(stream):
            yield load_record(row, root)
