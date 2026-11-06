"""Predict all selected validation masks and write present-class mIoU honestly."""
import argparse
import csv
from pathlib import Path
import numpy as np
from plateclock.io.loaders import load_subset
from plateclock.plate.segment_food import FoodSegmenter
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/foodseg-b0');a=p.parse_args()
model=FoodSegmenter(a.checkpoint);conf=np.zeros((104,104),np.int64)
out=Path('data/processed/predictions');out.mkdir(parents=True,exist_ok=True)
from PIL import Image
n=0
for sample in load_subset():
    pred,_=model.predict(sample.image)
    Image.fromarray(pred).save(out/f'{sample.sample_id}.png')
    truth=sample.mask; valid=(truth!=255)&(truth<104)&(pred<104)
    conf+=np.bincount(104*truth[valid].astype(int)+pred[valid].astype(int),minlength=104**2).reshape(104,104)
    n+=1
if n!=300:raise ValueError(f'Expected 300 selected images, got {n}')
iou=np.diag(conf)/(conf.sum(1)+conf.sum(0)-np.diag(conf)).clip(min=1)
present=(conf.sum(1)+conf.sum(0))>0
miou=float(iou[present].mean())
path=Path('results/tables/seg_baseline.csv');path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['images','present_classes','miou','checkpoint']);w.writerow([n,int(present.sum()),f'{miou:.5f}',a.checkpoint])
print(path.read_text())
