"""Week 1 plate-only path: detection, rectification, segmentation, placement, overlay."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from plateclock.plate.detect_plate import detect_plate,ellipse_mask
from plateclock.plate.rectify import rectification,warp
from plateclock.plate.geometry import place_items
from plateclock.io.class_names import CLASS_NAMES
from plateclock.language.plate_nlg import describe
from plateclock.plate.segment_food import FoodSegmenter

def run(image, segmenter, debug_food_mask=None):
    # Predicted food guides plate ranking, but never use evaluation GT here.
    labels,confidence=segmenter.predict(image)
    fit=detect_plate(image, labels>0)
    affine,radius=rectification(fit,image.shape)
    rectified=warp(image,affine)
    rect_labels=warp(labels,affine,is_mask=True)
    plate=warp(ellipse_mask(image.shape,fit).astype('uint8'),affine,is_mask=True).astype(bool)
    items=place_items(rect_labels,CLASS_NAMES,fit.center,radius,plate_mask=plate)
    overlay=rectified.copy()
    for rad in [radius*.33,radius*.72,radius]:
        cv2.circle(overlay,tuple(map(int,fit.center)),int(rad),(250,230,100),1)
    for hour in range(12):
        a=np.deg2rad(hour*30)
        end=(int(fit.center[0]+radius*np.sin(a)),int(fit.center[1]-radius*np.cos(a)))
        cv2.line(overlay,tuple(map(int,fit.center)),end,(250,230,100),1)
    for item in items:
        a=np.deg2rad(item.angle_deg); pos=(int(fit.center[0]+radius*.65*np.sin(a)),int(fit.center[1]-radius*.65*np.cos(a)))
        cv2.putText(overlay,f'{item.label} {item.hour}',pos,cv2.FONT_HERSHEY_SIMPLEX,.4,(255,50,50),1)
    return dict(fit=fit,items=items,text=describe(items),overlay=overlay,mean_confidence=float(confidence[labels>0].mean()) if (labels>0).any() else 0)

def batch(subset='data/processed/plate_subset.csv', count=20, checkpoint='checkpoints/foodseg-b0'):
    from plateclock.io.loaders import load_subset
    model=FoodSegmenter(checkpoint); out=Path('results/figures/checkpoint1');out.mkdir(parents=True,exist_ok=True)
    reviewed=[]
    for sample in load_subset(subset):
        try:
            result=run(sample.image,model)
            Image.fromarray(result['overlay']).save(out/f'{sample.sample_id}.png')
            reviewed.append((sample.sample_id,result['text'],'overlay saved; visual review pending',result['fit'].score))
        except ValueError as e: reviewed.append((sample.sample_id,'','FAIL: '+str(e),0))
        if len(reviewed)>=count:break
    if len(reviewed)!=count:raise ValueError(f'Only {len(reviewed)} rows available')
    import csv
    with (out/'review.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','description','status','plate_score']);w.writerows(reviewed)
    print('Review overlays and replace status with correct / failure cause:',out)
