"""Synthetic affine tilt: compare angular errors before and after ellipse correction."""
import csv
import numpy as np
from plateclock.plate.detect_plate import PlateFit,detect_plate
from plateclock.plate.rectify import rectification
from plateclock.plate.geometry import angle_at
import cv2

def diff(a,b):return abs((a-b+180)%360-180)
rows=[]
for tilt in (0,20,40):
    raw=[];fixed=[];fail=0
    for angle in np.arange(0,360,10):
        # Known near-top-down disk and target point; compress image vertical axis.
        base=np.zeros((300,300,3),np.uint8)
        cv2.circle(base,(150,150),105,(230,230,230),-1)
        cv2.circle(base,(150,150),101,(150,150,150),2)
        transform=np.array([[1,0,0],[0,np.cos(np.deg2rad(tilt)),150*(1-np.cos(np.deg2rad(tilt)))]],np.float32)
        tilted=cv2.warpAffine(base,transform,(300,300))
        point=np.array([150+70*np.sin(np.deg2rad(angle)),150-70*np.cos(np.deg2rad(angle)),1])
        warped=transform@point
        try: fit=detect_plate(tilted)
        except ValueError:fail+=1;continue
        correction,_=rectification(fit,tilted.shape)
        new=correction@np.array([*warped,1])
        raw.append(diff(angle_at(*warped,150,150),angle))
        fixed.append(diff(angle_at(*new,*fit.center),angle))
    rows.append([tilt,len(raw),fail,np.median(raw) if raw else '',np.median(fixed) if fixed else ''])
with open('results/tables/tilt_experiment.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['tilt_deg','passed','failed','raw_median_deg','corrected_median_deg']);w.writerows(rows)
print(*rows,sep='\n')
