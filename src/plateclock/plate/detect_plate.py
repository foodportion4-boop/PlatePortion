"""Fit a plate ellipse from color/edge contours; report confidence and failures."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class PlateFit:
    center: tuple[float,float]
    axes: tuple[float,float]  # ellipse DIAMETERS along major/minor axes
    rotation: float          # OpenCV angle of major axis, degrees
    score: float

def detect_plate(rgb, food_mask=None):
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    gray=cv2.GaussianBlur(gray,(7,7),1.5)
    edges=cv2.Canny(gray,35,100)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # Also consider broad closed contours from adaptive background/plate contrast.
    candidates=[]
    for c in contours:
        if len(c)<20: continue
        fit=cv2.fitEllipse(c)
        (cx,cy),(aa,bb),angle=fit
        if aa<bb: aa,bb=bb,aa; angle+=90
        if bb<min(h,w)*.22 or aa>max(h,w)*1.35: continue
        poly=np.zeros((h,w),np.uint8)
        cv2.ellipse(poly,fit,1,-1)
        area=np.count_nonzero(poly)
        if area<.12*h*w or area>.92*h*w: continue
        center_penalty=np.hypot((cx-w/2)/w,(cy-h/2)/h)
        if center_penalty>.34: continue
        contour_area=abs(cv2.contourArea(c))
        fill=min(contour_area/max(area,1),1)
        aspect=bb/aa
        overlap=(np.count_nonzero((food_mask>0)&(poly>0))/max(np.count_nonzero(food_mask),1)) if food_mask is not None else .5
        score=.32*fill+.24*aspect+.30*overlap+.14*(1-center_penalty/.34)
        candidates.append(PlateFit((cx,cy),(aa,bb),angle%180,float(score)))
    if not candidates: raise ValueError('No plausible full plate ellipse: check photo angle, cropping, and contrast')
    return max(candidates,key=lambda p:p.score)

def ellipse_mask(shape, fit):
    mask=np.zeros(shape[:2],np.uint8)
    cv2.ellipse(mask,(tuple(map(int,fit.center)),tuple(map(int,fit.axes)),fit.rotation),1,-1)
    return mask.astype(bool)
