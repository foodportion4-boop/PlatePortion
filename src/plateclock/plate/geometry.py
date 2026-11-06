"""Stable vision-to-language placement contract in diner-oriented coordinates."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ItemPlacement:
    label: str
    hour: int
    zone: str
    share: float         # fraction of TOTAL FOOD pixels
    span: tuple[int,int] # inclusive start/end clock hours, may wrap 12
    angle_deg: float

def hour_at(angle):
    return int(np.floor(((angle%360)+15)/30))%12 or 12

def angle_at(x,y,cx,cy):
    return float(np.degrees(np.arctan2(x-cx,cy-y))%360)

def place_items(label_mask, names, center, radius, plate_mask=None, minimum_fraction=.02,
                center_boundary=.33, rim_boundary=.72, max_items=5):
    if radius<=0: raise ValueError('Plate radius must be positive')
    yy,xx=np.indices(label_mask.shape)
    inside=plate_mask.astype(bool) if plate_mask is not None else (xx-center[0])**2+(yy-center[1])**2<=radius**2
    food=inside&(label_mask>0)&(label_mask!=255)
    total=int(food.sum()); plate_area=int(inside.sum())
    if not total:return []
    result=[]
    for ident in np.unique(label_mask[food]):
        region=food&(label_mask==ident)
        size=int(region.sum())
        if size/max(plate_area,1)<minimum_fraction:continue
        ys,xs=np.where(region); cx,cy=xs.mean(),ys.mean()
        angle=angle_at(cx,cy,*center)
        radial=float(np.hypot(cx-center[0],cy-center[1])/radius)
        zone='center' if radial<center_boundary else 'middle' if radial<rim_boundary else 'near the rim'
        # Minimal circular arc covering 95% of region pixels to resist isolated noise.
        angles=np.sort(np.degrees(np.arctan2(xs-center[0],center[1]-ys))%360)
        lo=int(.025*len(angles)); hi=max(lo+1,int(.975*len(angles)))
        n=hi-lo
        extended=np.concatenate([angles,angles+360])
        widths=extended[np.arange(n)+hi]-extended[np.arange(n)+lo]
        k=int(np.argmin(widths)); start=extended[k+lo]%360; end=extended[k+hi]%360
        span=(hour_at(start),hour_at(end))
        result.append(ItemPlacement(names.get(int(ident),f'class {ident}'),hour_at(angle),zone,size/total,span,angle))
    return sorted(result,key=lambda p:p.share,reverse=True)[:max_items]
