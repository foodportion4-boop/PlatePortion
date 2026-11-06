"""Affine ellipse-to-circle approximation that preserves vertical diner axis."""
import cv2
import numpy as np

def rectification(fit, shape):
    h,w=shape[:2]; cx,cy=fit.center
    # Symmetric positive-definite inverse square root: avoids rotating the diner direction.
    t=np.deg2rad(fit.rotation); c,s=np.cos(t),np.sin(t)
    rot=np.array([[c,-s],[s,c]])
    a,b=fit.axes[0]/2,fit.axes[1]/2
    radius=np.sqrt(a*b)
    linear=rot @ np.diag([radius/a,radius/b]) @ rot.T
    center=np.array([cx,cy])
    affine=np.column_stack([linear,center-linear@center]).astype(np.float32)
    return affine,float(radius)

def warp(array, affine, is_mask=False):
    h,w=array.shape[:2]
    return cv2.warpAffine(array,affine,(w,h),flags=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
