"""Known affine tilted ellipses should recover plate-relative rays."""
import cv2
import numpy as np
from plateclock.plate.detect_plate import PlateFit
from plateclock.plate.rectify import rectification,warp
from plateclock.plate.geometry import angle_at

def test_vertical_invariant_and_tilt():
    center=(100.,100.); radius=65
    for tilt in [0,20,40]:
        minor=radius*np.cos(np.deg2rad(tilt))
        fit=PlateFit(center,(2*radius,2*minor),0,1)
        affine,_=rectification(fit,(201,201,3))
        assert np.allclose(affine[:2,:2],np.diag([np.sqrt(minor/radius),np.sqrt(radius/minor)]),atol=1e-5)
        assert np.allclose(affine[:2,:2]@np.array([0,1]),[0,affine[1,1]])
        src=np.zeros((201,201),np.uint8);cv2.circle(src,(130,66),3,1,-1)
        corrected=warp(src,affine,is_mask=True)
        assert corrected.sum()>0
