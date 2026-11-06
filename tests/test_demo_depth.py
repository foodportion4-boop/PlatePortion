"""Synthetic depth stays visibly illustrative and encodes valid raw PNG units."""
import numpy as np
import pytest
from plateclock.demo_depth import synthesize
from plateclock.integrated import IntegratedService


def test_demo_depth_preserves_raw_shape_and_assumptions():
    y,x=np.ogrid[:240,:320]
    plate=(x-160)**2+(y-120)**2 < 100**2
    labels=np.zeros((240,320),np.uint8)
    labels[(x-160)**2+(y-120)**2 < 45**2]=3
    raw, result=synthesize(labels,plate)
    assert raw.shape==(480,640) and raw.dtype==np.uint16
    assert raw.min()<4000 and raw.max()==4000
    assert result['status']=='illustrative'
    assert result['volume_cm3']>0
    assert result['nutrition']['mass']>0


def test_demo_rejects_real_depth_mix():
    with pytest.raises(ValueError,match='either a real depth'):
        IntegratedService().analyze(np.zeros((1,1,3),np.uint8),np.zeros((1,1),np.uint16),demo=True)
