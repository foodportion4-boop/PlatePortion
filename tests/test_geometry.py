"""Twelve clock hours, strict 15-degree rounding, wrap span and share."""
import numpy as np
from plateclock.plate.geometry import hour_at,place_items

def test_hours_and_edges():
    for k in range(12):
        assert hour_at(k*30)==(k or 12)
        assert hour_at(k*30+14.999)==(k or 12)
        assert hour_at(k*30+15.001)==((k+1)%12 or 12)

def test_zones_share():
    mask=np.zeros((101,101),np.uint8)
    mask[48:53,48:53]=1;mask[48:53,88:93]=2
    items=place_items(mask,{1:'rice',2:'dal'},(50,50),50,minimum_fraction=0)
    assert len(items)==2
    assert {p.zone for p in items}=={'center','near the rim'}
    assert all(abs(p.share-.5)<.001 for p in items)
