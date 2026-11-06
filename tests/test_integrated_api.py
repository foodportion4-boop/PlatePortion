"""API gates and integration decisions, independent of model weights."""
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from plateclock.integrated import IntegratedService
from web.server import app, service

def png(mode='RGB'):
    buf=BytesIO()
    Image.new(mode,(32,32)).save(buf,format='PNG')
    buf.seek(0)
    return buf

def test_health_and_missing_upload():
    client=app.test_client()
    assert client.get('/api/health').status_code==200
    response=client.post('/api/analyze')
    assert response.status_code==400
    assert 'Choose' in response.json['error']

def test_rgb_without_checkpoint_is_honest(monkeypatch):
    monkeypatch.setattr(service,'readiness',lambda:{'plate':False,'direct':False,'calibration':False})
    response=app.test_client().post('/api/analyze',data={'photo':(png(),'plate.png')})
    assert response.status_code==200
    assert response.json['plate'] is None
    assert response.json['portion'] is None
    assert any('model missing' in w.lower() for w in response.json['warnings'])

def test_reject_fake_depth_before_numeric_inference():
    svc=IntegratedService()
    with pytest.raises(ValueError,match='16-bit'):
        svc._analyze_depth(np.zeros((480,640,3),dtype=np.uint8),np.zeros((480,640),dtype=np.uint8),{})
