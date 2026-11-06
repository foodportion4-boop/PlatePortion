"""Bridge PlateClock RGB narration with Nutrition5k RGB-D portion estimation.

No model is downloaded, and numeric nutrition is never inferred from a single RGB photo.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image

from plateclock.pipeline import run as plate_run, public_result
from plateclock.plate.segment_food import FoodSegmenter
from n5k.data.dataset import DEPTH_NEAR_M, DEPTH_FAR_M, IMAGENET_MEAN, IMAGENET_STD
from n5k.geometric.volume import estimate_volume

class IntegratedService:
    def __init__(self, plate_checkpoint='checkpoints/foodseg-b0',
                 direct_checkpoint='checkpoints/nutrition-direct/best.pt',
                 calibration='checkpoints/nutrition-calibration.json'):
        self.plate_checkpoint=Path(plate_checkpoint)
        self.direct_checkpoint=Path(direct_checkpoint)
        self.calibration=Path(calibration)
        self._segmenter=None
        self._direct=None
        self._scaler=None
        self._device=None

    def readiness(self):
        pc=self.plate_checkpoint
        return {'plate': (pc/'config.json').is_file() and (pc/'preprocessor_config.json').is_file() and
                any((pc/n).is_file() for n in ('model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json')),
                'direct': self.direct_checkpoint.is_file(),
                'calibration': self.calibration.is_file()}

    def analyze(self, rgb: np.ndarray, depth_raw: np.ndarray | None=None, mode='full', demo=False):
        if demo and depth_raw is not None:
            raise ValueError('Choose either a real depth image or RGB demo mode, not both.')
        state=self.readiness()
        result={'plate':None,'portion':None,'warnings':[]}
        if state['plate']:
            if self._segmenter is None:
                self._segmenter=FoodSegmenter(str(self.plate_checkpoint))
            plate=plate_run(rgb,self._segmenter,mode=mode)
            result['plate']=public_result(plate)
            result['_overlay']=plate['overlay']
            result['plate']['portion_shares']=[{
                'label':r.label,'plate_area_percent':round(r.area_fraction*100,1),
                'food_area_percent':round(r.share*100,1)
            } for r,d in zip(plate.get('regions',[]),plate.get('all_decisions',[])) if d.action!='omit'][:5]
            if plate['status']!='ok':
                result['warnings'].append('Plate narration is unavailable or incomplete for this photo.')
        else:
            result['warnings'].append('Plate model missing: copy the saved FoodSeg103 checkpoint into checkpoints/foodseg-b0.')
        if demo:
            if result['plate'] is not None and plate['status'] == 'ok':
                from plateclock.demo_depth import synthesize
                try:
                    result['_demo_depth'], result['portion'] = synthesize(plate['labels'], plate['plate'])
                    result['warnings'].append('RGB demo: synthetic depth and illustrative numbers, not physical measurements.')
                except ValueError as exc:
                    result['warnings'].append(str(exc))
            else:
                result['warnings'].append('RGB demo requires a successfully detected plate and food regions.')
        elif depth_raw is not None:
            result['portion']=self._analyze_depth(rgb,depth_raw,state)
        else:
            result['warnings'].append('An RGB photo supports relative plate share only. Gram, calorie and macro estimates require matching depth and a trained/calibrated Nutrition5k model.')
        return result

    def _analyze_depth(self,rgb,raw,state):
        if raw.ndim!=2 or raw.dtype!=np.uint16:
            raise ValueError('Depth must be a single-channel 16-bit raw PNG; do not use a colored depth preview.')
        if rgb.shape[:2]!=raw.shape or rgb.shape[:2]!=(480,640):
            raise ValueError('RGB and depth must be aligned 640×480 Nutrition5k-style overhead frames (same pixels and orientation).')
        valid=raw>0
        if valid.mean()<.5:raise ValueError('Depth contains too many missing pixels; check that it is the matching raw depth frame.')
        depth=raw.astype(np.float32)/10000.0
        plausible=valid & (depth>DEPTH_NEAR_M) & (depth<DEPTH_FAR_M)
        if plausible.mean()<.5:
            raise ValueError('Depth units/range do not look like Nutrition5k raw depth (10,000 units per metre).')
        vol=estimate_volume(depth,valid)
        portion={'method':'Nutrition5k depth geometry','status':'estimated' if vol.reliable else 'unreliable',
                 'volume_cm3':round(vol.volume_cm3,1) if vol.reliable else None,
                 'container':vol.container,'filled_fraction':round(vol.filled_fraction,3),
                 'reason':'' if vol.reliable else 'Depth surface fit, container shape or missing pixels failed the geometric reliability check.',
                 'nutrition':None,'direct':None}
        if vol.reliable and state['calibration']:
            from n5k.geometric.calibrate import Calibration
            pred=Calibration.load(self.calibration).predict(vol.volume_cm3)
            portion['nutrition']={k:round(float(v),1) for k,v in pred.items()}
        if state['direct']:
            portion['direct']=self._direct_predict(rgb,depth,valid)
        return portion

    def _direct_predict(self,rgb,depth,valid):
        import torch
        from n5k.models.direct import DirectNutritionNet,TargetScaler,pick_device
        from n5k.data.metadata import TARGETS
        if self._direct is None:
            # torch.load unpickles: supply ONLY a checkpoint trained by your team.
            ckpt=torch.load(self.direct_checkpoint,map_location='cpu',weights_only=False)
            saved=ckpt.get('args',{})
            self._device=pick_device()
            self._direct=DirectNutritionNet(backbone=saved.get('backbone','resnet18'),pretrained=False)
            self._direct.load_state_dict(ckpt['model'],strict=True)
            self._direct=self._direct.to(self._device).eval()
            self._scaler=TargetScaler.from_state_dict(ckpt['scaler']).to(self._device)
            self._size=int(saved.get('image_size',224))
        size=(self._size,self._size)
        rgb_s=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR),dtype=np.float32)/255
        depth_s=np.asarray(Image.fromarray(depth).resize(size,Image.Resampling.NEAREST),dtype=np.float32)
        valid_s=np.asarray(Image.fromarray(valid.astype(np.uint8)).resize(size,Image.Resampling.NEAREST)).astype(bool)
        rgb_t=torch.from_numpy(rgb_s.copy()).permute(2,0,1).unsqueeze(0)
        rgb_t=(rgb_t-torch.tensor(IMAGENET_MEAN).view(1,3,1,1))/torch.tensor(IMAGENET_STD).view(1,3,1,1)
        depth_t=torch.from_numpy(np.clip((depth_s-DEPTH_NEAR_M)/(DEPTH_FAR_M-DEPTH_NEAR_M),0,1)).float().unsqueeze(0).unsqueeze(0)
        valid_t=torch.from_numpy(valid_s.copy()).unsqueeze(0).unsqueeze(0)
        depth_t=torch.where(valid_t,depth_t,torch.ones_like(depth_t))
        with torch.inference_mode():
            value=self._scaler.decode(self._direct(rgb_t.to(self._device),depth_t.to(self._device),valid_t.to(self._device))).cpu()[0]
        return {k:round(float(v),1) for k,v in zip(TARGETS,value)}
