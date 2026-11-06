"""SegFormer-B0 inference from locally trained weights only; no silent oracle mode."""
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

class FoodSegmenter:
    def __init__(self, checkpoint='checkpoints/foodseg-b0'):
        from pathlib import Path
        if not (Path(checkpoint)/'config.json').exists():
            raise FileNotFoundError(f'No trained food checkpoint at {checkpoint}; run scripts/train_foodseg.py first')
        self.device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model=SegformerForSemanticSegmentation.from_pretrained(checkpoint, local_files_only=True).to(self.device).eval()
        if self.model.config.num_labels != 104:
            raise ValueError('Expected the Week 1 104-class checkpoint')
        self.processor=SegformerImageProcessor.from_pretrained(checkpoint, local_files_only=True)
    @torch.no_grad()
    def predict(self, image):
        from torch.nn.functional import interpolate
        inputs=self.processor(images=Image.fromarray(image),return_tensors='pt').to(self.device)
        logits=self.model(**inputs).logits
        logits=interpolate(logits,size=image.shape[:2],mode='bilinear',align_corners=False)
        prob=logits.softmax(1)
        return prob.argmax(1)[0].cpu().numpy().astype(np.uint8),prob.max(1).values[0].cpu().numpy()
