"""Synthetic integration check only. Does NOT measure food recognition accuracy."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from plateclock.pipeline import run, public_result

class SyntheticSegmenter:
    def predict(self, image):
        labels = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(labels, (160, 100), 22, 66, -1)  # rice
        cv2.circle(labels, (205, 185), 22, 48, -1)  # chicken duck
        return labels, np.full(labels.shape, .95, np.float32)

if __name__ == '__main__':
    image = np.full((321, 321, 3), 25, np.uint8)
    cv2.circle(image, (160,160), 125, (235,235,235), -1)
    result = run(image, SyntheticSegmenter())
    assert result['status'] == 'ok', result['reason']
    assert len(result['items']) == 2
    out = Path('results/week2/synthetic'); out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result['overlay']).save(out/'overlay.png')
    (out/'result.json').write_text(json.dumps(public_result(result), indent=2))
    print('SYNTHETIC TEST ONLY:', result['text'])
