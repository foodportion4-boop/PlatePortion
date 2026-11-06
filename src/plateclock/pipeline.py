"""Week 2 plate pipeline. Ground truth is never accepted as an inference input."""
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
import yaml
from plateclock.io.class_names import CLASS_NAMES
from plateclock.plate.detect_plate import detect_plate, ellipse_mask
from plateclock.plate.rectify import rectification, warp
from plateclock.plate.geometry import ItemPlacement, angle_at, hour_at
from plateclock.plate.fit_quality import quality, fully_visible
from plateclock.plate.group_regions import group_regions
from plateclock.plate.confidence import decide
from plateclock.language.style_guide import load_style
from plateclock.language.plate_nlg import describe
from plateclock.language.style_lint import require_valid

def load_config(path='configs/week2.yaml'):
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))

def placement(region, center, radius, cfg):
    ys, xs = np.where(region.mask)
    angle = angle_at(float(xs.mean()), float(ys.mean()), *center)
    radial = float(np.hypot(xs.mean()-center[0], ys.mean()-center[1])/radius)
    zone = 'center' if radial < cfg['center_boundary'] else 'middle' if radial < cfg['rim_boundary'] else 'near the rim'
    # Correct shortest arc containing at least 95% of region pixels, including wraparound.
    angles = np.sort(np.degrees(np.arctan2(xs-center[0], center[1]-ys)) % 360)
    n = len(angles); k = max(1, int(np.ceil(.95*n)))
    ext = np.concatenate([angles, angles+360]); starts = np.arange(n)
    j = int(np.argmin(ext[starts+k-1]-ext[starts]))
    span = (hour_at(ext[j]), hour_at(ext[j+k-1]))
    return ItemPlacement(region.label, hour_at(angle), zone, region.share, span, angle)

def run(image, segmenter, config=None, style=None, mode='full'):
    config, style = config or load_config(), style or load_style()
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected RGB uint8 image')
    labels, probabilities = segmenter.predict(image)
    if labels.shape != image.shape[:2] or probabilities.shape != labels.shape:
        raise ValueError('Segmenter output must match original image dimensions')
    try:
        fit = detect_plate(image, (labels > 0) & (labels != 255))
    except ValueError as exc:
        return failure(image, style, str(exc))
    q, residual = quality(image, fit, config['plate']['residual_scale'])
    if (not fully_visible(fit, image.shape) or fit.axes[1]/fit.axes[0] < config['plate']['minimum_axis_ratio']
            or q < config['plate']['minimum_fit_quality']):
        return failure(image, style, 'Ellipse clipped, too oblique, or weak rim support')
    affine, radius = rectification(fit, image.shape)
    # Fixed Week 1 canvas can crop the corrected circle. Refuse such cases.
    x, y = fit.center; h, w = labels.shape
    if x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
        return failure(image, style, 'Rectified circle would exceed the canvas')
    rect_labels = warp(labels, affine, is_mask=True)
    rect_prob = np.clip(warp(probabilities.astype(np.float32), affine), 0, 1)
    plate = warp(ellipse_mask(image.shape, fit).astype(np.uint8), affine, is_mask=True).astype(bool)
    regions = group_regions(rect_labels, rect_prob, plate, CLASS_NAMES, config)
    decisions = [decide(r.probability, r.area_fraction, q, config['confidence']) for r in regions]
    selected = [(r, d) for r, d in zip(regions, decisions) if d.action != 'omit'][:config['geometry']['max_items']]
    items = [placement(r, fit.center, radius, config['geometry']) for r, _ in selected]
    text = describe(items, [d.action for _, d in selected], mode, style)
    require_valid(text, mode, style)
    overlay = warp(image, affine)
    cv2.circle(overlay, tuple(map(int, fit.center)), int(radius), (30, 200, 30), 2)
    for r, d in selected:
        ys, xs = np.where(r.mask)
        cv2.putText(overlay, r.label, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 30, 30), 1)
    return dict(status='ok' if items else 'no_confident_food', reason='', text=text, items=items,
                decisions=[asdict(d) for _, d in selected], overlay=overlay, fit=fit,
                fit_quality=q, ellipse_residual=residual, regions=regions, all_decisions=decisions,
                affine=affine, plate=plate, labels=rect_labels, radius=radius)

def failure(image, style, reason):
    require_valid(style['failure'], style=style)
    return dict(status='plate_failure', reason=reason, text=style['failure'], items=[],
                decisions=[], overlay=image.copy(), regions=[], all_decisions=[])

def public_result(result):
    return {k: ([asdict(p) for p in v] if k == 'items' else v)
            for k, v in result.items() if k in {'status','reason','text','items','decisions','fit_quality','ellipse_residual'}}
