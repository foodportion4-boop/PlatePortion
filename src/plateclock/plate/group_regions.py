"""Broad food groups, keeping disconnected portions spatially separate."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Region:
    label: str
    mask: np.ndarray
    area_fraction: float
    share: float
    probability: float

def class_groups(names, config):
    mapping = {}
    for group, members in config['groups'].items():
        for name in members:
            if name in mapping:
                raise ValueError(f'Duplicate grouping entry: {name}')
            mapping[name] = group
    unknown = set(mapping) - set(names.values())
    if unknown:
        raise ValueError(f'Grouping names absent from class map: {sorted(unknown)}')
    return {i: mapping.get(name, 'other food') for i, name in names.items() if i not in (0, 255)}

def group_regions(labels, probabilities, plate, names, config):
    if labels.ndim != 2 or labels.shape != probabilities.shape or labels.shape != plate.shape:
        raise ValueError('Labels, probabilities and plate must have identical 2D shapes')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Probabilities must be finite and in [0,1]')
    unknown = set(np.unique(labels)) - set(names) - {255}
    if unknown:
        raise ValueError(f'Unknown class IDs: {unknown}')
    valid = plate.astype(bool) & (labels != 0) & (labels != 255)
    total, plate_area = int(valid.sum()), int(np.count_nonzero(plate))
    if not total or not plate_area:
        return []
    mapping = class_groups(names, config)
    result = []
    # Union touching classes in a group; never average separated portions into a false location.
    for group in config['groups']:
        ids = [i for i, g in mapping.items() if g == group]
        binary = (valid & np.isin(labels, ids)).astype(np.uint8)
        count, components = cv2.connectedComponents(binary, connectivity=8)
        for index in range(1, count):
            mask = components == index
            size = int(mask.sum())
            fraction = size / plate_area
            if fraction < config['geometry']['minimum_plate_area_fraction']:
                continue
            result.append(Region(group, mask, fraction, size / total,
                                 float(probabilities[mask].mean())))
    return sorted(result, key=lambda r: (-r.area_fraction, r.label))
