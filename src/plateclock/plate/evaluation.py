"""Candidate correctness: greedy one-to-one same-group mask IoU >= 0.5."""
import numpy as np

def match_regions(predicted, gold, valid, threshold=0.5):
    candidates = []
    for i, pred in enumerate(predicted):
        for j, truth in enumerate(gold):
            if pred.label != truth.label:
                continue
            a, b = pred.mask & valid, truth.mask & valid
            union = np.count_nonzero(a | b)
            iou = np.count_nonzero(a & b)/union if union else 0.
            if iou >= threshold:
                candidates.append((iou, i, j))
    used_p, used_g = set(), set()
    matches = [0] * len(predicted)
    for iou, i, j in sorted(candidates, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); matches[i] = 1
    return matches

def threshold_table(scores, correct):
    scores, correct = np.asarray(scores), np.asarray(correct)
    result = []
    for threshold in np.linspace(0, 1, 21):
        keep = scores >= threshold
        n = int(keep.sum())
        result.append(dict(threshold=float(threshold), retained=n,
                           coverage=n/len(scores) if len(scores) else 0.,
                           precision=float(correct[keep].mean()) if n else None))
    return result
