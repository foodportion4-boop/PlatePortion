"""Evaluate contour support for the selected Week 1 ellipse without changing its API."""
import cv2
import numpy as np

def quality(rgb, fit, scale=0.08):
    if scale <= 0:
        raise ValueError('Residual scale must be positive')
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (7, 7), 1.5)
    contours, _ = cv2.findContours(cv2.Canny(gray, 35, 100), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    t = np.deg2rad(fit.rotation)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    best, best_residual = 0., float('inf')
    for c in contours:
        if len(c) < 20:
            continue
        local = (c[:, 0, :].astype(float) - fit.center) @ rot
        scaled = local / (np.array(fit.axes) / 2)
        errors = np.abs(np.linalg.norm(scaled, axis=1) - 1.)
        residual = float(np.median(errors))
        # Reward support around the whole rim, not just a small coincident arc.
        angles = np.arctan2(scaled[:, 1], scaled[:, 0])
        near = errors <= 2 * scale
        bins = np.floor((angles[near] + np.pi) / (2 * np.pi) * 36).astype(int) % 36
        coverage = len(np.unique(bins)) / 36
        score = float(np.exp(-residual / scale) * coverage)
        if score > best:
            best, best_residual = score, residual
    return best, best_residual

def fully_visible(fit, shape, margin=2):
    h, w = shape[:2]
    a, b = np.array(fit.axes) / 2
    t = np.deg2rad(fit.rotation)
    dx = np.hypot(a * np.cos(t), b * np.sin(t))
    dy = np.hypot(a * np.sin(t), b * np.cos(t))
    x, y = fit.center
    return x-dx >= margin and y-dy >= margin and x+dx < w-margin and y+dy < h-margin
