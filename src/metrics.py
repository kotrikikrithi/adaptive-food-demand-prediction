"""
Evaluation metrics. y and pred are in normalised units (1.0 = the outlet's average day)
unless stated otherwise.
"""
import numpy as np

import config


def wape(y, pred):
    """Weighted absolute percentage error: total |error| / total demand. 0.10 = 10% off."""
    return float(np.abs(y - pred).sum() / max(np.abs(y).sum(), 1e-9))


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def group_bias(y, pred, groups):
    """Relative bias per outlet type: +0.05 means that type is over-predicted by 5%."""
    return {g: float((pred[groups == g] - y[groups == g]).sum() / max(y[groups == g].sum(), 1e-9))
            for g in config.GROUPS if (groups == g).any()}


def bias_gap(y, pred, groups):
    """Calibration gap between types: max bias - min bias (0 = equally calibrated)."""
    b = group_bias(y, pred, groups)
    return float(max(b.values()) - min(b.values()))


def coverage_by_group(y, lo, hi, groups):
    """Share of true values that fall inside the prediction interval, per type."""
    inside = (y >= lo) & (y <= hi)
    return {g: float(inside[groups == g].mean()) for g in config.GROUPS if (groups == g).any()}


def waste_and_shortage(demand_meals, pred_meals):
    """Meals over-prepared (waste) and meals short (shortage)."""
    diff = pred_meals - demand_meals
    return float(diff.clip(min=0).sum()), float((-diff).clip(min=0).sum())


def fitness(acc, var, cost, fair):
    """Single score (LOWER is better) = weighted sum of the four objectives."""
    return float(config.W_ACCURACY * acc + config.W_VARIANCE * var
                 + config.W_COST * cost + config.W_FAIRNESS * fair)
