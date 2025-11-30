# core/metrics.py
import numpy as np
import pandas as pd

def wmape(y_true, y_pred, eps=1e-8):
    numer = np.sum(np.abs(y_true - y_pred))
    denom = np.sum(np.abs(y_true)) + eps
    return 100.0 * numer / denom

def smape(y_true, y_pred, eps=1e-6):
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    return 100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom)

def pinball_loss(y_true, y_pred, q):
    # q in (0,1), lower loss is better
    diff = y_true - y_pred
    return np.mean(np.maximum(q*diff, (q-1)*diff))
