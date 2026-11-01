"""Fixed-threshold, per-image metrics; topology scores are diagnostics."""
import numpy as np
from scipy.ndimage import label


def skeletonize(mask):
    """Zhang-Suen thinning, with zero-padded borders and convergence stopping."""
    x = np.pad(np.asarray(mask, dtype=bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            ns = [x[:-2, 1:-1], x[:-2, 2:], x[1:-1, 2:], x[2:, 2:],
                  x[2:, 1:-1], x[2:, :-2], x[1:-1, :-2], x[:-2, :-2]]
            neighbors = sum(t.astype(np.uint8) for t in ns)
            transitions = sum((~ns[i] & ns[(i+1) % 8]).astype(np.uint8) for i in range(8))
            if phase == 0:
                constraint = ~(ns[0] & ns[2] & ns[4]) & ~(ns[2] & ns[4] & ns[6])
            else:
                constraint = ~(ns[0] & ns[2] & ns[6]) & ~(ns[0] & ns[4] & ns[6])
            remove = x[1:-1, 1:-1] & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & constraint
            changed |= bool(remove.any())
            x[1:-1, 1:-1][remove] = False
        if not changed:
            return x[1:-1, 1:-1]


def image_metrics(pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    p, t = int(pred.sum()), int(target.sum())
    intersection, union = int((pred & target).sum()), int((pred | target).sum())
    ps, ts = skeletonize(pred), skeletonize(target)
    if not p and not t:
        cldice = 1.0
    elif not p or not t:
        cldice = 0.0
    else:
        precision = (ps & target).sum()/max(1, ps.sum())
        recall = (ts & pred).sum()/max(1, ts.sum())
        cldice = float(2*precision*recall/max(1e-12, precision+recall))
    return {"iou": intersection/union if union else 1.0,
            "dice": 2*intersection/(p+t) if p+t else 1.0,
            "cldice": cldice,
            "component_count_error": float(abs(label(pred, np.ones((3, 3)))[1]-label(target, np.ones((3, 3)))[1]))}


def average_metrics(items):
    return {key: float(np.mean([row[key] for row in items])) for key in items[0]}
