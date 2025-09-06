import numpy as np


def luminance(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def reinhard(rgb):
    return rgb / (1.0 + rgb)


def hable_filmic(rgb, exposure=1.0, white_point=11.2):
    # Uncharted 2 tone mapping
    a = 0.15
    b = 0.50
    c = 0.10
    d = 0.20
    e = 0.02
    f = 0.30

    def curve(x):
        return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f

    x = np.maximum(rgb * exposure, 0.0)
    W = white_point
    return np.clip(curve(x) / curve(W), 0.0, 1.0)


def auto_exposure_from_percentile(rgb, percentile=99.5, target=3.0):
    # Compute exposure scale so that luminance percentile maps to target pre-tonemap
    L = luminance(rgb)
    q = float(np.quantile(L, min(max(percentile, 0.0), 100.0) / 100.0))
    if not np.isfinite(q) or q <= 1e-6:
        return 1.0
    return target / q

