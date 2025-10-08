import numpy as np


def ring_mask(
    w: int,
    h: int,
    r: float = 0.33,
    thick: float = 0.03,
    cx: float = 0.5,
    cy: float = 0.5,
) -> np.ndarray:
    """Create a soft ring mask centered at (cx, cy) in UV space."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xs / float(w)
    v = ys / float(h)
    d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    m = np.clip(1.0 - np.abs(d - r) / max(thick, 1e-6), 0.0, 1.0)
    return m.astype(np.float32)
