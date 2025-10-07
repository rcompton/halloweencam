import numpy as np
from ghoulfluids.utils import ring_mask


def test_ring_mask_properties():
    w, h = 64, 64
    mask = ring_mask(w, h)
    assert mask.shape == (h, w), "Mask shape should match input dimensions"
    assert mask.dtype == np.float32, "Mask dtype should be float32"
    assert np.all(mask >= 0.0) and np.all(
        mask <= 1.0
    ), "Mask values should be in [0, 1] range"


def test_ring_mask_values():
    w, h = 64, 64
    radius = 0.25
    thickness = 0.1
    mask = ring_mask(w, h, r=radius, thick=thickness, cx=0.5, cy=0.5)

    # Pixel at the center of the ring should be close to 1.0
    center_x, center_y = int(w * 0.5), int(h * (0.5 - radius))
    assert np.isclose(
        mask[center_y, center_x], 1.0
    ), "Pixel on the ring should be close to 1.0"

    # Pixel far from the ring should be 0.0
    assert np.isclose(mask[0, 0], 0.0), "Pixel far from the ring should be 0.0"
