# tests/test_fluid_step.py

import numpy as np
from ghoulfluids.fluid import FluidSim
from ghoulfluids.utils import ring_mask


def tex_to_np(tex, comps: int) -> np.ndarray:
    """Read a ModernGL texture into a float32 numpy array.
    Works for FP16 ('f2') and FP32 ('f4') textures (and uint8 'f1' for camera).
    """
    dtype_map = {
        "f1": np.uint8,  # e.g., camera RGB8
        "f2": np.float16,  # half-float
        "f4": np.float32,  # float
    }
    # Some ModernGL versions expose Texture.dtype; fall back to f4 if missing.
    dt = dtype_map.get(getattr(tex, "dtype", "f4"), np.float32)

    raw = tex.read()  # bytes
    arr = np.frombuffer(raw, dtype=dt)

    expected_elems = tex.height * tex.width * comps
    if arr.size != expected_elems:
        raise AssertionError(
            f"Unexpected texture element count: got {arr.size}, "
            f"expected {expected_elems} ({tex.width}x{tex.height}x{comps}, dtype={getattr(tex, 'dtype', 'f4')})"
        )

    arr = arr.reshape(tex.height, tex.width, comps)
    return arr.astype(np.float32, copy=False)


def test_mask_force_changes_velocity(ctx, small_cfg):
    sim = FluidSim(ctx, small_cfg)

    # Upload a ring mask at sim resolution to create strong edges
    m = ring_mask(sim.sim_w, sim.sim_h)
    sim.upload_mask(m)

    # Step once with mask forces enabled
    sim.step(0.016, have_mask=True)

    # Velocity should change away from exactly zero
    v = tex_to_np(sim.vel_a, 2)
    max_mag = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2).max()
    assert max_mag > 1e-6, "Velocity did not change after applying mask force"


def test_advects_dye_even_without_mask(ctx, small_cfg):
    sim = FluidSim(ctx, small_cfg)

    # A few steps without a mask should be stable (no exceptions) and yield a valid dye tex
    for _ in range(3):
        sim.step(0.016, have_mask=False)

    d = tex_to_np(sim.dye_a, 4)
    # Dye runs at render-scaled resolution (sim.dye_w, sim.dye_h)
    assert d.shape == (sim.dye_h, sim.dye_w, 4)
