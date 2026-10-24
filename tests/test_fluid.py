# tests/test_fluid_step.py

import numpy as np
from unittest.mock import MagicMock
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


def test_set_palette_blend(ctx, small_cfg):
    sim = FluidSim(ctx, small_cfg)

    # Call the method
    sim.set_palette_blend(1, 2, 0.5)

    # Assert that the instance attributes were set correctly
    assert sim.palette_a == 1
    assert sim.palette_b == 2
    assert sim.palette_mix == 0.5


def _wall_hugging_mask(sim):
    # A body against the left side that stops 2px short of the frame edge, the way
    # YOLO masks do. That gap used to read as a person's outline along the wall.
    m = np.zeros((sim.sim_h, sim.sim_w), np.float32)
    # Full height (2px short top and bottom too), so its only real outline is the
    # inner vertical edge.
    m[2:-2, 2 : sim.sim_w // 8] = 1.0
    return m


def test_mask_edge_at_frame_border_adds_no_dye(ctx, small_cfg):
    from dataclasses import replace

    # Real sim width (YOLO's 2px gap is ~0.2% of it); the tiny test grid would make 2px look wide.
    cfg = replace(small_cfg, width=2048, height=1152, edge_dye_strength=0.5,
                  background_velocity=0.0)
    sim = FluidSim(ctx, cfg)
    for fbo in (sim.fbo_dye_a, sim.fbo_dye_b):
        fbo.use()
        ctx.clear(0, 0, 0, 1)
    sim.upload_mask(_wall_hugging_mask(sim))
    for _ in range(10):
        sim.step(0.016, have_mask=True)
    d = tex_to_np(sim.dye_a, 4)[..., :3].sum(axis=2)
    wall = d[:, :2]
    inner_edge = d[:, sim.dye_w // 8 - 2 : sim.dye_w // 8 + 2]
    assert inner_edge.max() > 0.1, "real outline should still emit dye"
    assert wall.max() < 0.15 * inner_edge.max(), "frame border must not act as an outline"


def test_edge_dye_is_capped(ctx, small_cfg):
    from dataclasses import replace

    cfg = replace(small_cfg, edge_dye_strength=1.0, edge_dye_cap=1.0, background_velocity=0.0,
                  edge_normal_amp=0.0, edge_tangent_amp=0.0)
    sim = FluidSim(ctx, cfg)
    m = ring_mask(sim.sim_w, sim.sim_h)
    sim.upload_mask(m)
    for _ in range(60):
        sim.step(0.016, have_mask=True)
    d = tex_to_np(sim.dye_a, 4)[..., :3]
    luma = d @ np.array([0.299, 0.587, 0.114], np.float32)
    assert luma.max() < 2.0, f"edge dye ran away: max luma {luma.max():.2f}"
