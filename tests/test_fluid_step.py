import numpy as np
from ghoulfluids.fluid import FluidSim


def ring_mask(w, h, r=0.33, thick=0.03, cx=0.5, cy=0.5):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xs / w
    v = ys / h
    d = np.sqrt((u - cx) ** 2 + (v - cy) ** 2)
    m = np.clip(1.0 - np.abs(d - r) / thick, 0.0, 1.0)
    return m.astype(np.float32)


def tex_to_np(tex, comps):
    arr = np.frombuffer(tex.read(), dtype=np.float32)
    return arr.reshape(tex.height, tex.width, comps)


def test_mask_force_changes_velocity(ctx, small_cfg):
    sim = FluidSim(ctx, small_cfg)
    m = ring_mask(sim.sim_w, sim.sim_h)
    sim.upload_mask(m)
    sim.step(0.016, have_mask=True)
    v = tex_to_np(sim.vel_a, 2)
    max_mag = np.sqrt((v[..., 0] ** 2 + v[..., 1] ** 2)).max()
    assert max_mag > 1e-6, "Velocity did not change after applying mask force"


def test_advects_dye_even_without_mask(ctx, small_cfg):
    sim = FluidSim(ctx, small_cfg)
    for _ in range(3):
        sim.step(0.016, have_mask=False)
    d = tex_to_np(sim.dye_a, 4)
    assert d.shape == (small_cfg.height, small_cfg.width, 4)
