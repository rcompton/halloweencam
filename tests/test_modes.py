import numpy as np
import pytest

from ghoulfluids.config import AppConfig
from ghoulfluids.fluid import FluidSim
from ghoulfluids.modes import EmberSim, ModeCycler, SlimeSim, WaveSim


@pytest.fixture(scope="module")
def mode_cfg():
    return AppConfig(width=512, height=288, sim_scale=0.5, slime_agents=20_000)


def make(ctx, cfg, cls):
    sim = FluidSim(ctx, cfg)
    return sim, cls(ctx, cfg, sim.sim_w, sim.sim_h, (sim.mask_curr, sim.mask_prev))


def read(tex, comps, dtype):
    return np.frombuffer(tex.read(), dtype).astype(np.float32).reshape(tex.height, tex.width, comps)


def test_slime_lays_trail_and_agents_stay_on_screen(ctx, mode_cfg):
    _, slime = make(ctx, mode_cfg, SlimeSim)
    for _ in range(60):
        slime.step(1 / 30, False)
    trail = read(slime.output, 4, np.float16)
    assert np.isfinite(trail).all()
    assert trail[..., 0].max() > 0.1
    agents = read(slime.agents_a, 4, np.float32).reshape(-1, 4)
    assert ((agents[:, :2] >= 0.0) & (agents[:, :2] <= 1.0)).all()


def test_slime_gathers_on_people(ctx, mode_cfg):
    sim, slime = make(ctx, mode_cfg, SlimeSim)
    m = np.zeros((sim.sim_h, sim.sim_w), np.float32)
    x0, x1, y0, y1 = sim.sim_w * 5 // 8, sim.sim_w * 7 // 8, sim.sim_h // 4, sim.sim_h * 3 // 4
    m[y0:y1, x0:x1] = 1.0
    sim.upload_mask(m)

    def frac_on_person():
        a = read(slime.agents_a, 4, np.float32).reshape(-1, 4)
        x, y = a[:, 0] * sim.sim_w, a[:, 1] * sim.sim_h
        return ((x > x0) & (x < x1) & (y > y0) & (y < y1)).mean()

    before = frac_on_person()
    for _ in range(90):
        slime.step(1 / 30, True)
    assert frac_on_person() > 2 * before


def test_embers_are_pushed_off_people(ctx, mode_cfg):
    sim, embers = make(ctx, mode_cfg, EmberSim)
    m = np.zeros((sim.sim_h, sim.sim_w), np.float32)
    x0, x1, y0, y1 = sim.sim_w * 3 // 8, sim.sim_w * 5 // 8, sim.sim_h // 4, sim.sim_h * 3 // 4
    m[y0:y1, x0:x1] = 1.0
    sim.upload_mask(m)

    def frac_on_person():
        a = read(embers.agents_a, 4, np.float32).reshape(-1, 4)
        x, y = a[:, 0] * sim.sim_w, a[:, 1] * sim.sim_h
        return ((x > x0) & (x < x1) & (y > y0) & (y < y1)).mean()

    before = frac_on_person()
    for _ in range(60):
        embers.step(1 / 30, True)
    assert frac_on_person() < 0.5 * before
    assert np.isfinite(read(embers.output, 4, np.float16)).all()


def test_wave_drop_spreads_and_settles(ctx, mode_cfg):
    _, wave = make(ctx, mode_cfg, WaveSim)
    wave.next_drop = 1e9  # no random drops
    wave.drop(0.5, 0.5, 1.0, 0.03)
    for _ in range(10):
        wave.step(1 / 30, False)
    h = read(wave.h_a, 2, np.float32)[..., 0]
    assert np.isfinite(h).all()
    ring = np.abs(h).max()
    assert ring > 0.01
    for _ in range(600):
        wave.step(1 / 30, False)
    assert np.abs(read(wave.h_a, 2, np.float32)[..., 0]).max() < ring


def test_wave_rings_around_people_stay_bounded(ctx, mode_cfg):
    sim, wave = make(ctx, mode_cfg, WaveSim)
    wave.next_drop = 1e9
    m = np.zeros((sim.sim_h, sim.sim_w), np.float32)
    for i in range(150):  # a person walking across
        m[:] = 0.0
        x = 20 + i
        m[sim.sim_h // 4 : sim.sim_h * 3 // 4, x : x + 30] = 1.0
        sim.upload_mask(m)
        wave.step(1 / 30, True)
    h = read(wave.h_a, 2, np.float32)[..., 0]
    assert np.isfinite(h).all()
    assert 0.01 < np.abs(h).max() <= 4.0


def test_mode_cycler_dwells_then_fades():
    c = ModeCycler(["fluid", "slime", "wave"], dwell=10.0, fade=2.0)
    assert c.step(5.0) == ("fluid", "slime", 0.0)
    cur, nxt, mix = c.step(6.0)
    assert (cur, nxt) == ("fluid", "slime") and mix == pytest.approx(0.5)
    assert c.step(1.5)[:2] == ("slime", "wave")
    c.skip()
    assert c.step(0.0)[0] == "slime"
    assert c.step(2.0)[0] == "wave"
    assert c.step(30.0)[0] == "fluid"  # wraps around


def test_mode_cycler_single_mode_never_fades():
    c = ModeCycler(["slime"], dwell=1.0, fade=1.0)
    c.skip()
    assert c.step(100.0) == ("slime", "slime", 0.0)


def test_show_pass_crossfades_modes(ctx, mode_cfg):
    sim, slime = make(ctx, mode_cfg, SlimeSim)
    for _ in range(30):
        slime.step(1 / 30, False)
    tex = ctx.texture((mode_cfg.width, mode_cfg.height), 3)
    fbo = ctx.framebuffer([tex])

    def frame(mix):
        fbo.use()
        sim.set_show_source(sim.dye_a, slime.output, mix)
        sim.render_split(False)
        return np.frombuffer(fbo.read(components=3), np.uint8).astype(np.float32)

    a, half, b = frame(0.0), frame(0.5), frame(1.0)
    assert np.abs(a - b).mean() > 1.0
    assert np.abs(half - a).mean() > 0.2 and np.abs(half - b).mean() > 0.2
