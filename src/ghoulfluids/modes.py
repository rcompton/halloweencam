# src/ghoulfluids/modes.py
"""
Alternatives to the fluid sim that the display cycles through:

- SlimeSim: physarum slime mold. Agents follow and lay down a scent trail,
  growing a branching vein network. People are food, so it crawls over them.
- WaveSim: a rippling surface. Silhouettes pulse rings outward, moving makes
  splashes, and the occasional drop lands when nobody's there.

Each exposes `output`: an RGBA texture the show shader colours like the dye.
"""
from __future__ import annotations
import math
import random

import numpy as np
import moderngl

from . import shaders as S
from .fluid import make_tex, fullscreen_quad


class SlimeSim:
    AGENT_SHADER = "FS_SLIME_AGENTS"
    PREFIX = "slime_"  # config keys for this mode

    def __init__(self, ctx: moderngl.Context, cfg, sim_w: int, sim_h: int, masks, seed=3):
        self.ctx = ctx
        self.cfg = cfg
        # The trail can be finer than the fluid grid: thin veins read better at 4K.
        k = max(1.0, self.opt("scale"))
        self.w, self.h = int(sim_w * k), int(sim_h * k)
        sim_w, sim_h = self.w, self.h
        self.mask_curr = masks[0]
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

        n = max(1, int(self.opt("agents")))
        self.agent_w = int(math.ceil(math.sqrt(n)))
        self.agent_h = int(math.ceil(n / self.agent_w))
        self.n_agents = self.agent_w * self.agent_h

        self.prog_agents = ctx.program(
            vertex_shader=S.VS, fragment_shader=getattr(S, self.AGENT_SHADER)
        )
        self.prog_deposit = ctx.program(
            vertex_shader=S.VS_SLIME_DEPOSIT, fragment_shader=S.FS_SLIME_DEPOSIT
        )
        self.prog_diffuse = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SLIME_DIFFUSE)
        self.prog_copy = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_COPY_R)
        self.vbo = fullscreen_quad(ctx)
        self.vao_agents = ctx.simple_vertex_array(self.prog_agents, self.vbo, "in_vert")
        self.vao_diffuse = ctx.simple_vertex_array(self.prog_diffuse, self.vbo, "in_vert")
        self.vao_deposit = ctx.vertex_array(self.prog_deposit, [])
        self.vao_copy = ctx.simple_vertex_array(self.prog_copy, self.vbo, "in_vert")

        self.agents_a = ctx.texture((self.agent_w, self.agent_h), 4, dtype="f4")
        self.agents_b = ctx.texture((self.agent_w, self.agent_h), 4, dtype="f4")
        for t in (self.agents_a, self.agents_b):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo_agents_a = ctx.framebuffer([self.agents_a])
        self.fbo_agents_b = ctx.framebuffer([self.agents_b])

        self.trail_a = make_tex(ctx, (sim_w, sim_h), 4, dtype="f2")
        self.trail_b = make_tex(ctx, (sim_w, sim_h), 4, dtype="f2")
        self.fbo_trail_a = ctx.framebuffer([self.trail_a])
        self.fbo_trail_b = ctx.framebuffer([self.trail_b])

        self.mask_mips = ctx.texture((sim_w, sim_h), 1, dtype="f2")
        self.mask_mips.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.mask_mips.repeat_x = self.mask_mips.repeat_y = False
        self.fbo_mask_mips = ctx.framebuffer([self.mask_mips])

        texel = (1.0 / sim_w, 1.0 / sim_h)
        self.prog_agents["texel"].value = texel
        self.prog_diffuse["texel"].value = texel
        self.prog_deposit["agent_w"].value = self.agent_w
        self.reset()

    def opt(self, name):
        return getattr(self.cfg, self.PREFIX + name)

    @property
    def output(self):
        return self.trail_a

    def set_agent_uniforms(self, p, have_mask: bool):
        cfg = self.cfg
        p["sensor_angle"].value = math.radians(cfg.slime_sensor_angle)
        p["sensor_dist"].value = cfg.slime_sensor_dist
        p["turn"].value = math.radians(cfg.slime_turn_angle)
        p["speed"].value = cfg.slime_speed
        p["food"].value = cfg.slime_food if have_mask else 0.0
        p["outline"].value = cfg.slime_outline

    def reset(self):
        """Scatter the agents over the whole screen, so slime is near whoever walks up."""
        n = self.n_agents
        agents = np.zeros((n, 4), np.float32)
        agents[:, :2] = self.rng.random((n, 2))
        agents[:, 2] = self.rng.random(n) * 2 * np.pi  # heading, radians in pixel space
        self.agents_a.write(agents.tobytes())
        for fbo in (self.fbo_trail_a, self.fbo_trail_b):
            fbo.use()
            self.ctx.clear(0, 0, 0, 0)

    def step(self, dt: float, have_mask: bool):
        self.t += dt
        steps = max(1, int(self.opt("substeps")))
        decay = math.exp(-dt / steps / max(1e-3, self.opt("decay_secs")))
        p = self.prog_agents
        self.set_agent_uniforms(p, have_mask)
        if "dt" in p:
            p["dt"].value = dt / steps
        self.prog_deposit["amount"].value = self.opt("deposit")
        self.prog_deposit["feed_glow"].value = self.opt("feed_glow") if have_mask else 0.0
        self.prog_diffuse["decay"].value = decay
        self.prog_diffuse["diffuse"].value = self.opt("diffuse")
        if have_mask:
            self.fbo_mask_mips.use()
            self.mask_curr.use(location=0)
            self.prog_copy["src"].value = 0
            self.vao_copy.render(moderngl.TRIANGLE_STRIP)
            self.mask_mips.build_mipmaps()
        for i in range(steps):
            # move agents
            self.fbo_agents_b.use()
            self.agents_a.use(location=0)
            self.trail_a.use(location=1)
            self.mask_curr.use(location=2)
            self.mask_mips.use(location=3)
            for name, unit in (("agents", 0), ("trail", 1), ("mask", 2), ("mask_mips", 3)):
                if name in p:
                    p[name].value = unit
            p["seed"].value = (self.t * 61.7 + i * 7.3) % 1000.0
            self.vao_agents.render(moderngl.TRIANGLE_STRIP)
            self.agents_a, self.agents_b = self.agents_b, self.agents_a
            self.fbo_agents_a, self.fbo_agents_b = self.fbo_agents_b, self.fbo_agents_a

            # blur + fade the trail
            self.fbo_trail_b.use()
            self.trail_a.use(location=0)
            self.prog_diffuse["trail"].value = 0
            self.vao_diffuse.render(moderngl.TRIANGLE_STRIP)
            self.trail_a, self.trail_b = self.trail_b, self.trail_a
            self.fbo_trail_a, self.fbo_trail_b = self.fbo_trail_b, self.fbo_trail_a

            # each agent adds scent where it stands
            self.fbo_trail_a.use()
            self.agents_a.use(location=0)
            self.mask_mips.use(location=1)
            self.prog_deposit["agents"].value = 0
            self.prog_deposit["mask_mips"].value = 1
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self.vao_deposit.render(moderngl.POINTS, vertices=self.n_agents)
            self.ctx.disable(moderngl.BLEND)


class EmberSim(SlimeSim):
    """
    Glowing sparks drifting on a swirling current. People push them aside, so
    they stream around silhouettes and leave each person outlined in a dark gap.
    Same machinery as the slime: agents in a texture, points into a fading trail.
    """

    AGENT_SHADER = "FS_EMBER_AGENTS"
    PREFIX = "ember_"

    def set_agent_uniforms(self, p, have_mask: bool):
        cfg = self.cfg
        p["t"].value = self.t
        p["flow"].value = cfg.ember_flow
        p["rise"].value = cfg.ember_rise
        p["push"].value = cfg.ember_push if have_mask else 0.0
        p["inertia"].value = cfg.ember_follow
        p["aspect"].value = self.w / float(self.h)

    def reset(self):
        n = self.n_agents
        agents = np.zeros((n, 4), np.float32)  # rg = position (UV), ba = velocity (UV/s)
        agents[:, :2] = self.rng.random((n, 2))
        self.agents_a.write(agents.tobytes())
        for fbo in (self.fbo_trail_a, self.fbo_trail_b):
            fbo.use()
            self.ctx.clear(0, 0, 0, 0)


class WaveSim:
    def __init__(self, ctx: moderngl.Context, cfg, sim_w: int, sim_h: int, masks, seed=5):
        self.ctx = ctx
        self.cfg = cfg
        self.w, self.h = sim_w, sim_h
        self.mask_curr, self.mask_prev = masks
        self.rng = random.Random(seed)
        self.t = 0.0
        self.next_drop = 0.0

        self.prog_step = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_WAVE_STEP)
        self.prog_drop = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_WAVE_DROP)
        self.prog_shade = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_WAVE_SHADE)
        self.vbo = fullscreen_quad(ctx)
        self.vao_step = ctx.simple_vertex_array(self.prog_step, self.vbo, "in_vert")
        self.vao_drop = ctx.simple_vertex_array(self.prog_drop, self.vbo, "in_vert")
        self.vao_shade = ctx.simple_vertex_array(self.prog_shade, self.vbo, "in_vert")

        # r = height now, g = height one step ago
        self.h_a = make_tex(ctx, (sim_w, sim_h), 2, dtype="f4")
        self.h_b = make_tex(ctx, (sim_w, sim_h), 2, dtype="f4")
        self.fbo_a = ctx.framebuffer([self.h_a])
        self.fbo_b = ctx.framebuffer([self.h_b])
        self.out = make_tex(ctx, (sim_w, sim_h), 4, dtype="f2")
        self.fbo_out = ctx.framebuffer([self.out])

        texel = (1.0 / sim_w, 1.0 / sim_h)
        self.prog_step["texel"].value = texel
        self.prog_shade["texel"].value = texel
        self.prog_drop["aspect"].value = sim_w / float(sim_h)
        self.reset()

    @property
    def output(self):
        return self.out

    def reset(self):
        for fbo in (self.fbo_a, self.fbo_b, self.fbo_out):
            fbo.use()
            self.ctx.clear(0, 0, 0, 0)

    def drop(self, u: float, v: float, amount: float, radius: float):
        self.fbo_b.use()
        self.h_a.use(location=0)
        self.prog_drop["field"].value = 0
        self.prog_drop["point"].value = (u, v)
        self.prog_drop["amount"].value = amount
        self.prog_drop["radius"].value = radius
        self.vao_drop.render(moderngl.TRIANGLE_STRIP)
        self._swap()

    def _swap(self):
        self.h_a, self.h_b = self.h_b, self.h_a
        self.fbo_a, self.fbo_b = self.fbo_b, self.fbo_a

    def step(self, dt: float, have_mask: bool):
        cfg = self.cfg
        self.t += dt

        # raindrops keep it alive when nobody's there (and a few while they are)
        self.next_drop -= dt * (1.0 if not have_mask else 0.3)
        if self.next_drop <= 0.0:
            self.next_drop = self.rng.uniform(0.5, 1.5) * cfg.wave_drop_interval
            self.drop(
                self.rng.uniform(0.1, 0.9),
                self.rng.uniform(0.15, 0.9),
                self.rng.uniform(0.6, 1.0) * cfg.wave_drop_amp,
                self.rng.uniform(0.6, 1.4) * 0.02,
            )

        steps = max(1, int(cfg.wave_substeps))
        p = self.prog_step
        p["c2"].value = cfg.wave_speed
        p["damp"].value = math.exp(-1.0 / max(1.0, cfg.wave_damp_steps))
        p["have_mask"].value = 1 if have_mask else 0
        p["move_amp"].value = cfg.wave_move_amp / steps
        p["pulse_amp"].value = cfg.wave_pulse_amp
        for i in range(steps):
            ts = self.t + dt * (i / steps - 1.0)
            p["pulse"].value = math.sin(2 * math.pi * cfg.wave_pulse_hz * ts)
            self.fbo_b.use()
            self.h_a.use(location=0)
            self.mask_curr.use(location=1)
            self.mask_prev.use(location=2)
            p["field"].value = 0
            p["mask_curr"].value = 1
            p["mask_prev"].value = 2
            self.vao_step.render(moderngl.TRIANGLE_STRIP)
            self._swap()

        # height -> brightness (crests and glints) for the show shader
        self.fbo_out.use()
        self.h_a.use(location=0)
        self.prog_shade["field"].value = 0
        self.prog_shade["gain"].value = cfg.wave_gain
        self.vao_shade.render(moderngl.TRIANGLE_STRIP)


class ModeCycler:
    """
    Which simulation is showing. Holds each mode for `dwell` seconds, then
    crossfades to the next over `fade` seconds.

    step() returns (current, upcoming, mix); mix 0 = all current.
    """

    def __init__(self, modes, dwell: float, fade: float):
        self.modes = list(modes) or ["fluid"]
        self.dwell = dwell
        self.fade = fade
        self.i = 0
        self.t = 0.0

    @property
    def current(self) -> str:
        return self.modes[self.i]

    @property
    def upcoming(self) -> str:
        return self.modes[(self.i + 1) % len(self.modes)]

    def skip(self):
        """Start fading to the next mode now."""
        if len(self.modes) > 1:
            self.t = max(self.t, self.dwell)

    def step(self, dt: float):
        if len(self.modes) < 2:
            return self.current, self.current, 0.0
        self.t += dt
        if self.t >= self.dwell + self.fade:
            self.i = (self.i + 1) % len(self.modes)
            self.t = 0.0
        mix = max(0.0, (self.t - self.dwell) / max(1e-3, self.fade))
        return self.current, self.upcoming, min(1.0, mix)
