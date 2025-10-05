# src/ghoulfluids/ambient.py
from __future__ import annotations
import moderngl
import math
import random
import numpy as np


class AmbientController:
    """
    Drives the fluid when no mask is detected.
    - A set of drifting 'emitters' inject small amounts of velocity and dye.
    - Periodic vortex dipoles seed curls to keep motion interesting.
    """

    def __init__(self, cfg, seed: int = 42):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.emitters = []
        self.vortex_timer = 0.0
        self._build_emitters()

    def _build_emitters(self):
        self.emitters.clear()
        m = self.cfg.ambient_margin
        for _ in range(self.cfg.ambient_emitters):
            # start away from walls
            x = m + (1 - 2 * m) * self.rng.random()
            y = m + (1 - 2 * m) * self.rng.random()
            ang = self.rng.random() * 2.0 * math.pi
            vx = self.cfg.ambient_speed * math.cos(ang)
            vy = self.cfg.ambient_speed * math.sin(ang)
            # halloweeny color between orange & purple
            k = self.rng.random()
            orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
            purple = np.array([0.45, 0.0, 0.6], dtype=np.float32)
            col = (1.0 - k) * orange + k * purple
            self.emitters.append(
                {
                    "pos": np.array([x, y], dtype=np.float32),
                    "vel": np.array([vx, vy], dtype=np.float32),
                    "col": col.astype(np.float32),
                }
            )

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.vortex_timer = 0.0
        self._build_emitters()

    def _gauss2(self, sigma: float):
        # 2D gaussian using Python's random for determinism
        return np.array(
            [self.rng.gauss(0.0, sigma), self.rng.gauss(0.0, sigma)], dtype=np.float32
        )

    def step(self, dt: float, sim):
        """
        Inject ambient velocity & dye BEFORE the solver step so it affects this frame.
        `sim` is your FluidSim instance.
        """
        cfg = self.cfg

        # 1) Update emitter positions and inject small splats
        for e in self.emitters:
            # jitter velocity slightly
            if cfg.ambient_jitter > 0.0:
                j = self._gauss2(cfg.ambient_speed * cfg.ambient_jitter) * dt
                e["vel"] += j

            # clamp speed
            sp = float(np.linalg.norm(e["vel"]))
            if sp > cfg.ambient_speed:
                e["vel"] *= cfg.ambient_speed / (sp + 1e-6)

            # integrate position
            e["pos"] += e["vel"] * dt

            # soft bounce at borders
            for k in (0, 1):
                lo = cfg.ambient_margin
                hi = 1.0 - cfg.ambient_margin
                if e["pos"][k] < lo:
                    e["pos"][k] = lo
                    e["vel"][k] = abs(e["vel"][k])
                elif e["pos"][k] > hi:
                    e["pos"][k] = hi
                    e["vel"][k] = -abs(e["vel"][k])

            u, v = float(e["pos"][0]), float(e["pos"][1])
            vx, vy = float(e["vel"][0]), float(e["vel"][1])

            # (a) inject a bit of velocity along travel direction
            sim.fbo_vel_b.use()
            sim.vel_a.use(location=0)
            sim.prog_splat["field"].value = 0
            sim.prog_splat["point"].value = (u, v)
            sim.prog_splat["value"].value = (
                vx * cfg.ambient_vel_amp,
                vy * cfg.ambient_vel_amp,
                0.0,
            )
            sim.prog_splat["radius"].value = cfg.ambient_radius
            sim.vao_splat.render(moderngl.TRIANGLE_STRIP)
            sim.swap_vel()

            # (b) wisp of dye
            col = (e["col"] * cfg.ambient_dye).astype(np.float32)
            sim.fbo_dye_b.use()
            sim.dye_a.use(location=0)
            sim.prog_splat["field"].value = 0
            sim.prog_splat["point"].value = (u, v)
            sim.prog_splat["value"].value = (
                float(col[0]),
                float(col[1]),
                float(col[2]),
            )
            sim.prog_splat["radius"].value = cfg.ambient_radius * 0.9
            sim.vao_splat.render(moderngl.TRIANGLE_STRIP)
            sim.swap_dye()

        # 2) Occasional vortex dipole to seed curls
        self.vortex_timer += dt
        if self.vortex_timer >= cfg.vortex_interval:
            self.vortex_timer = 0.0
            cx = cfg.ambient_margin + (1 - 2 * cfg.ambient_margin) * self.rng.random()
            cy = cfg.ambient_margin + (1 - 2 * cfg.ambient_margin) * self.rng.random()
            base = np.array([cx, cy], dtype=np.float32)
            ang = self.rng.random() * 2.0 * math.pi
            off = (
                np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)
                * cfg.vortex_radius
            )
            p1 = np.clip(base + off, 0.0, 1.0)
            p2 = np.clip(base - off, 0.0, 1.0)
            t = np.array([-off[1], off[0]], dtype=np.float32)
            n = float(np.linalg.norm(t))
            if n > 1e-6:
                t /= n
            kick = t * cfg.vortex_strength
            for (px, py), sgn in ((p1, +1.0), (p2, -1.0)):
                sim.fbo_vel_b.use()
                sim.vel_a.use(location=0)
                sim.prog_splat["field"].value = 0
                sim.prog_splat["point"].value = (float(px), float(py))
                sim.prog_splat["value"].value = (
                    float(kick[0] * sgn),
                    float(kick[1] * sgn),
                    0.0,
                )
                sim.prog_splat["radius"].value = cfg.ambient_radius * 1.2
                sim.vao_splat.render(moderngl.TRIANGLE_STRIP)
                sim.swap_vel()
