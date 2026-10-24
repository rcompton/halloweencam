from __future__ import annotations
import time, math
import numpy as np
import moderngl
from .config import AppConfig
from . import shaders as S
from .logging import get_logger
from .profiler import get_profiler


def make_tex(ctx, size, comps, dtype="f4", clamp=True):
    tex = ctx.texture(size, comps, dtype=dtype)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        tex.repeat_x = False
        tex.repeat_y = False
    return tex


def fullscreen_quad(ctx):
    v = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    return ctx.buffer(v.tobytes())


class FluidSim:
    def __init__(self, ctx: moderngl.Context, cfg: AppConfig):
        self.ctx = ctx
        self.cfg = cfg
        logger = get_logger(__name__)
        self.profiler = get_profiler()

        self.sim_w = max(32, int(cfg.width * cfg.sim_scale))
        self.sim_h = max(32, int(cfg.height * cfg.sim_scale))

        max_dim = max(self.sim_w, self.sim_h)
        if max_dim > cfg.sim_max_dim:
            s = cfg.sim_max_dim / float(max_dim)
            self.sim_w = max(32, int(self.sim_w * s))
            self.sim_h = max(32, int(self.sim_h * s))

        # Ensure dimensions are a multiple of 32 for performance
        self.sim_w = int(round(self.sim_w / 32.0)) * 32
        self.sim_h = int(round(self.sim_h / 32.0)) * 32
        logger.info(f"Calculated simulation dimensions: {self.sim_w}x{self.sim_h}")

        self.texel = (1.0 / self.sim_w, 1.0 / self.sim_h)

        self.dye_w = max(32, int(cfg.width * cfg.render_scale))
        self.dye_h = max(32, int(cfg.height * cfg.render_scale))

        max_dim = max(self.dye_w, self.dye_h)
        if max_dim > cfg.dye_max_dim:
            s = cfg.dye_max_dim / float(max_dim)
            self.dye_w = max(32, int(self.dye_w * s))
            self.dye_h = max(32, int(self.dye_h * s))
        logger.info(f"Calculated dye dimensions: {self.dye_w}x{self.dye_h}")

        # programs / VAOs
        self.prog_adv = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_ADVECT)
        self.prog_splat = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SPLAT)
        self.prog_force = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_FORCE)
        self.prog_div = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_DIVERGENCE)
        self.prog_jac = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_JACOBI)
        self.prog_grad = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_GRADIENT)
        self.prog_curl = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_CURL)
        self.prog_vort = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_VORTICITY)
        self.prog_maskF = ctx.program(
            vertex_shader=S.VS, fragment_shader=S.FS_MASK_FORCE
        )
        self.prog_maskF_full = ctx.program(
            vertex_shader=S.VS, fragment_shader=S.FS_MASK_FORCE_FULL
        )
        self.prog_maskD = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_MASK_DYE)
        self.prog_show = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SHOW)
        self.prog_cam = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SHOW_CAM)
        self.prog_ghost = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_GHOST)

        # palette state (updated each frame by the app)
        self.palette_a = self.cfg.palette_id
        self.palette_b = self.cfg.palette_id
        self.palette_mix = 0.0

        self.vbo = fullscreen_quad(ctx)
        self.vao_adv = ctx.simple_vertex_array(self.prog_adv, self.vbo, "in_vert")
        self.vao_splat = ctx.simple_vertex_array(self.prog_splat, self.vbo, "in_vert")
        self.vao_force = ctx.simple_vertex_array(self.prog_force, self.vbo, "in_vert")
        self.vao_div = ctx.simple_vertex_array(self.prog_div, self.vbo, "in_vert")
        self.vao_jac = ctx.simple_vertex_array(self.prog_jac, self.vbo, "in_vert")
        self.vao_grad = ctx.simple_vertex_array(self.prog_grad, self.vbo, "in_vert")
        self.vao_curl = ctx.simple_vertex_array(self.prog_curl, self.vbo, "in_vert")
        self.vao_vort = ctx.simple_vertex_array(self.prog_vort, self.vbo, "in_vert")
        self.vao_maskF = ctx.simple_vertex_array(self.prog_maskF, self.vbo, "in_vert")
        self.vao_maskF_full = ctx.simple_vertex_array(
            self.prog_maskF_full, self.vbo, "in_vert"
        )
        self.vao_maskD = ctx.simple_vertex_array(self.prog_maskD, self.vbo, "in_vert")
        self.vao_show = ctx.simple_vertex_array(self.prog_show, self.vbo, "in_vert")
        self.vao_cam = ctx.simple_vertex_array(self.prog_cam, self.vbo, "in_vert")
        self.vao_ghost = ctx.simple_vertex_array(self.prog_ghost, self.vbo, "in_vert")

        # textures / FBOs
        self.vel_a = make_tex(ctx, (self.sim_w, self.sim_h), 2)
        self.vel_b = make_tex(ctx, (self.sim_w, self.sim_h), 2)
        dye_dtype = "f2" if self.cfg.dye_fp16 else "f4"
        self.dye_a = make_tex(ctx, (self.dye_w, self.dye_h), 4, dtype=dye_dtype)
        self.dye_b = make_tex(ctx, (self.dye_w, self.dye_h), 4, dtype=dye_dtype)
        self.prs = make_tex(ctx, (self.sim_w, self.sim_h), 1)
        self.prs_b = make_tex(ctx, (self.sim_w, self.sim_h), 1)
        self.div = make_tex(ctx, (self.sim_w, self.sim_h), 1)
        self.curl = make_tex(ctx, (self.sim_w, self.sim_h), 1)

        self.mask_curr = make_tex(ctx, (self.sim_w, self.sim_h), 1)
        self.mask_prev = make_tex(ctx, (self.sim_w, self.sim_h), 1)
        self.ghost_a = make_tex(ctx, (self.sim_w, self.sim_h), 1, dtype="f2")
        self.ghost_b = make_tex(ctx, (self.sim_w, self.sim_h), 1, dtype="f2")

        self.cam_tex = make_tex(ctx, (640, 480), 3, dtype="f1")

        self.fbo_vel_a = ctx.framebuffer([self.vel_a])
        self.fbo_vel_b = ctx.framebuffer([self.vel_b])
        self.fbo_dye_a = ctx.framebuffer([self.dye_a])
        self.fbo_dye_b = ctx.framebuffer([self.dye_b])
        self.fbo_prs = ctx.framebuffer([self.prs])
        self.fbo_prs_b = ctx.framebuffer([self.prs_b])
        self.fbo_div = ctx.framebuffer([self.div])
        self.fbo_curl = ctx.framebuffer([self.curl])
        self.fbo_ghost_a = ctx.framebuffer([self.ghost_a])
        self.fbo_ghost_b = ctx.framebuffer([self.ghost_b])

        for fbo in (
            self.fbo_vel_a,
            self.fbo_vel_b,
            self.fbo_dye_a,
            self.fbo_dye_b,
            self.fbo_prs,
            self.fbo_prs_b,
            self.fbo_div,
            self.fbo_curl,
            self.fbo_ghost_a,
            self.fbo_ghost_b,
        ):
            fbo.use()
            ctx.clear(0, 0, 0, 1)
        try:
            self.ctx.screen.use()
        except Exception:
            print("Warning: could not set screen framebuffer (yet?)")
            pass

        # static uniforms
        self.prog_div["texel"].value = self.texel
        self.prog_jac["texel"].value = self.texel
        self.prog_grad["texel"].value = self.texel
        self.prog_jac["open_top"].value = 1 if self.cfg.open_top else 0
        self.prog_grad["open_top"].value = 1 if self.cfg.open_top else 0
        self.prog_curl["texel"].value = self.texel
        self.prog_vort["texel"].value = self.texel
        self.prog_vort["eps"].value = self.cfg.vorticity_eps
        self.prog_maskF["texel"].value = self.texel
        self.prog_maskF["edge_thresh"].value = self.cfg.edge_thresh
        self.prog_maskF["amp_normal"].value = self.cfg.edge_normal_amp
        self.prog_maskF["amp_tangent"].value = self.cfg.edge_tangent_amp
        self.prog_maskF["use_temporal"].value = 1 if self.cfg.edge_use_temporal else 0
        self.prog_maskF_full["texel"].value = self.texel
        self.prog_maskF_full["amp_normal"].value = self.cfg.edge_normal_amp
        self.prog_maskD["texel"].value = (1.0 / self.dye_w, 1.0 / self.dye_h)
        self.prog_maskD["edge_thresh"].value = self.cfg.edge_thresh
        self.prog_maskD["edge_color"].value = (1.0, 0.45, 0.1)
        self.prog_maskD["border"].value = self.cfg.edge_border
        self.prog_maskD["dye_cap"].value = self.cfg.edge_dye_cap
        self.prog_maskF["border"].value = self.cfg.edge_border

        # seed a puff
        self.fbo_dye_b.use()
        self.dye_a.use(location=0)
        self.prog_splat["field"].value = 0
        self.prog_splat["point"].value = (0.5, 0.5)
        self.prog_splat["value"].value = (0.9, 0.4, 0.9)
        self.prog_splat["radius"].value = 0.12
        self.vao_splat.render(moderngl.TRIANGLE_STRIP)
        self.swap_dye()

        self.last_mask_small = None
        self.crowd_k = 1.0  # edge force/dye multiplier from how much of the frame people fill
        self.eyes = []  # [(alpha, (u, v), openness, scale)] per pair, from EyesSwarm

    # small helpers to swap ping-pong targets
    def swap_vel(self):
        self.vel_a, self.vel_b = self.vel_b, self.vel_a
        self.fbo_vel_a, self.fbo_vel_b = self.fbo_vel_b, self.fbo_vel_a

    def swap_dye(self):
        self.dye_a, self.dye_b = self.dye_b, self.dye_a
        self.fbo_dye_a, self.fbo_dye_b = self.fbo_dye_b, self.fbo_dye_a

    def upload_camera(self, cam_rgb_flipped):
        self.cam_tex.write(cam_rgb_flipped.tobytes())

    def upload_mask(self, mask_small):
        if self.last_mask_small is None:
            self.last_mask_small = mask_small.copy()
        self.mask_prev.write(self.last_mask_small.tobytes())
        self.mask_curr.write(mask_small.tobytes())
        self.last_mask_small = mask_small.copy()

    def set_crowd(self, area: float):
        """Scale edge forces and dye with how much of the frame people fill.
        One person far away is ~0.05 of the frame; a close group 0.3+."""
        lo, hi = 0.08, 0.35
        x = max(0.0, min(1.0, (area - lo) / (hi - lo)))
        target = 1.0 + self.cfg.crowd_boost * x
        # ease toward it so detections flickering don't pump the effect
        self.crowd_k += 0.1 * (target - self.crowd_k)

    MAX_EYES = 4  # keep in sync with MAX_EYES in FS_SHOW

    def set_eyes(self, pairs):
        self.eyes = list(pairs)[: self.MAX_EYES]

    def update_ghost(self, dt: float, have_mask: bool):
        """Once per frame: fade the silhouette trail and stamp the current mask."""
        if self.cfg.ghost_strength <= 0.0:
            return
        self.fbo_ghost_b.use()
        self.ghost_a.use(location=0)
        self.mask_curr.use(location=1)
        self.prog_ghost["ghost_in"].value = 0
        self.prog_ghost["mask_curr"].value = 1
        self.prog_ghost["decay"].value = math.exp(-dt / max(1e-3, self.cfg.ghost_fade))
        self.prog_ghost["stamp"].value = 1.0 if have_mask else 0.0
        self.vao_ghost.render(moderngl.TRIANGLE_STRIP)
        self.ghost_a, self.ghost_b = self.ghost_b, self.ghost_a
        self.fbo_ghost_a, self.fbo_ghost_b = self.fbo_ghost_b, self.fbo_ghost_a

    def set_palette_blend(self, a: int, b: int, mix: float):
        self.palette_a = int(a)
        self.palette_b = int(b)
        self.palette_mix = max(0.0, min(1.0, float(mix)))

    def step(self, dt: float, have_mask: bool):
        sdt = dt / self.cfg.substeps
        k = self.crowd_k
        self.prog_maskF["amp_normal"].value = self.cfg.edge_normal_amp * k
        self.prog_maskF["amp_tangent"].value = self.cfg.edge_tangent_amp * k
        self.prog_maskF_full["amp_normal"].value = self.cfg.edge_normal_amp * k
        for _ in range(self.cfg.substeps):
            # Background velocity
            if self.cfg.background_velocity > 0.0:
                with self.profiler.record("background_v"):
                    self.fbo_vel_b.use()
                    self.vel_a.use(location=0)
                    self.prog_force["field"].value = 0
                    self.prog_force["value"].value = (
                        0.0,
                        self.cfg.background_velocity * sdt,
                        0.0,
                    )
                    self.vao_force.render(moderngl.TRIANGLE_STRIP)
                    self.swap_vel()

            # Advect velocity
            with self.profiler.record("advect_v"):
                self.fbo_vel_b.use()
                self.vel_a.use(location=0)
                self.vel_a.use(location=1)
                self.prog_adv["vel"].value = 0
                self.prog_adv["src"].value = 1
                self.prog_adv["dt"].value = sdt
                self.prog_adv["dissipation"].value = self.cfg.vel_dissipation
                self.prog_adv["top_fade"].value = 0.0
                self.vao_adv.render(moderngl.TRIANGLE_STRIP)
                self.swap_vel()

            # Edge forces
            if have_mask:
                with self.profiler.record("edge_force"):
                    self.fbo_vel_b.use()
                    self.vel_a.use(location=0)
                    self.mask_curr.use(location=1)
                    self.mask_prev.use(location=2)
                    if self.cfg.force_mode == "full":
                        self.prog_maskF_full["vel_in"].value = 0
                        self.prog_maskF_full["mask_curr"].value = 1
                        self.prog_maskF_full["dt"].value = sdt
                        self.vao_maskF_full.render(moderngl.TRIANGLE_STRIP)
                    else:  # 'edges'
                        self.prog_maskF["vel_in"].value = 0
                        self.prog_maskF["mask_curr"].value = 1
                        self.prog_maskF["mask_prev"].value = 2
                        self.prog_maskF["dt"].value = sdt
                        self.vao_maskF.render(moderngl.TRIANGLE_STRIP)
                    self.swap_vel()

            # Vorticity
            if self.cfg.vorticity_eps > 0.0:
                with self.profiler.record("vorticity"):
                    self.fbo_curl.use()
                    self.vel_a.use(location=0)
                    self.prog_curl["vel"].value = 0
                    self.vao_curl.render(moderngl.TRIANGLE_STRIP)

                    self.fbo_vel_b.use()
                    self.vel_a.use(location=0)
                    self.curl.use(location=1)
                    self.prog_vort["vel"].value = 0
                    self.prog_vort["curlTex"].value = 1
                    self.prog_vort["dt"].value = sdt
                    self.vao_vort.render(moderngl.TRIANGLE_STRIP)
                    self.swap_vel()

            # Projection
            with self.profiler.record("projection"):
                self.fbo_div.use()
                self.vel_a.use(location=0)
                self.prog_div["vel"].value = 0
                self.vao_div.render(moderngl.TRIANGLE_STRIP)

                for _j in range(self.cfg.jacobi_iters):
                    self.fbo_prs_b.use()
                    self.prs.use(location=0)
                    self.div.use(location=1)
                    self.prog_jac["prs"].value = 0
                    self.prog_jac["div"].value = 1
                    self.vao_jac.render(moderngl.TRIANGLE_STRIP)
                    self.prs, self.prs_b = self.prs_b, self.prs
                    self.fbo_prs, self.fbo_prs_b = self.fbo_prs_b, self.fbo_prs

                self.fbo_vel_b.use()
                self.vel_a.use(location=0)
                self.prs.use(location=1)
                self.prog_grad["vel"].value = 0
                self.prog_grad["prs"].value = 1
                self.vao_grad.render(moderngl.TRIANGLE_STRIP)
                self.swap_vel()

            # Advect dye
            with self.profiler.record("advect_d"):
                self.fbo_dye_b.use()
                self.vel_a.use(location=0)
                self.dye_a.use(location=1)
                self.prog_adv["vel"].value = 0
                self.prog_adv["src"].value = 1
                self.prog_adv["dt"].value = sdt
                self.prog_adv["dissipation"].value = self.cfg.dye_dissipation
                self.prog_adv["top_fade"].value = self.cfg.top_fade
                self.vao_adv.render(moderngl.TRIANGLE_STRIP)
                self.swap_dye()

            # Outline dye (optional)
            if have_mask and self.cfg.edge_dye_strength > 0.0:
                with self.profiler.record("edge_dye"):
                    self.fbo_dye_b.use()
                    self.dye_a.use(location=0)
                    self.mask_curr.use(location=1)
                    self.prog_maskD["dye_in"].value = 0
                    self.prog_maskD["mask_curr"].value = 1
                    self.prog_maskD["strength"].value = self.cfg.edge_dye_strength * k
                    self.vao_maskD.render(moderngl.TRIANGLE_STRIP)
                    self.swap_dye()

    def render_split(self, split: bool):
        if split:
            # left = camera
            self.ctx.viewport = (0, 0, self.cfg.width // 2, self.cfg.height)
            self.cam_tex.use(location=0)
            self.prog_cam["cam"].value = 0
            self.vao_cam.render(moderngl.TRIANGLE_STRIP)
            # right = fluid
            self.ctx.viewport = (
                self.cfg.width // 2,
                0,
                self.cfg.width - self.cfg.width // 2,
                self.cfg.height,
            )
        else:
            self.ctx.viewport = (0, 0, self.cfg.width, self.cfg.height)

        self.dye_a.use(location=0)
        self.prog_show["dye"].value = 0
        self.prog_show["palette_on"].value = self.cfg.palette_on
        self.prog_show["palette_id"].value = int(self.palette_a)
        self.prog_show["palette_id2"].value = int(self.palette_b)
        self.prog_show["palette_mix"].value = float(self.palette_mix)
        self.prog_show["glow_strength"].value = self.cfg.glow_strength
        self.prog_show["glow_radius"].value = self.cfg.glow_radius
        self.prog_show["vignette"].value = self.cfg.vignette
        self.prog_show["aspect"].value = self.cfg.width / float(self.cfg.height)
        self.ghost_a.use(location=1)
        self.mask_curr.use(location=2)
        self.prog_show["ghost"].value = 1
        self.prog_show["mask_now"].value = 2
        self.prog_show["ghost_strength"].value = self.cfg.ghost_strength
        self.prog_show["ghost_color"].value = self.cfg.ghost_color
        eyes = [(0.0, (0.5, 0.5), 1.0, 1.0)] * self.MAX_EYES
        eyes[: len(self.eyes)] = self.eyes
        self.prog_show["eyes"].value = [(a, p[0], p[1], o) for a, p, o, _ in eyes]
        self.prog_show["eyes_size"].value = [self.cfg.eyes_size * s for *_, s in eyes]
        self.vao_show.render(moderngl.TRIANGLE_STRIP)

        # restore full viewport
        self.ctx.viewport = (0, 0, self.cfg.width, self.cfg.height)
