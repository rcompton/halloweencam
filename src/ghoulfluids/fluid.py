from __future__ import annotations
import time, math
import numpy as np
import moderngl
from .config import AppConfig
from . import shaders as S

def make_tex(ctx, size, comps, dtype='f4', clamp=True):
    tex = ctx.texture(size, comps, dtype=dtype)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        tex.repeat_x = False; tex.repeat_y = False
    return tex

def fullscreen_quad(ctx):
    v = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')
    return ctx.buffer(v.tobytes())

class FluidSim:
    def __init__(self, ctx: moderngl.Context, cfg: AppConfig):
        self.ctx = ctx
        self.cfg = cfg

        self.sim_w = max(32, int(cfg.width * cfg.sim_scale))
        self.sim_h = max(32, int(cfg.height * cfg.sim_scale))
        self.texel = (1.0/self.sim_w, 1.0/self.sim_h)

        # programs / VAOs
        self.prog_adv   = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_ADVECT)
        self.prog_splat = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SPLAT)
        self.prog_div   = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_DIVERGENCE)
        self.prog_jac   = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_JACOBI)
        self.prog_grad  = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_GRADIENT)
        self.prog_curl  = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_CURL)
        self.prog_vort  = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_VORTICITY)
        self.prog_maskF = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_MASK_FORCE)
        self.prog_maskD = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_MASK_DYE)
        self.prog_show  = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SHOW)
        self.prog_cam   = ctx.program(vertex_shader=S.VS, fragment_shader=S.FS_SHOW_CAM)

        self.vbo = fullscreen_quad(ctx)
        self.vao_adv   = ctx.simple_vertex_array(self.prog_adv,   self.vbo, "in_vert")
        self.vao_splat = ctx.simple_vertex_array(self.prog_splat, self.vbo, "in_vert")
        self.vao_div   = ctx.simple_vertex_array(self.prog_div,   self.vbo, "in_vert")
        self.vao_jac   = ctx.simple_vertex_array(self.prog_jac,   self.vbo, "in_vert")
        self.vao_grad  = ctx.simple_vertex_array(self.prog_grad,  self.vbo, "in_vert")
        self.vao_curl  = ctx.simple_vertex_array(self.prog_curl,  self.vbo, "in_vert")
        self.vao_vort  = ctx.simple_vertex_array(self.prog_vort,  self.vbo, "in_vert")
        self.vao_maskF = ctx.simple_vertex_array(self.prog_maskF, self.vbo, "in_vert")
        self.vao_maskD = ctx.simple_vertex_array(self.prog_maskD, self.vbo, "in_vert")
        self.vao_show  = ctx.simple_vertex_array(self.prog_show,  self.vbo, "in_vert")
        self.vao_cam   = ctx.simple_vertex_array(self.prog_cam,   self.vbo, "in_vert")

        # textures / FBOs
        self.vel_a = make_tex(ctx, (self.sim_w,self.sim_h), 2)
        self.vel_b = make_tex(ctx, (self.sim_w,self.sim_h), 2)
        self.dye_a = make_tex(ctx, (cfg.width,cfg.height), 4)
        self.dye_b = make_tex(ctx, (cfg.width,cfg.height), 4)
        self.prs   = make_tex(ctx, (self.sim_w,self.sim_h), 1)
        self.prs_b = make_tex(ctx, (self.sim_w,self.sim_h), 1)
        self.div   = make_tex(ctx, (self.sim_w,self.sim_h), 1)
        self.curl  = make_tex(ctx, (self.sim_w,self.sim_h), 1)

        self.mask_curr = make_tex(ctx, (self.sim_w,self.sim_h), 1)
        self.mask_prev = make_tex(ctx, (self.sim_w,self.sim_h), 1)

        self.cam_tex = make_tex(ctx, (cfg.width,cfg.height), 3, dtype='f1')

        self.fbo_vel_a = ctx.framebuffer([self.vel_a])
        self.fbo_vel_b = ctx.framebuffer([self.vel_b])
        self.fbo_dye_a = ctx.framebuffer([self.dye_a])
        self.fbo_dye_b = ctx.framebuffer([self.dye_b])
        self.fbo_prs   = ctx.framebuffer([self.prs])
        self.fbo_prs_b = ctx.framebuffer([self.prs_b])
        self.fbo_div   = ctx.framebuffer([self.div])
        self.fbo_curl  = ctx.framebuffer([self.curl])

        for fbo in (self.fbo_vel_a,self.fbo_vel_b,self.fbo_dye_a,self.fbo_dye_b,
                    self.fbo_prs,self.fbo_prs_b,self.fbo_div,self.fbo_curl):
            fbo.use(); ctx.clear(0,0,0,1)
        ctx.screen.use()

        # static uniforms
        self.prog_div['texel'].value = self.texel
        self.prog_jac['texel'].value = self.texel
        self.prog_grad['texel'].value = self.texel
        self.prog_curl['texel'].value = self.texel
        self.prog_vort['texel'].value = self.texel
        self.prog_vort['eps'].value = self.cfg.vorticity_eps
        self.prog_maskF['texel'].value = self.texel
        self.prog_maskF['edge_thresh'].value = self.cfg.edge_thresh
        self.prog_maskF['amp_normal'].value = self.cfg.edge_normal_amp
        self.prog_maskF['amp_tangent'].value = self.cfg.edge_tangent_amp
        self.prog_maskF['use_temporal'].value = 1 if self.cfg.edge_use_temporal else 0
        self.prog_maskD['texel'].value = (1.0/self.cfg.width, 1.0/self.cfg.height)
        self.prog_maskD['edge_thresh'].value = self.cfg.edge_thresh
        self.prog_maskD['edge_color'].value = (1.0, 0.45, 0.1)

        # seed a puff
        self.fbo_dye_b.use()
        self.dye_a.use(location=0)
        self.prog_splat['field'].value = 0
        self.prog_splat['point'].value = (0.5,0.5)
        self.prog_splat['value'].value = (0.9,0.4,0.9)
        self.prog_splat['radius'].value = 0.12
        self.vao_splat.render(moderngl.TRIANGLE_STRIP)
        self.swap_dye()

        self.last_mask_small = None

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

    def step(self, dt: float, have_mask: bool):
        sdt = dt / self.cfg.substeps
        for _ in range(self.cfg.substeps):
            # Advect velocity
            self.fbo_vel_b.use()
            self.vel_a.use(location=0); self.vel_a.use(location=1)
            self.prog_adv['vel'].value = 0
            self.prog_adv['src'].value = 1
            self.prog_adv['dt'].value = sdt
            self.prog_adv['dissipation'].value = self.cfg.vel_dissipation
            self.vao_adv.render(moderngl.TRIANGLE_STRIP)
            self.swap_vel()

            # Edge forces
            if have_mask:
                self.fbo_vel_b.use()
                self.vel_a.use(location=0); self.mask_curr.use(location=1); self.mask_prev.use(location=2)
                self.prog_maskF['vel_in'].value = 0
                self.prog_maskF['mask_curr'].value = 1
                self.prog_maskF['mask_prev'].value = 2
                self.prog_maskF['dt'].value = sdt
                self.vao_maskF.render(moderngl.TRIANGLE_STRIP)
                self.swap_vel()

            # Vorticity
            if self.cfg.vorticity_eps > 0.0:
                self.fbo_curl.use()
                self.vel_a.use(location=0)
                self.prog_curl['vel'].value = 0
                self.vao_curl.render(moderngl.TRIANGLE_STRIP)

                self.fbo_vel_b.use()
                self.vel_a.use(location=0); self.curl.use(location=1)
                self.prog_vort['vel'].value = 0
                self.prog_vort['curlTex'].value = 1
                self.prog_vort['dt'].value = sdt
                self.vao_vort.render(moderngl.TRIANGLE_STRIP)
                self.swap_vel()

            # Projection
            self.fbo_div.use()
            self.vel_a.use(location=0)
            self.prog_div['vel'].value = 0
            self.vao_div.render(moderngl.TRIANGLE_STRIP)

            for _j in range(self.cfg.jacobi_iters):
                self.fbo_prs_b.use()
                self.prs.use(location=0); self.div.use(location=1)
                self.prog_jac['prs'].value = 0
                self.prog_jac['div'].value = 1
                self.vao_jac.render(moderngl.TRIANGLE_STRIP)
                self.prs, self.prs_b = self.prs_b, self.prs
                self.fbo_prs, self.fbo_prs_b = self.fbo_prs_b, self.fbo_prs

            self.fbo_vel_b.use()
            self.vel_a.use(location=0); self.prs.use(location=1)
            self.prog_grad['vel'].value = 0
            self.prog_grad['prs'].value = 1
            self.vao_grad.render(moderngl.TRIANGLE_STRIP)
            self.swap_vel()

            # Advect dye
            self.fbo_dye_b.use()
            self.vel_a.use(location=0); self.dye_a.use(location=1)
            self.prog_adv['vel'].value = 0
            self.prog_adv['src'].value = 1
            self.prog_adv['dt'].value = sdt
            self.prog_adv['dissipation'].value = self.cfg.dye_dissipation
            self.vao_adv.render(moderngl.TRIANGLE_STRIP)
            self.swap_dye()

            # Outline dye (optional)
            if have_mask and self.cfg.edge_dye_strength > 0.0:
                self.fbo_dye_b.use()
                self.dye_a.use(location=0); self.mask_curr.use(location=1)
                self.prog_maskD['dye_in'].value = 0
                self.prog_maskD['mask_curr'].value = 1
                self.prog_maskD['strength'].value = self.cfg.edge_dye_strength
                self.vao_maskD.render(moderngl.TRIANGLE_STRIP)
                self.swap_dye()

    def render_split(self, split: bool):
        if split:
            # left = camera
            self.ctx.viewport = (0, 0, self.cfg.width//2, self.cfg.height)
            self.cam_tex.use(location=0)
            self.prog_cam['cam'].value = 0
            self.vao_cam.render(moderngl.TRIANGLE_STRIP)
            # right = fluid
            self.ctx.viewport = (self.cfg.width//2, 0, self.cfg.width - self.cfg.width//2, self.cfg.height)
        else:
            self.ctx.viewport = (0, 0, self.cfg.width, self.cfg.height)

        self.dye_a.use(location=0)
        self.prog_show['dye'].value = 0
        self.prog_show['palette_on'].value = self.cfg.palette_on
        self.vao_show.render(moderngl.TRIANGLE_STRIP)
        # restore full viewport
        self.ctx.viewport = (0, 0, self.cfg.width, self.cfg.height)
