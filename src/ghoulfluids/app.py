from __future__ import annotations
import argparse

import time, math
import glfw, moderngl
import numpy as np

from .config import AppConfig
from .fluid import FluidSim
from .segmentation import MediaPipeSegmenter


def main(argv=None):
    # --- CLI ---
    parser = argparse.ArgumentParser(description="Ghoul Fluids")
    parser.add_argument(
        "--split",
        dest="split",
        action="store_true",
        help="Enable split view (camera | fluid)",
    )
    parser.set_defaults(split=False)
    args = parser.parse_args(argv)

    cfg = AppConfig()

    # --- window / context ---
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(
        cfg.width, cfg.height, "Ghoul Fluids (Split View)", None, None
    )
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    sim = FluidSim(ctx, cfg)
    seg = MediaPipeSegmenter(cfg.camera_index, cfg.width, cfg.height)

    running = True
    split_view = bool(args.split)

    prev_t = time.time()

    print(
        "SPACE pause  C clear  [/] vorticity -/+  P palette  V toggle split/fullscreen  ESC quit"
    )
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS:
                running = not running
                time.sleep(0.12)
            if glfw.get_key(win, glfw.KEY_C) == glfw.PRESS:
                for fbo in (
                    sim.fbo_vel_a,
                    sim.fbo_vel_b,
                    sim.fbo_dye_a,
                    sim.fbo_dye_b,
                    sim.fbo_prs,
                    sim.fbo_prs_b,
                    sim.fbo_div,
                    sim.fbo_curl,
                ):
                    fbo.use()
                    ctx.clear(0, 0, 0, 1)
                try:
                    ctx.screen.use()
                    time.sleep(0.12)
                except Exception:
                    print("Warning: could not re-bind screen framebuffer after clear")
                    pass
            if glfw.get_key(win, glfw.KEY_LEFT_BRACKET) == glfw.PRESS:
                sim.prog_vort["eps"].value = max(0.0, sim.prog_vort["eps"].value - 0.1)
                time.sleep(0.05)
            if glfw.get_key(win, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS:
                sim.prog_vort["eps"].value = min(5.0, sim.prog_vort["eps"].value + 0.1)
                time.sleep(0.05)
            if glfw.get_key(win, glfw.KEY_P) == glfw.PRESS:
                cfg.palette_on = 1 - cfg.palette_on
                time.sleep(0.12)
            if glfw.get_key(win, glfw.KEY_V) == glfw.PRESS:
                split_view = not split_view
                time.sleep(0.12)

            now = time.time()
            dt = min(cfg.dt_clamp, now - prev_t)
            prev_t = now

            frame_bgr, cam_rgb_flipped, mask_small, area = seg.read_frame_and_mask(
                sim.sim_w, sim.sim_h, cfg.width, cfg.height
            )

            have_mask = False
            if frame_bgr is not None:
                sim.upload_camera(cam_rgb_flipped)
                if mask_small is not None and area > cfg.mask_min_area:
                    sim.upload_mask(mask_small)
                    have_mask = True

            if running:
                sim.step(dt, have_mask)

                # ambient dye when no mask
                if cfg.use_fallback_orbit and not have_mask:
                    t = now
                    base_u = 0.5 + cfg.orbit_r * math.cos(t * cfg.orbit_speed)
                    base_v = 0.5 + cfg.orbit_r * math.sin(t * cfg.orbit_speed)
                    for i in range(cfg.micro_splats_per_frame):
                        ang = (
                            2
                            * math.pi
                            * (i / cfg.micro_splats_per_frame + 0.1 * math.sin(t * 0.4))
                        )
                        u = base_u + 0.010 * math.cos(ang)
                        v = base_v + 0.010 * math.sin(ang)
                        orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
                        purple = np.array([0.45, 0.0, 0.6], dtype=np.float32)
                        k = (
                            i / cfg.micro_splats_per_frame + 0.2 * math.sin(t * 0.7)
                        ) % 1.0
                        col = (
                            orange * (1.0 - k) + purple * k
                        ) * cfg.micro_splat_strength

                        sim.fbo_dye_b.use()
                        sim.dye_a.use(location=0)
                        sim.prog_splat["field"].value = 0
                        sim.prog_splat["point"].value = (float(u), float(v))
                        sim.prog_splat["value"].value = (
                            float(col[0]),
                            float(col[1]),
                            float(col[2]),
                        )
                        sim.prog_splat["radius"].value = cfg.micro_splat_radius
                        sim.vao_splat.render(moderngl.TRIANGLE_STRIP)
                        sim.swap_dye()

            # Present
            ctx.screen.use()
            sim.render_split(split_view)
            glfw.swap_buffers(win)

    finally:
        seg.release()
        glfw.terminate()
