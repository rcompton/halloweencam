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
    p = argparse.ArgumentParser(description="Ghoul Fluids")
    p.add_argument(
        "--split",
        action="store_true",
        help="Show split view (left=camera | right=fluid). Default: fullscreen fluid.",
    )
    p.set_defaults(split=False)
    # window / camera
    p.add_argument("--width", type=int, help="Window width (default from config.py)")
    p.add_argument("--height", type=int, help="Window height (default from config.py)")
    p.add_argument("--camera", type=int, help="Camera index (default from config.py)")

    # edge force params
    p.add_argument("--edge-normal", type=float, help="Edge normal force amplitude")
    p.add_argument("--edge-tangent", type=float, help="Edge tangential force amplitude")
    p.add_argument("--edge-thresh", type=float, help="Edge gradient threshold")

    # core fluid knobs
    p.add_argument("--vorticity", type=float, help="Vorticity confinement (eps)")
    p.add_argument("--vel-diss", type=float, help="Velocity dissipation (<=1.0)")
    p.add_argument("--dye-diss", type=float, help="Dye dissipation (<=1.0)")
    p.add_argument("--substeps", type=int, help="Simulation substeps per frame")
    p.add_argument("--jacobi", type=int, help="Jacobi iterations for pressure")
    p.add_argument("--dt-clamp", type=float, help="Max dt per frame (seconds)")

    args = p.parse_args(argv)

    # --- config ---
    cfg = AppConfig()
    if args.width:
        cfg.width = args.width
    if args.height:
        cfg.height = args.height
    if args.camera is not None:
        cfg.camera_index = args.camera

    if args.edge_normal is not None:
        cfg.edge_normal_amp = args.edge_normal
    if args.edge_tangent is not None:
        cfg.edge_tangent_amp = args.edge_tangent
    if args.edge_thresh is not None:
        cfg.edge_thresh = args.edge_thresh

    if args.vorticity is not None:
        cfg.vorticity_eps = args.vorticity
    if args.vel_diss is not None:
        cfg.vel_dissipation = args.vel_diss
    if args.dye_diss is not None:
        cfg.dye_dissipation = args.dye_diss
    if args.substeps is not None:
        cfg.substeps = args.substeps
    if args.jacobi is not None:
        cfg.jacobi_iters = args.jacobi
    if args.dt_clamp is not None:
        cfg.dt_clamp = args.dt_clamp

    print(
        f"edge_normal={cfg.edge_normal_amp:.3f}  "
        f"edge_tangent={cfg.edge_tangent_amp:.3f}  "
        f"edge_thresh={cfg.edge_thresh:.4f}  "
        f"vorticity={cfg.vorticity_eps:.2f}  "
        f"vel_diss={cfg.vel_dissipation:.6f}  dye_diss={cfg.dye_dissipation:.6f}  "
        f"substeps={cfg.substeps}  jacobi={cfg.jacobi_iters}"
    )

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
            # Edge strength hotkeys
            if glfw.get_key(win, glfw.KEY_1) == glfw.PRESS:
                cfg.edge_normal_amp = max(0.0, cfg.edge_normal_amp - 0.1)
                sim.prog_maskF["amp_normal"].value = cfg.edge_normal_amp
                print(f"edge_normal -> {cfg.edge_normal_amp:.3f}")
                time.sleep(0.05)

            if glfw.get_key(win, glfw.KEY_2) == glfw.PRESS:
                cfg.edge_normal_amp += 0.1
                sim.prog_maskF["amp_normal"].value = cfg.edge_normal_amp
                print(f"edge_normal -> {cfg.edge_normal_amp:.3f}")
                time.sleep(0.05)

            if glfw.get_key(win, glfw.KEY_3) == glfw.PRESS:
                cfg.edge_tangent_amp = max(0.0, cfg.edge_tangent_amp - 0.1)
                sim.prog_maskF["amp_tangent"].value = cfg.edge_tangent_amp
                print(f"edge_tangent -> {cfg.edge_tangent_amp:.3f}")
                time.sleep(0.05)

            if glfw.get_key(win, glfw.KEY_4) == glfw.PRESS:
                cfg.edge_tangent_amp += 0.1
                sim.prog_maskF["amp_tangent"].value = cfg.edge_tangent_amp
                print(f"edge_tangent -> {cfg.edge_tangent_amp:.3f}")
                time.sleep(0.05)

            if glfw.get_key(win, glfw.KEY_9) == glfw.PRESS:
                cfg.edge_thresh = max(0.0, cfg.edge_thresh - 0.001)
                sim.prog_maskF["edge_thresh"].value = cfg.edge_thresh
                sim.prog_maskD["edge_thresh"].value = cfg.edge_thresh
                print(f"edge_thresh -> {cfg.edge_thresh:.4f}")
                time.sleep(0.05)

            if glfw.get_key(win, glfw.KEY_0) == glfw.PRESS:
                cfg.edge_thresh += 0.001
                sim.prog_maskF["edge_thresh"].value = cfg.edge_thresh
                sim.prog_maskD["edge_thresh"].value = cfg.edge_thresh
                print(f"edge_thresh -> {cfg.edge_thresh:.4f}")
                time.sleep(0.05)

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
