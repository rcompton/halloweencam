from __future__ import annotations
import argparse
import random
import shutil
import sys

import time, math
import glfw, moderngl
import numpy as np

from .config import AppConfig
from .fluid import FluidSim
from .segmentation import MediaPipeSegmenter
from .ambient import AmbientController


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
    p.add_argument(
        "--fullscreen",
        action="store_true",
        help="Open in fullscreen on the primary monitor.",
    )
    p.add_argument("--width", type=int, help="Window width (default from config.py)")
    p.add_argument("--height", type=int, help="Window height (default from config.py)")
    p.add_argument(
        "--render-scale",
        type=float,
        help="Dye render scale (0.3–1.0). Default from config.",
    )
    p.add_argument(
        "--dye-f32",
        action="store_true",
        help="Use 32-bit float for dye (default is FP16).",
    )
    p.add_argument(
        "--sim-max", type=int, help="Cap for max(sim_w, sim_h); default from config."
    )

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
    if args.render_scale is not None:
        cfg.render_scale = max(0.3, min(1.0, args.render_scale))
    if args.dye_f32:
        cfg.dye_fp16 = False
    if args.sim_max is not None:
        cfg.sim_max_dim = max(128, args.sim_max)

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

    monitor = None
    if args.fullscreen:
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        # Use the monitor's native pixel size
        cfg.width, cfg.height = mode.size.width, mode.size.height

    win = glfw.create_window(cfg.width, cfg.height, "Ghoul Fluids", monitor, None)
    glfw.make_context_current(win)

    def _linux_gl_hint():
        if sys.platform.startswith("linux"):
            have_gl = shutil.which("glxinfo") is not None
            return (
                "Linux OpenGL loaders not found.\n"
                "Install the dev libraries:\n"
                "  sudo apt install -y libgl1-mesa-dev libegl1-mesa-dev libglvnd-dev mesa-utils\n"
                "Then re-run, or create symlinks:\n"
                "  sudo ln -s /usr/lib/x86_64-linux-gnu/libEGL.so.1 /usr/local/lib/libEGL.so && \\\n"
                "  sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1  /usr/local/lib/libGL.so && sudo ldconfig\n"
            )

    try:
        ctx = moderngl.create_context()
    except Exception as e:
        print(_linux_gl_hint() or "Failed to create ModernGL context.", file=sys.stderr)
        raise

    sim = FluidSim(ctx, cfg)
    seg = MediaPipeSegmenter(cfg.camera_index, cfg.width, cfg.height)
    ambient = AmbientController(cfg, seed=42)

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
                # Ambient dye & velocity
                if not have_mask and cfg.ambient_emitters > 0:
                    ambient.step(dt, sim)
                # Main step of the simulation
                sim.step(dt, have_mask)

            # Present
            ctx.screen.use()
            sim.render_split(split_view)
            glfw.swap_buffers(win)

    finally:
        seg.release()
        glfw.terminate()
