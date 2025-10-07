from __future__ import annotations
import argparse
import random
import shutil
import sys

import time, math
import glfw, moderngl
import numpy as np

from .ambient import AmbientController
from .config import AppConfig
from .fluid import FluidSim
from .logging import get_logger, setup_logging
from .segmentation import MediaPipeSegmenter, YOLOSegmenter


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
    p.add_argument(
        "--render-scale",
        type=float,
        help="Dye render scale (0.3–1.0). Default from config.",
    )
    p.add_argument(
        "--segmenter",
        type=str,
        choices=["mediapipe", "yolo"],
        default=None,
        help="Segmentation backend (mediapipe, yolo). Default: from config.",
    )
    p.add_argument("--log-file", type=str, help="Path to log file.")
    args = p.parse_args(argv)

    # --- config ---
    cfg = AppConfig()
    if args.render_scale is not None:
        cfg.render_scale = max(0.3, min(1.0, args.render_scale))
    if args.segmenter is not None:
        cfg.segmenter = args.segmenter
    if args.log_file is not None:
        cfg.log_file = args.log_file

    # --- logging ---
    setup_logging(cfg.log_level, cfg.log_file)
    logger = get_logger(__name__)

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
        logger.error(_linux_gl_hint() or "Failed to create ModernGL context.")
        raise

    sim = FluidSim(ctx, cfg)

    if cfg.segmenter == "yolo":
        logger.info("Using YOLO segmenter")
        seg = YOLOSegmenter(cfg.camera_index, cfg.width, cfg.height, cfg.yolo_model)
    elif cfg.segmenter == "mediapipe":
        logger.info("Using MediaPipe segmenter")
        seg = MediaPipeSegmenter(cfg.camera_index, cfg.width, cfg.height)
    else:
        raise ValueError(f"Unknown segmenter: {cfg.segmenter}")

    ambient = AmbientController(cfg, seed=42)

    # ---- palette cycle state ----
    N_PALETTES = 6  # keep in sync with shader list above
    palette_cycle_on = bool(cfg.palette_cycle)
    curr_pal = int(cfg.palette_id) % N_PALETTES
    next_pal = (curr_pal + 1) % N_PALETTES
    dwell_t = 0.0
    fade_t = 0.0
    in_fade = False
    sim.set_palette_blend(curr_pal, curr_pal, 0.0)

    running = True
    split_view = bool(args.split)

    prev_t = time.time()
    frame_count = 0
    log_interval = 1.0  # seconds
    time_since_log = 0.0

    logger.info("SPACE pause  C clear  P palette  V toggle split/fullscreen  ESC quit")
    try:
        while not glfw.window_should_close(win):
            # --- Event handling ---
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
                    logger.warning("Could not re-bind screen framebuffer after clear")
                    pass
            if glfw.get_key(win, glfw.KEY_P) == glfw.PRESS:
                cfg.palette_on = 1 - cfg.palette_on
                time.sleep(0.12)
            if glfw.get_key(win, glfw.KEY_V) == glfw.PRESS:
                split_view = not split_view
                time.sleep(0.12)

            # --- Time delta ---
            now = time.time()
            dt = min(cfg.dt_clamp, now - prev_t)
            prev_t = now

            # --- Segmentation processing ---
            # Get camera frame and segmentation mask for the fluid sim.
            frame_bgr, cam_rgb_flipped, mask_small, area = seg.read_frame_and_mask(
                sim.sim_w, sim.sim_h, cfg.width, cfg.height
            )

            # Upload to GPU textures
            have_mask = False
            if frame_bgr is not None:
                sim.upload_camera(cam_rgb_flipped)
                if mask_small is not None and area > cfg.mask_min_area:
                    sim.upload_mask(mask_small)
                    have_mask = True

            if running:
                # --- Ambient emitters ---
                # If no mask is detected, emit from ambient regions to keep it varied.
                if not have_mask and cfg.ambient_emitters > 0:
                    ambient.step(dt, sim)

                # --- Palette cycling ---
                # Auto-cycle through color palettes.
                if palette_cycle_on:
                    if not in_fade:
                        dwell_t += dt
                        if dwell_t >= cfg.palette_dwell:
                            in_fade = True
                            fade_t = 0.0
                            next_pal = (curr_pal + 1) % N_PALETTES
                    else:
                        fade_t += dt
                        mix = min(1.0, fade_t / max(0.001, cfg.palette_fade))
                        sim.set_palette_blend(curr_pal, next_pal, mix)
                        if mix >= 1.0:
                            # commit and start next dwell
                            curr_pal = next_pal
                            dwell_t = 0.0
                            in_fade = False
                            sim.set_palette_blend(curr_pal, curr_pal, 0.0)
                else:
                    # no auto-cycle; ensure stable state
                    sim.set_palette_blend(
                        curr_pal,
                        curr_pal if not in_fade else next_pal,
                        (
                            0.0
                            if not in_fade
                            else min(1.0, fade_t / max(0.001, cfg.palette_fade))
                        ),
                    )

                # --- Simulation step ---
                # This is where all the fluid magic happens.
                sim.step(dt, have_mask)

            # --- Render to screen ---
            # All simulation work is done in offscreen framebuffers.
            # This renders the final result to the screen.
            ctx.screen.use()
            sim.render_split(split_view)

            # --- Performance logging ---
            frame_count += 1
            time_since_log += dt
            if time_since_log >= log_interval:
                fps = frame_count / time_since_log
                logger.info(f"FPS: {fps:.2f} | dt: {dt:.4f}")
                frame_count = 0
                time_since_log = 0.0

            glfw.swap_buffers(win)

    finally:
        seg.release()
        glfw.terminate()
