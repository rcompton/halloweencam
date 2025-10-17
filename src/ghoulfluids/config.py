from dataclasses import dataclass


@dataclass
class AppConfig:
    # --- Debugging ---
    debug: bool = False  # Show debug overlay

    # --- Logging ---
    log_level: str = "INFO"  # Minimum log level to output
    log_file: str | None = None  # Redirect logs to a file instead of the console

    # --- Window and Rendering ---
    width: int = 1024  # Initial window width
    height: int = 576  # Initial window height
    render_scale: float = 0.5  # Dye texture resolution relative to window size
    dye_fp16: bool = True  # Use half-float (16-bit) for dye textures for performance

    # --- Simulation Core ---
    sim_scale: float = 0.5  # Simulation grid resolution relative to window size
    sim_max_dim: int = (
        1024  # Maximum dimension (width or height) for the simulation grid
    )
    substeps: int = 10  # Number of simulation substeps per frame
    dt_clamp: float = 0.03  # Maximum time delta to prevent instability (in seconds)
    jacobi_iters: int = (
        40  # Number of iterations for the Jacobi solver (pressure projection)
    )
    vorticity_eps: float = 1.9  # Vorticity confinement strength; 0 to disable
    vel_dissipation: float = 0.999  # Velocity dissipation factor per step (damping)
    dye_dissipation: float = 0.997  # Dye dissipation factor per step (fading)
    background_velocity: float = (
        0.03  # Constant upward velocity, scaled by sim height, for a fire-like effect
    )

    # --- Camera ---
    camera_index: int = 0  # Index of the camera to use (e.g., 0 for /dev/video0)

    # --- Segmentation ---
    segmenter: str = "mediapipe"  # Segmentation backend ('mediapipe' or 'yolo')
    seg_width: int | None = 640  # Fixed width for the segmentation mask
    seg_height: int | None = 384  # Fixed height for the segmentation mask
    yolo_model: str = "yolo11s-seg.pt"  # Path to the YOLO model file
    mask_threshold: float = 0.25  # Confidence threshold for segmentation masks
    mask_min_area: float = (
        0.01  # Minimum area fraction for a mask to be considered valid
    )

    # --- Edge Forces (from mask) ---
    force_mode: str = "edges"  # How to apply forces from the mask ('edges' or 'full')
    edge_thresh: float = 0.02  # Threshold for detecting edges in the segmentation mask
    edge_normal_amp: float = (
        5.5  # Strength of the force pushing away from the mask edge
    )
    edge_tangent_amp: float = 2.2  # Strength of the force moving along the mask edge
    edge_use_temporal: bool = (
        True  # Use mask from the previous frame to calculate velocity
    )
    edge_dye_strength: float = 0.06  # How much dye to emit from the mask edges

    # --- Ambient (when no mask is detected) ---
    ambient_emitters: int = 30  # Number of drifting wisps to simulate
    ambient_speed: float = 0.10  # Speed of ambient emitters (in UV units per second)
    ambient_radius: float = 0.050  # Splat radius for ambient emitters (in UV units)
    ambient_dye: float = 0.07  # Dye intensity per ambient splat
    ambient_vel_amp: float = 0.25  # Velocity injection strength for ambient emitters
    ambient_jitter: float = 0.35  # Random wiggle factor for emitter paths (0 to 1)
    ambient_margin: float = 0.05  # Keep emitters away from the simulation borders

    # --- Vortices (periodic kicks) ---
    vortex_interval: float = 1.6  # Seconds between applying vortex kicks
    vortex_strength: float = 1.8  # Tangential velocity strength of the vortices
    vortex_radius: float = 0.050  # Distance of the vortex dipole from the center

    # --- Palette ---
    palette_on: int = 1  # Whether to use the color palettes (1 for on, 0 for off)
    palette_id: int = 5  # Initial palette index (0 to 5)
    palette_cycle: bool = True  # Whether to automatically cycle through palettes
    palette_dwell: float = 15.0  # Seconds to stay on each palette
    palette_fade: float = 5.0  # Seconds to fade between palettes
