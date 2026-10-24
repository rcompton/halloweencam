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
    render_scale: float = 0.6  # Dye texture resolution relative to window size
    dye_max_dim: int = 1024  # Maximum dimension (width or height) for the dye texture
    dye_fp16: bool = True  # Use half-float (16-bit) for dye textures for performance

    # --- Simulation Core ---
    sim_scale: float = 0.6  # Simulation grid resolution relative to window size
    sim_max_dim: int = (
        1024  # Maximum dimension (width or height) for the simulation grid
    )
    substeps: int = 12  # Number of simulation substeps per frame
    dt_clamp: float = 0.03  # Maximum time delta to prevent instability (in seconds)
    jacobi_iters: int = (
        30  # Number of iterations for the Jacobi solver (pressure projection)
    )
    vorticity_eps: float = 2.6  # Vorticity confinement strength; 0 to disable
    vel_dissipation: float = 0.998  # Velocity dissipation factor per step (damping)
    dye_dissipation: float = 0.994  # Dye dissipation factor per step (fading)
    background_velocity: float = (
        0.095  # Constant upward velocity, scaled by sim height, for a fire-like effect
    )
    open_top: bool = True  # Smoke flows out the top edge instead of piling up there
    top_fade: float = 0.06  # Band at the top (UV) where dye fades out; 0 to disable

    # --- Camera ---
    camera_index: int = 0  # Index of the camera to use (e.g., 0 for /dev/video0)
    mirror: bool = True  # Mirror the camera feed horizontally

    # --- Segmentation ---
    segmenter: str = "yolo"  # Segmentation backend ('mediapipe' or 'yolo')
    seg_width: int | None = 640  # Fixed width for the segmentation mask
    seg_height: int | None = 384  # Fixed height for the segmentation mask
    yolo_model: str = "yolo11s-seg.pt"  # Path to the YOLO model file
    yolo_device: str = "cuda"  # Device for YOLO inference ('cuda', 'cpu', 'mps')
    mask_threshold: float = 0.25  # Confidence threshold for segmentation masks
    mask_min_area: float = (
        0.01  # Minimum area fraction for a mask to be considered valid
    )

    # --- Edge Forces (from mask) ---
    force_mode: str = "edges"  # How to apply forces from the mask ('edges' or 'full')
    edge_thresh: float = 0.02  # Threshold for detecting edges in the segmentation mask
    edge_normal_amp: float = (
        15.5  # Strength of the force pushing away from the mask edge
    )
    edge_tangent_amp: float = 5.5  # Strength of the force moving along the mask edge
    edge_use_temporal: bool = (
        True  # Use mask from the previous frame to calculate velocity
    )
    edge_dye_strength: float = 0.10  # How much dye to emit from the mask edges
    edge_dye_cap: float = 1.0  # Edges stop adding dye where it's already this bright
    edge_border: float = 0.03  # Ignore mask edges this close to the frame border (UV);
    # YOLO masks stop a few pixels short of the frame, which made the walls spray smoke

    # --- Ambient (when no mask is detected) ---
    ambient_emitters: int = 50  # Number of drifting wisps to simulate
    ambient_speed: float = 0.25  # Speed of ambient emitters (in UV units per second)
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
    palette_dwell: float = 1.0  # Seconds to stay on each palette
    palette_fade: float = 5.0  # Seconds to fade between palettes

    # --- Look ---
    glow_strength: float = 0.6  # Soft glow around bright smoke; 0 to disable
    glow_radius: float = 0.025  # Glow spread (in UV units of screen height)
    vignette: float = 0.6  # Darken screen edges; 0 to disable, 1 = black corners

    # --- Effects ---
    ghost_strength: float = 0.0  # Pale after-image trail behind people; 0 to disable
    ghost_fade: float = 0.6  # Seconds for the trail to fade to ~37%
    ghost_color: tuple = (0.55, 0.8, 1.0)  # Cold blue-white
    crowd_boost: float = 1.0  # Extra edge force/dye when people fill the frame (1 = up to 2x)
    ambient_with_person: float = 0.35  # Background wisps keep going at this strength when someone's there
    eyes: bool = True  # Glowing demon eyes appear in the smoke while nobody's in front
    eyes_count: int = 4  # Pairs that can be on screen at once (max 4)
    eyes_flee: bool = True  # Fade out when someone shows up (False: they stay and watch)
    eyes_min_empty: float = 0.0  # Only appear after nobody's been seen this long; 0 = always
    eyes_min_wait: float = 2.0  # Random wait per pair between appearances (seconds)
    eyes_max_wait: float = 8.0
    eyes_fade: float = 1.0  # Fade in/out seconds
    eyes_hold: float = 3.0  # Seconds fully visible (blinks happen here)
    eyes_size: float = 0.05  # Eye width as a fraction of screen height (each pair varies 0.6-1.4x)
