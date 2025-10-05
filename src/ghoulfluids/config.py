from dataclasses import dataclass


@dataclass
class AppConfig:
    width: int = 1024
    height: int = 576
    render_scale: float = 0.6  # 60% of window size by default
    dye_fp16: bool = True  # use half-float RGBA for dye
    sim_scale: float = 0.7
    sim_max_dim: int = 1024
    substeps: int = 6
    dt_clamp: float = 0.033
    jacobi_iters: int = 50
    vorticity_eps: float = 3.0
    vel_dissipation: float = 0.995
    dye_dissipation: float = 0.994
    palette_on: int = 1

    # camera
    camera_index: int = 0
    mask_threshold: float = 0.30
    mask_min_area: float = 0.01

    # edge forces
    edge_thresh: float = 0.02
    edge_normal_amp: float = 4.0
    edge_tangent_amp: float = 1.5
    edge_use_temporal: bool = True
    edge_dye_strength: float = 0.10

    # Ambient (when no mask is detected)
    ambient_emitters: int = 30  # how many drifting wisps
    ambient_speed: float = 0.10  # UV units per second
    ambient_radius: float = 0.050  # splat radius (UV)
    ambient_dye: float = 0.07  # dye intensity per splat
    ambient_vel_amp: float = 0.25  # velocity injection strength (UV/s)
    ambient_jitter: float = 0.35  # random wiggle factor (0..1)
    ambient_margin: float = 0.05  # keep emitters away from walls

    # occasional vortices to keep things rolling
    vortex_interval: float = 1.6  # seconds between kicks
    vortex_strength: float = 1.8  # tangential velocity strength
    vortex_radius: float = 0.050  # how far the dipole is from the center
