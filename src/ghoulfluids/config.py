from dataclasses import dataclass


@dataclass
class AppConfig:
    width: int = 1024
    height: int = 576
    sim_scale: float = 0.7
    substeps: int = 3
    dt_clamp: float = 0.033
    jacobi_iters: int = 28
    vorticity_eps: float = 2.4
    vel_dissipation: float = 0.999
    dye_dissipation: float = 0.999
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

    # fallback ambient dye
    use_fallback_orbit: bool = True
    orbit_r: float = 0.18
    orbit_speed: float = 0.35
    micro_splats_per_frame: int = 3
    micro_splat_radius: float = 0.018
    micro_splat_strength: float = 0.05
