import pytest
from ghoulfluids.config import AppConfig


def test_app_config_defaults():
    """Verify that AppConfig default values are set as expected."""
    cfg = AppConfig()

    # Basic dimensions and scaling
    assert cfg.width > 0 and cfg.height > 0
    assert 0.0 < cfg.sim_scale <= 1.0
    assert 0.0 < cfg.render_scale <= 1.0
    assert cfg.sim_max_dim > 0

    # Simulation parameters
    assert cfg.substeps >= 1
    assert cfg.jacobi_iters >= 1
    assert cfg.dt_clamp > 0.0
    assert 0.0 <= cfg.vel_dissipation <= 1.0
    assert 0.0 <= cfg.dye_dissipation <= 1.0

    # Camera and mask settings
    assert cfg.camera_index >= 0
    assert 0.0 < cfg.mask_threshold < 1.0
    assert cfg.mask_min_area > 0.0

    # Edge force parameters
    assert cfg.edge_thresh >= 0.0
    assert cfg.edge_normal_amp >= 0.0
    assert cfg.edge_tangent_amp >= 0.0

    # Ambient controller settings
    assert cfg.ambient_emitters >= 0
    assert cfg.ambient_speed > 0.0
    assert cfg.vortex_interval > 0.0

    # Palette settings
    assert cfg.palette_on in [0, 1]
    assert cfg.palette_id >= 0
    assert cfg.palette_dwell > 0.0
    assert cfg.palette_fade > 0.0
