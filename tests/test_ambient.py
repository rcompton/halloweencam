from unittest.mock import MagicMock
import numpy as np
import pytest

from ghoulfluids.ambient import AmbientController
from ghoulfluids.config import AppConfig


@pytest.fixture
def mock_sim():
    """Provides a mock FluidSim object for testing."""
    sim = MagicMock()
    sim.fbo_vel_b = MagicMock()
    sim.fbo_dye_b = MagicMock()
    sim.vel_a = MagicMock()
    sim.dye_a = MagicMock()
    sim.prog_splat = {
        "field": MagicMock(),
        "point": MagicMock(),
        "value": MagicMock(),
        "radius": MagicMock(),
    }
    sim.vao_splat = MagicMock()
    sim.swap_vel = MagicMock()
    sim.swap_dye = MagicMock()
    return sim


def test_ambient_controller_init():
    cfg = AppConfig()
    cfg.ambient_emitters = 7
    controller = AmbientController(cfg, seed=42)
    assert len(controller.emitters) == 7
    assert controller.vortex_timer == 0.0


def test_ambient_controller_reset():
    cfg = AppConfig()
    controller = AmbientController(cfg, seed=123)
    original_pos = controller.emitters[0]["pos"].copy()

    # Modify state by stepping
    controller.step(0.1, MagicMock())
    controller.vortex_timer = 5.0
    assert not np.array_equal(original_pos, controller.emitters[0]["pos"])

    # Reset and check if state is restored
    controller.reset(seed=123)
    assert controller.vortex_timer == 0.0
    assert np.array_equal(original_pos, controller.emitters[0]["pos"])


def test_step_updates_emitters_and_splats(mock_sim):
    cfg = AppConfig()
    cfg.ambient_emitters = 1
    controller = AmbientController(cfg, seed=42)
    initial_pos = controller.emitters[0]["pos"].copy()

    controller.step(dt=0.1, sim=mock_sim)

    final_pos = controller.emitters[0]["pos"]
    assert not np.array_equal(initial_pos, final_pos)

    # Check that it splats velocity and dye
    assert mock_sim.swap_vel.call_count == 1
    assert mock_sim.swap_dye.call_count == 1

    # Check that render was called for both splats
    assert mock_sim.vao_splat.render.call_count == 2


def test_vortex_is_triggered_after_interval(mock_sim):
    cfg = AppConfig()
    cfg.vortex_interval = 1.0  # Short interval for testing
    cfg.ambient_emitters = 1
    controller = AmbientController(cfg, seed=42)

    # Before trigger time
    controller.step(dt=0.5, sim=mock_sim)
    assert mock_sim.swap_vel.call_count == 1  # Just the emitter splat

    # After trigger time
    controller.step(dt=0.6, sim=mock_sim)
    # 1 call from before + 1 emitter splat + 2 vortex splats
    assert mock_sim.swap_vel.call_count == 1 + 1 + 2
    assert controller.vortex_timer == 0.0  # Timer should reset
