import pytest, moderngl
from ghoulfluids.config import AppConfig


@pytest.fixture(scope="module")
def ctx():
    try:
        return moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"Could not create headless GL context: {e}")


@pytest.fixture(scope="module")
def small_cfg():
    # tiny, fast config for tests
    return AppConfig(
        width=256,
        height=144,
        sim_scale=0.5,
        substeps=1,
        dt_clamp=0.016,
        jacobi_iters=6,
        vorticity_eps=1.5,
        vel_dissipation=1.0,
        dye_dissipation=1.0,
        edge_dye_strength=0.0,
    )
