from ghoulfluids.config import AppConfig

def test_defaults():
    cfg = AppConfig()
    assert cfg.width > 0 and cfg.height > 0
    assert 0.0 < cfg.sim_scale <= 1.0
    assert cfg.jacobi_iters >= 1
