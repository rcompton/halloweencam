from ghoulfluids.config import AppConfig
from ghoulfluids.eyes import EyesController


def _run(eyes, secs, person=False, dt=1 / 30):
    alphas = []
    for _ in range(int(secs / dt)):
        alpha, pos, openness = eyes.step(dt, person)
        alphas.append(alpha)
        assert 0.0 <= alpha <= 1.0 and 0.0 <= openness <= 1.0
    return alphas


def test_eyes_appear_only_after_empty_and_then_fade_out():
    cfg = AppConfig(eyes_min_empty=5.0, eyes_min_wait=1.0, eyes_max_wait=1.0)
    eyes = EyesController(cfg)
    assert max(_run(eyes, 4.9)) == 0.0  # not before the room has been empty long enough
    alphas = _run(eyes, 2 * cfg.eyes_fade + cfg.eyes_hold + 1.0)
    assert max(alphas) == 1.0
    assert alphas[-1] == 0.0


def test_eyes_leave_quickly_when_someone_arrives():
    cfg = AppConfig(eyes_min_empty=0.0, eyes_min_wait=0.1, eyes_max_wait=0.1)
    eyes = EyesController(cfg)
    _run(eyes, 0.1 + cfg.eyes_fade + 0.5)
    assert eyes.alpha == 1.0
    assert _run(eyes, 0.5, person=True)[-1] == 0.0


def test_eyes_disabled():
    cfg = AppConfig(eyes=False, eyes_min_empty=0.0, eyes_min_wait=0.1, eyes_max_wait=0.1)
    assert max(_run(EyesController(cfg), 10.0)) == 0.0
