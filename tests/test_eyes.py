from ghoulfluids.config import AppConfig
from ghoulfluids.eyes import EyesController, EyesSwarm


def _run(eyes, secs, person=False, dt=1 / 30):
    alphas = []
    for _ in range(int(secs / dt)):
        alpha, pos, openness, scale = eyes.step(dt, person)
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
    cfg = AppConfig(eyes_flee=True, eyes_min_empty=0.0, eyes_min_wait=0.1, eyes_max_wait=0.1)
    eyes = EyesController(cfg)
    _run(eyes, 0.1 + cfg.eyes_fade + 0.5)
    assert eyes.alpha == 1.0
    assert _run(eyes, 0.5, person=True)[-1] == 0.0


def test_eyes_disabled():
    cfg = AppConfig(eyes=False, eyes_min_empty=0.0, eyes_min_wait=0.1, eyes_max_wait=0.1)
    assert max(_run(EyesController(cfg), 10.0)) == 0.0


def test_eyes_stay_when_people_arrive_by_default():
    cfg = AppConfig(eyes_min_wait=0.1, eyes_max_wait=0.1)
    eyes = EyesController(cfg)
    _run(eyes, 0.1 + cfg.eyes_fade + 0.5, person=True)
    assert eyes.alpha == 1.0


def test_swarm_shows_several_pairs_apart():
    cfg = AppConfig(eyes_count=4, eyes_min_wait=0.1, eyes_max_wait=0.5)
    swarm = EyesSwarm(cfg)
    most = []
    for _ in range(150):
        shown = [(a, pos) for a, pos, *_ in swarm.step(1 / 30, False) if a > 0]
        if len(shown) > len(most):
            most = shown
    assert len(most) >= 3
    pts = [pos for _, pos in most]
    gaps = [((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for i, a in enumerate(pts) for b in pts[i + 1 :]]
    assert min(gaps) > 0.1
