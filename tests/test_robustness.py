from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ghoulfluids import app, segmentation
from ghoulfluids.segmentation import CameraReader


class _Reader(CameraReader):
    pass


@pytest.fixture
def clock(monkeypatch):
    t = [0.0]
    monkeypatch.setattr(segmentation.time, "monotonic", lambda: t[0])
    return t


def _failing_cap():
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (False, None)
    return cap


def test_camera_reopens_then_recovers(clock):
    bad, good = _failing_cap(), _failing_cap()
    good.read.return_value = (True, np.zeros((4, 4, 3), np.uint8))
    with patch("cv2.VideoCapture", side_effect=[bad, good]) as vc:
        r = _Reader()
        r._init_camera(0, 640, 480)
        assert r._read_camera() is None
        clock[0] = segmentation.CAMERA_FAIL_SECS + 0.1
        assert r._read_camera() is None  # triggers reopen
        assert vc.call_count == 2
        bad.release.assert_called_once()
        assert r._read_camera() is not None


def test_camera_gives_up_after_failed_reopen(clock):
    with patch("cv2.VideoCapture", side_effect=[_failing_cap(), _failing_cap()]):
        r = _Reader()
        r._init_camera(0, 640, 480)
        r._read_camera()
        clock[0] = segmentation.CAMERA_FAIL_SECS + 0.1
        r._read_camera()  # reopen
        clock[0] = 2 * segmentation.CAMERA_FAIL_SECS + 0.2
        with pytest.raises(RuntimeError):
            r._read_camera()


def test_watchdog_exits_when_loop_stalls():
    with patch("ghoulfluids.app.os._exit") as exit_:
        exit_.side_effect = lambda code: stop.set()
        tick, stop = app.start_watchdog(MagicMock(), timeout=0.1)
        assert stop.wait(5.0)
        exit_.assert_called_once_with(3)


def test_watchdog_quiet_while_ticking():
    import time

    with patch("ghoulfluids.app.os._exit") as exit_:
        tick, stop = app.start_watchdog(MagicMock(), timeout=0.5)
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            tick()
            time.sleep(0.05)
        stop.set()
        exit_.assert_not_called()
