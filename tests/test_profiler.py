import time
from unittest.mock import MagicMock, patch
import pytest
from ghoulfluids.profiler import Profiler, get_profiler


@pytest.fixture
def profiler():
    """Returns a new Profiler instance for each test."""
    # Reset the global profiler instance before each test
    with patch("ghoulfluids.profiler._profiler", None):
        yield get_profiler()


def test_get_profiler_singleton():
    """Test that get_profiler always returns the same instance."""
    profiler1 = get_profiler()
    profiler2 = get_profiler()
    assert profiler1 is profiler2


def test_profiler_record(profiler):
    """Test the record context manager."""
    with profiler.record("test_op"):
        time.sleep(0.01)

    timings = profiler.get_timings()
    assert "test_op" in timings
    assert timings["test_op"] > 0.0


def test_profiler_get_timings(profiler):
    """Test the get_timings method."""
    with profiler.record("op1"):
        time.sleep(0.01)
    with profiler.record("op2"):
        time.sleep(0.02)

    timings = profiler.get_timings()
    assert "op1" in timings
    assert "op2" in timings
    assert timings["op1"] < timings["op2"]


@patch("ghoulfluids.profiler.get_logger")
def test_profiler_log_stats(mock_get_logger):
    """Test that log_stats calls the logger with the correct stats."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # Instantiate Profiler after the logger is mocked
    profiler = Profiler()

    with profiler.record("log_op"):
        time.sleep(0.01)

    profiler.log_stats()

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "log_op" in call_args
    assert "ms" in call_args
