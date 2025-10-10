import pytest
from unittest.mock import MagicMock, patch, call
from ghoulfluids import app


@pytest.fixture
def mock_glfw():
    """Provides a mocked glfw module."""
    with patch("ghoulfluids.app.glfw") as mock:
        mock.init.return_value = True
        mock.get_primary_monitor.return_value = MagicMock()
        mock.get_video_mode.return_value = MagicMock(
            size=MagicMock(width=1920, height=1080)
        )
        mock.create_window.return_value = MagicMock()
        # Simulate a few frames and then exit
        mock.window_should_close.side_effect = [False, False, True]
        yield mock


@pytest.fixture
def mock_moderngl():
    """Provides a mocked moderngl module."""
    with patch("ghoulfluids.app.moderngl") as mock:
        mock.create_context.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_segmenters():
    """Provides mocked segmenter classes."""
    with (
        patch("ghoulfluids.app.YOLOSegmenter") as mock_yolo,
        patch("ghoulfluids.app.MediaPipeSegmenter") as mock_mediapipe,
    ):

        mock_yolo.return_value.read_frame_and_mask.return_value = (None, None, None, 0)
        mock_mediapipe.return_value.read_frame_and_mask.return_value = (
            None,
            None,
            None,
            0,
        )

        yield {"yolo": mock_yolo, "mediapipe": mock_mediapipe}


def test_app_main_defaults(mock_glfw, mock_moderngl, mock_segmenters):
    """Test the main function with default arguments."""
    app.main([])
    # Check that mediapipe is used by default
    mock_segmenters["mediapipe"].assert_called_once()
    mock_segmenters["yolo"].assert_not_called()
    # Check that the main loop runs
    assert mock_glfw.poll_events.call_count > 1
    assert mock_glfw.swap_buffers.call_count > 1


def test_app_main_yolo_segmenter(mock_glfw, mock_moderngl, mock_segmenters):
    """Test the main function with the YOLO segmenter."""
    app.main(["--segmenter", "yolo"])
    mock_segmenters["yolo"].assert_called_once()
    mock_segmenters["mediapipe"].assert_not_called()


def test_app_main_split_view(mock_glfw, mock_moderngl, mock_segmenters):
    """Test the main function with split view enabled."""
    with patch("ghoulfluids.app.FluidSim") as mock_sim:
        instance = mock_sim.return_value
        app.main(["--split"])
        # Check that render_split is called with True
        instance.render_split.assert_has_calls([call(True), call(True)])


def test_app_main_debug_mode(mock_glfw, mock_moderngl, mock_segmenters):
    """Test the main function with debug mode enabled."""
    with patch("ghoulfluids.app.DebugOverlay") as mock_overlay:
        app.main(["--debug"])
        mock_overlay.assert_called_once()
        # Check that the render method is called
        instance = mock_overlay.return_value
        assert instance.render.call_count > 0
