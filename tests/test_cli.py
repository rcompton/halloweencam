from unittest.mock import patch, MagicMock

from ghoulfluids.app import main
from ghoulfluids.config import AppConfig


@patch("ghoulfluids.app.glfw.terminate")
@patch("ghoulfluids.app.glfw.swap_buffers")
@patch("ghoulfluids.app.glfw.window_should_close")
@patch("ghoulfluids.app.glfw.make_context_current")
@patch("ghoulfluids.app.YOLOSegmenter")
@patch("ghoulfluids.app.FluidSim")
@patch("ghoulfluids.app.moderngl.create_context")
@patch("ghoulfluids.app.glfw.create_window")
@patch("ghoulfluids.app.glfw.init")
def test_cli_args_smoke_test(
    mock_glfw_init,
    mock_create_window,
    mock_create_context,
    mock_fluid_sim,
    mock_yolo_segmenter,
    mock_make_context_current,
    mock_window_should_close,
    mock_swap_buffers,
    mock_glfw_terminate,
):
    """Test that all CLI arguments can be passed without crashing."""
    mock_glfw_init.return_value = True
    # Prevent the main loop from running
    mock_window_should_close.return_value = True

    test_args = [
        "ghoulfluids",
        "--split",
        "--debug",
        "--seg-height",
        "240",
        "--segmenter",
        "yolo",
        "--yolo-model",
        "yolov8n-seg.pt",
        "--fluid-force-mode",
        "full",
        "--log-file",
        "test.log",
    ]

    with patch("sys.argv", test_args):
        try:
            main()
        except SystemExit as e:
            # A clean exit is fine
            assert e.code == 0

    # Check that FluidSim was initialized with a config that reflects the CLI args
    mock_fluid_sim.assert_called_once()
    call_args, _ = mock_fluid_sim.call_args
    config = call_args[1]
    assert isinstance(config, AppConfig)

    assert config.debug is True
    assert config.seg_height == 240
    assert config.segmenter == "yolo"
    assert config.yolo_model == "yolov8n-seg.pt"
    assert config.force_mode == "full"
    assert config.log_file == "test.log"
