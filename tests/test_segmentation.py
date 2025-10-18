import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import glfw

from ghoulfluids.config import AppConfig
from ghoulfluids.segmentation import MediaPipeSegmenter, YOLOSegmenter
from ghoulfluids.app import main


@pytest.fixture
def mock_video_capture():
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = mock_cap.return_value
        mock_instance.isOpened.return_value = True
        mock_instance.read.return_value = (
            True,
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        yield mock_instance


@patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
def test_mediapipe_segmenter_init(mock_mp, mock_video_capture):
    segmenter = MediaPipeSegmenter(0, 1920, 1080)
    assert segmenter.cap == mock_video_capture
    mock_mp.assert_called_once()


@patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
def test_mediapipe_segmenter_read_frame_and_mask(mock_mp, mock_video_capture):
    mock_segmentation = mock_mp.return_value
    mock_results = MagicMock()
    mock_results.segmentation_mask = np.ones((480, 640), dtype=np.float32)
    mock_segmentation.process.return_value = mock_results

    segmenter = MediaPipeSegmenter(0, 640, 480)
    frame, cam_rgb, mask, area = segmenter.read_frame_and_mask(320, 240, 640, 480)

    assert frame is not None
    assert cam_rgb is not None
    assert mask is not None
    assert area > 0


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_init(mock_yolo, mock_video_capture):
    segmenter = YOLOSegmenter(0, 1920, 1080, "yolov8n-seg.pt", 640, 480, device="cpu")
    assert segmenter.cap == mock_video_capture
    mock_yolo.assert_called_once_with("yolov8n-seg.pt")


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_read_frame_and_mask(mock_yolo, mock_video_capture):
    mock_model = mock_yolo.return_value

    # Mock YOLO model output
    mock_mask = MagicMock()
    mock_mask.data = [MagicMock()]
    # Make the mask the same size as the frame to avoid resize issues in test
    mock_mask.data[0].cpu().numpy.return_value = np.ones((480, 640), dtype=np.float32)
    mock_results = [MagicMock()]
    mock_results[0].masks = mock_mask
    mock_model.return_value = mock_results

    segmenter = YOLOSegmenter(0, 640, 480, "yolov8n-seg.pt", 640, 480, device="cpu")
    frame, cam_rgb, mask, area = segmenter.read_frame_and_mask(320, 240, 640, 480)

    assert frame is not None
    assert cam_rgb is not None
    assert mask is not None
    assert area > 0


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_read_frame_no_mask(mock_yolo, mock_video_capture):
    mock_model = mock_yolo.return_value
    mock_model.return_value = []  # No results

    segmenter = YOLOSegmenter(0, 640, 480, "yolov8n-seg.pt", 640, 480, device="cpu")
    frame, cam_rgb, mask, area = segmenter.read_frame_and_mask(320, 240, 640, 480)

    assert frame is not None
    assert cam_rgb is not None
    assert mask is None
    assert area == 0.0


@patch("ghoulfluids.app.glfw")
@patch("moderngl.create_context")
@patch("ghoulfluids.app.FluidSim")
@patch("ghoulfluids.app.AmbientController")
@patch("ghoulfluids.app.MediaPipeSegmenter")
@patch("ghoulfluids.app.YOLOSegmenter")
def test_app_main_segmenter_selection(
    mock_yolo, mock_mp, mock_ambient, mock_fluid, mock_gl, mock_glfw
):
    mock_glfw.init.return_value = True
    mock_glfw.get_key.return_value = glfw.RELEASE

    # Set up mock return values for segmenter instances
    mock_frame = np.zeros((10, 10, 3), np.uint8)
    mock_mask = np.zeros((5, 5), np.float32)
    mock_return = (mock_frame, mock_frame, mock_mask, 0.5)
    mock_mp.return_value.read_frame_and_mask.return_value = mock_return
    mock_yolo.return_value.read_frame_and_mask.return_value = mock_return

    # Test default (mediapipe)
    with patch.object(mock_glfw, "window_should_close", side_effect=[False, True]):
        main([])
    mock_mp.assert_called()
    mock_yolo.assert_not_called()
    mock_mp.return_value.read_frame_and_mask.assert_called()

    mock_mp.reset_mock()
    mock_yolo.reset_mock()

    # Test explicit mediapipe
    with patch.object(mock_glfw, "window_should_close", side_effect=[False, True]):
        main(["--segmenter", "mediapipe"])
    mock_mp.assert_called()
    mock_yolo.assert_not_called()
    mock_mp.return_value.read_frame_and_mask.assert_called()

    mock_mp.reset_mock()
    mock_yolo.reset_mock()

    # Test yolo
    with patch.object(mock_glfw, "window_should_close", side_effect=[False, True]):
        main(["--segmenter", "yolo"])
    mock_mp.assert_not_called()
    mock_yolo.assert_called()
    mock_yolo.return_value.read_frame_and_mask.assert_called()


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_mask_resize(mock_yolo, mock_video_capture):
    """
    Verify that the output mask is resized to the simulation dimensions (sim_w, sim_h)
    and not the segmentation dimensions (seg_w, seg_h) or the window dimensions.
    """
    mock_model = mock_yolo.return_value

    # Mock YOLO model output - a mask of the same size as the model input
    mock_mask_data = np.ones((240, 320), dtype=np.float32)
    mock_mask_tensor = MagicMock()
    mock_mask_tensor.cpu().numpy.return_value = mock_mask_data

    mock_masks = MagicMock()
    mock_masks.data = [mock_mask_tensor]

    mock_results = [MagicMock()]
    mock_results[0].masks = mock_masks
    mock_model.return_value = mock_results

    # Use distinct dimensions for each component
    win_w, win_h = 1280, 720
    seg_w, seg_h = 320, 240
    sim_w, sim_h = 160, 120

    segmenter = YOLOSegmenter(
        0, win_w, win_h, "yolov8n-seg.pt", seg_w, seg_h, device="cpu"
    )
    _, _, mask, _ = segmenter.read_frame_and_mask(sim_w, sim_h, win_w, win_h)

    assert mask is not None
    # The final mask should be flipped vertically for OpenGL, hence (sim_h, sim_w)
    assert mask.shape == (sim_h, sim_w)
