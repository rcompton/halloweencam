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


@pytest.fixture
def yolo_cfg():
    cfg = AppConfig()
    cfg.camera_index = 0
    cfg.yolo_model = "yolov8n-seg.pt"
    return cfg


@patch("mediapipe.solutions.selfie_segmentation.SelfieSegmentation")
def test_mediapipe_segmenter_init(mock_mp, mock_video_capture):
    segmenter = MediaPipeSegmenter(0, 1920, 1080)
    assert segmenter.cap == mock_video_capture
    mock_mp.assert_called_once()


@patch("pathlib.Path.exists", return_value=False)
@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_init(mock_yolo, mock_exists, mock_video_capture, yolo_cfg):
    segmenter = YOLOSegmenter(yolo_cfg)
    assert segmenter.cap == mock_video_capture
    mock_yolo.assert_called_once()
    args, _ = mock_yolo.call_args
    assert str(args[0]) == "yolov8n-seg.pt"


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_read_frame_and_mask(mock_yolo, mock_video_capture, yolo_cfg):
    mock_model = mock_yolo.return_value

    # Mock YOLO model output
    mock_predict_result = MagicMock()
    mock_masks = MagicMock()

    # Mock tensor chain
    mock_tensor = MagicMock()
    mock_tensor.float.return_value.max.return_value.values.detach.return_value.to.return_value.numpy.return_value = np.ones((yolo_cfg.mask_in_w, yolo_cfg.mask_in_w), dtype=np.float32)

    mock_masks.data = mock_tensor
    mock_predict_result.masks = mock_masks
    mock_model.predict.return_value = [mock_predict_result]

    segmenter = YOLOSegmenter(yolo_cfg)
    frame, cam_rgb, mask, area = segmenter.read_frame_and_mask(320, 240, 640, 480)

    assert frame is not None
    assert cam_rgb is not None
    assert mask is not None
    assert area > 0


@patch("ghoulfluids.segmentation.YOLO")
def test_yolo_segmenter_read_frame_no_mask(mock_yolo, mock_video_capture, yolo_cfg):
    mock_model = mock_yolo.return_value

    # Mock YOLO model output
    mock_predict_result = MagicMock()
    mock_predict_result.masks = None
    mock_model.predict.return_value = [mock_predict_result]

    segmenter = YOLOSegmenter(yolo_cfg)
    frame, cam_rgb, mask, area = segmenter.read_frame_and_mask(320, 240, 640, 480)

    assert frame is not None
    assert cam_rgb is not None
    assert mask is not None
    assert np.all(mask == 0)
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
