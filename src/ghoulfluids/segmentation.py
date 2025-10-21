from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
import torch
from ultralytics import YOLO

from .profiler import get_profiler


class MediaPipeSegmenter:
    def __init__(
        self,
        camera_index: int,
        cam_w: int,
        cam_h: int,
        mirror: bool = True,
    ):
        self.profiler = get_profiler()
        self.cap = cv2.VideoCapture(camera_index)
        self.mirror = mirror
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        mp_seg = mp.solutions.selfie_segmentation
        self.segmenter = mp_seg.SelfieSegmentation(model_selection=0)

    def read_frame_and_mask(self, sim_w: int, sim_h: int, win_w: int, win_h: int):
        """
        Returns (frame_bgr, cam_rgb_flippedV, mask_small_flippedV, mask_area_frac)
        mask_small matches (sim_w,sim_h), both flipped vertically for GL UV convention.
        """
        with self.profiler.record("cam_read"):
            ok, frame = self.cap.read()
        if not ok:
            return None, None, None, 0.0

        if self.mirror:
            frame = cv2.flip(frame, 1)
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb_flipped = cv2.flip(cam_rgb, 0)

        # ✅ Ensure the camera texture matches the window size (so tex.write size matches)
        if cam_rgb_flipped.shape[1] != win_w or cam_rgb_flipped.shape[0] != win_h:
            cam_rgb_flipped = cv2.resize(
                cam_rgb_flipped, (win_w, win_h), interpolation=cv2.INTER_AREA
            )

        with self.profiler.record("mediapipe_process"):
            res = self.segmenter.process(cam_rgb)
        mask = res.segmentation_mask
        if mask is None:
            return frame, cam_rgb_flipped, None, 0.0

        m_big = cv2.resize(
            mask.astype(np.float32), (win_w, win_h), interpolation=cv2.INTER_LINEAR
        )
        area = float((m_big > 0.30).mean())

        m_small = cv2.flip(
            cv2.resize(
                mask.astype(np.float32), (sim_w, sim_h), interpolation=cv2.INTER_LINEAR
            ),
            0,
        )
        return frame, cam_rgb_flipped, m_small, area

    def release(self):
        self.cap.release()
        self.segmenter.close()


class YOLOSegmenter:
    def __init__(
        self,
        camera_index: int,
        cam_w: int,
        cam_h: int,
        model_name: str,
        seg_w: int,
        seg_h: int,
        device: str = "cuda",
        mirror: bool = True,
    ):
        self.profiler = get_profiler()
        self.cap = cv2.VideoCapture(camera_index)
        self.mirror = mirror
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        self.model = YOLO(model_name)
        self.seg_w = seg_w
        self.seg_h = seg_h
        self.device = device

    def _preprocess_image(self, img: np.ndarray) -> torch.Tensor:
        """Converts a NumPy image to a pre-processed torch tensor."""
        # Ensure contiguous array
        img = np.ascontiguousarray(img)
        # Convert to tensor
        tensor = torch.from_numpy(img).to(self.device)
        # Use half precision on GPU
        if self.device == "cuda":
            tensor = tensor.half()
        # HWC to CHW, add batch dimension
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        # Normalize
        tensor = tensor / 255.0
        return tensor

    def read_frame_and_mask(self, sim_w: int, sim_h: int, win_w: int, win_h: int):
        """
        Returns (frame_bgr, cam_rgb_flippedV, mask_small_flippedV, mask_area_frac)
        mask_small matches (sim_w,sim_h), both flipped vertically for GL UV convention.
        """
        with self.profiler.record("cam_read"):
            ok, frame = self.cap.read()
        if not ok:
            return None, None, None, 0.0

        if self.mirror:
            frame = cv2.flip(frame, 1)
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb_flipped = cv2.flip(cam_rgb, 0)

        # ✅ Ensure the camera texture matches the window size (so tex.write size matches)
        if cam_rgb_flipped.shape[1] != win_w or cam_rgb_flipped.shape[0] != win_h:
            cam_rgb_flipped = cv2.resize(
                cam_rgb_flipped, (win_w, win_h), interpolation=cv2.INTER_AREA
            )

        with self.profiler.record("yolo_preprocess"):
            # Resize for the model input
            model_input_frame = cv2.resize(
                cam_rgb, (self.seg_w, self.seg_h), interpolation=cv2.INTER_LINEAR
            )
            input_tensor = self._preprocess_image(model_input_frame)

        with self.profiler.record("yolo_inference"):
            results = self.model(
                input_tensor,
                classes=[0],
                verbose=False,
                imgsz=(self.seg_h, self.seg_w),
            )  # class 0 is 'person'

        with self.profiler.record("yolo_postprocess"):
            if not results or not results[0].masks:
                return frame, cam_rgb_flipped, None, 0.0

            # Combine masks at their native resolution from the model
            masks = results[0].masks.data
            if len(masks) > 0:
                # Start with the first mask and add the others
                combined_mask_native = masks[0].cpu().numpy()
                for i in range(1, len(masks)):
                    combined_mask_native = np.maximum(
                        combined_mask_native, masks[i].cpu().numpy()
                    )
            else:
                combined_mask_native = np.zeros(
                    (self.seg_h, self.seg_w), dtype=np.float32
                )

            if combined_mask_native.max() == 0:
                return frame, cam_rgb_flipped, None, 0.0

        m_big = cv2.resize(
            combined_mask_native.astype(np.float32),
            (win_w, win_h),
            interpolation=cv2.INTER_LINEAR,
        )
        area = float((m_big > 0.30).mean())

        m_small = cv2.flip(
            cv2.resize(
                combined_mask_native.astype(np.float32),
                (sim_w, sim_h),
                interpolation=cv2.INTER_LINEAR,
            ),
            0,
        )
        return frame, cam_rgb_flipped, m_small, area

    def release(self):
        self.cap.release()
