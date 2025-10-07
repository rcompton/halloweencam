from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO


class MediaPipeSegmenter:
    def __init__(self, camera_index: int, width: int, height: int):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        mp_seg = mp.solutions.selfie_segmentation
        self.segmenter = mp_seg.SelfieSegmentation(model_selection=0)

    def read_frame_and_mask(self, sim_w: int, sim_h: int, win_w: int, win_h: int):
        """
        Returns (frame_bgr, cam_rgb_flippedV, mask_small_flippedV, mask_area_frac)
        mask_small matches (sim_w,sim_h), both flipped vertically for GL UV convention.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, 0.0

        frame = cv2.flip(frame, 1)  # mirror for user-view
        # segmentation.py (inside read_frame_and_mask)
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb_flipped = cv2.flip(cam_rgb, 0)

        # ✅ Ensure the camera texture matches the window size (so tex.write size matches)
        if cam_rgb_flipped.shape[1] != win_w or cam_rgb_flipped.shape[0] != win_h:
            cam_rgb_flipped = cv2.resize(
                cam_rgb_flipped, (win_w, win_h), interpolation=cv2.INTER_AREA
            )

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
    def __init__(self, camera_index: int, width: int, height: int, model_name: str):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.model = YOLO(model_name)

    def read_frame_and_mask(self, sim_w: int, sim_h: int, win_w: int, win_h: int):
        """
        Returns (frame_bgr, cam_rgb_flippedV, mask_small_flippedV, mask_area_frac)
        mask_small matches (sim_w,sim_h), both flipped vertically for GL UV convention.
        """
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, 0.0

        frame = cv2.flip(frame, 1)  # mirror for user-view
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb_flipped = cv2.flip(cam_rgb, 0)

        # ✅ Ensure the camera texture matches the window size (so tex.write size matches)
        if cam_rgb_flipped.shape[1] != win_w or cam_rgb_flipped.shape[0] != win_h:
            cam_rgb_flipped = cv2.resize(
                cam_rgb_flipped, (win_w, win_h), interpolation=cv2.INTER_AREA
            )

        results = self.model(cam_rgb, classes=[0], verbose=False)  # class 0 is 'person'

        if not results or not results[0].masks:
            return frame, cam_rgb_flipped, None, 0.0

        # Combine masks for all detected persons
        combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        for mask_tensor in results[0].masks.data:
            mask_np = mask_tensor.cpu().numpy()
            # The mask might be smaller than the frame, resize it
            if mask_np.shape != (frame.shape[0], frame.shape[1]):
                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
            combined_mask = np.maximum(combined_mask, mask_np)

        if combined_mask.max() == 0:
            return frame, cam_rgb_flipped, None, 0.0

        m_big = cv2.resize(
            combined_mask, (win_w, win_h), interpolation=cv2.INTER_LINEAR
        )
        area = float((m_big > 0.30).mean())

        m_small = cv2.flip(
            cv2.resize(combined_mask, (sim_w, sim_h), interpolation=cv2.INTER_LINEAR),
            0,
        )
        return frame, cam_rgb_flipped, m_small, area

    def release(self):
        self.cap.release()
