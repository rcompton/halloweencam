from __future__ import annotations
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import time
import torch
import pathlib


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
    """
    Drop-in replacement for your MediaPipe segmenter.
    - Runs on CUDA (device=0) with FP16
    - img size 512 (default)
    - Decimates to cfg.mask_hz and caches the last mask
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(int(getattr(cfg, "camera_index", 0)))
        # ask for a cheap camera mode (YOLO runs on its own resize anyway)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        model_path_pt = pathlib.Path(getattr(cfg, "yolo_model", "yolov8n-seg.pt"))
        model_path_engine = model_path_pt.with_suffix(".engine")

        if model_path_engine.exists():
            print(f"Loading TensorRT engine: {model_path_engine}")
            model_path = model_path_engine
        else:
            print(f"Loading PyTorch model: {model_path_pt}")
            model_path = model_path_pt

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half = (self.device.startswith("cuda"))
        self.imgsz = int(getattr(cfg, "mask_in_w", 512))
        self.conf  = float(getattr(cfg, "mask_threshold", 0.25))  # reuse your threshold
        self.iou   = 0.45

        torch.backends.cudnn.benchmark = True  # speed for fixed sizes
        self.model = YOLO(model_path)
        # pre-warm once (builds CUDA kernels)
        _ = self.model.predict(np.zeros((self.imgsz, self.imgsz, 3), np.uint8),
                               device=self.device, half=self.half, imgsz=self.imgsz,
                               conf=self.conf, iou=self.iou, verbose=False)

        self._last_mask_small = None
        self._last_mask_time = 0.0
        self._mask_period = 1.0 / float(getattr(cfg, "mask_hz", 20.0))

    def _infer_mask_small(self, small_bgr: np.ndarray) -> np.ndarray:
        """Run YOLO on a small BGR frame; return mask in [0,1] float32"""
        # Ultralytics accepts BGR ndarray
        r = self.model.predict(
            small_bgr,
            device=self.device, half=self.half, imgsz=self.imgsz,
            conf=self.conf, iou=self.iou, verbose=False
        )[0]
        if r.masks is None:
            return np.zeros((small_bgr.shape[0], small_bgr.shape[1]), np.float32)

        # union of instance masks
        m = r.masks.data  # [N, H, W] torch on device
        m = m.float().max(dim=0).values  # [H, W]
        m = m.detach().to("cpu").numpy().astype(np.float32)
        # quick denoise
        m = cv2.medianBlur((m*255).astype(np.uint8), 3).astype(np.float32) / 255.0
        return m

    def read_frame_and_mask(self, sim_w, sim_h, win_w, win_h):
        ok, frame = self.cap.read()
        if not ok:
            # graceful fallback
            zero_cam = np.zeros((win_h, win_w, 3), np.uint8)
            zero_mask = np.zeros((sim_h, sim_w), np.float32)
            return zero_cam, cv2.flip(zero_cam, 0), zero_mask, 0.0

        # camera preview sized for your GL texture
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_rgb = cv2.resize(cam_rgb, (win_w, win_h), interpolation=cv2.INTER_AREA)
        cam_rgb_flipped = cv2.flip(cam_rgb, 0)

        now = time.perf_counter()
        if (self._last_mask_small is None) or (now - self._last_mask_time >= self._mask_period):
            # build square letterboxed input at imgsz for best perf
            h, w = frame.shape[:2]
            scale = min(self.imgsz / w, self.imgsz / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            letter = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            y0 = (self.imgsz - nh) // 2; x0 = (self.imgsz - nw) // 2
            letter[y0:y0+nh, x0:x0+nw] = resized

            mask_sq = self._infer_mask_small(letter)
            # unletterbox back to resized, then to sim size
            mask_crop = mask_sq[y0:y0+nh, x0:x0+nw]
            mask_small = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_LINEAR)

            self._last_mask_small = mask_small
            self._last_mask_time = now

        # upsample to sim grid
        mask_sim = cv2.resize(self._last_mask_small, (sim_w, sim_h), interpolation=cv2.INTER_LINEAR)
        mask_area = float(mask_sim.mean())

        return frame, cam_rgb_flipped, mask_sim, mask_area

    def close(self):
        try: self.cap.release()
        except: pass
