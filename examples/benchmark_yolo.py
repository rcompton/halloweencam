import time
import os
import cv2
from ultralytics import YOLO
import torch
import numpy as np

# --- Configuration ---
MODEL_PATH = "yolov8n-seg.pt"
IMAGE_DIR = "examples/images"
RESOLUTIONS = [
    (640, 352),
    (864, 480),
    (1280, 736),
    (1920, 1088),
]
NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 100


def preprocess_image(image_path, size, device):
    """Loads an image, resizes it, and converts it to a torch tensor."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    if device == "cuda":
        img = img.half()  # half precision only on GPU
    img = img.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW, add batch dimension
    img = img / 255.0  # normalize
    return img


def run_benchmark():
    """Loads a model and benchmarks its inference speed for multiple resolutions using real images."""

    # 1. Check for CUDA device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Running on CPU. This will be slow.")

    # 2. Get image paths
    image_paths = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Please add some images to benchmark.")
        return
    print(f"Found {len(image_paths)} images for benchmarking.")

    # 3. Load the model
    print(f"\nLoading model from {MODEL_PATH}...")
    start_time = time.perf_counter()
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    end_time = time.perf_counter()
    print(f"Model loading and setup time: {end_time - start_time:.4f} seconds.")

    print("\n--- BENCHMARK RESULTS (using real images) ---")
    print(f"{'Resolution':<15} {'Avg. Time (ms)':<20} {'FPS':<10}")
    print("-" * 55)

    for width, height in RESOLUTIONS:
        # 4. Preprocess images for the current resolution
        try:
            input_tensors = [preprocess_image(p, (width, height), device) for p in image_paths]
            input_tensors = [t for t in input_tensors if t is not None]
            if not input_tensors:
                print(f"All images failed to load for resolution {width}x{height}. Skipping.")
                continue
        except Exception as e:
            print(f"Error during preprocessing at {width}x{height}: {e}")
            continue

        # 5. Perform warm-up runs
        for _ in range(NUM_WARMUP_RUNS):
            for tensor in input_tensors:
                model(tensor, verbose=False)

        # 6. Run the actual benchmark
        start_time = time.perf_counter()
        for _ in range(NUM_BENCHMARK_RUNS):
            for tensor in input_tensors:
                model(tensor, verbose=False)
        end_time = time.perf_counter()

        num_inferences = NUM_BENCHMARK_RUNS * len(input_tensors)
        total_time = end_time - start_time
        avg_inference_time_ms = (total_time / num_inferences) * 1000
        fps = num_inferences / total_time

        print(f"{f'{width}x{height}':<15} {avg_inference_time_ms:<20.2f} {fps:<10.2f}")


if __name__ == "__main__":
    run_benchmark()
