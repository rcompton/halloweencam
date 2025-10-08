import time
from ultralytics import YOLO
import torch

# --- Configuration ---
MODEL_PATH = "yolov8n-seg.pt"
RESOLUTIONS = [
    (640, 360),
    (854, 480),
    (1280, 720),
    (1920, 1080),
]
NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 100


def run_benchmark():
    """Loads a TensorRT model and benchmarks its inference speed for multiple resolutions."""

    # 1. Check for CUDA device
    if not torch.cuda.is_available():
        print("CUDA is not available. TensorRT requires a GPU.")
        return

    print(f"Using device: {torch.cuda.get_device_name(0)}")

    # 2. Load the TensorRT model
    print(f"\nLoading model from {MODEL_PATH}...")
    start_time = time.perf_counter()
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    end_time = time.perf_counter()
    print(f"Model loading and setup time: {end_time - start_time:.4f} seconds.")

    print("\n--- BENCHMARK RESULTS ---")
    print(f"{'Resolution':<15} {'Avg. Time (ms)':<20} {'FPS':<10}")
    print("-" * 45)

    for width, height in RESOLUTIONS:
        # 3. Create a dummy input image on the GPU
        dummy_input = torch.rand((1, 3, height, width), device="cuda")

        # 4. Perform warm-up runs
        for _ in range(NUM_WARMUP_RUNS):
            model(dummy_input, verbose=False)

        # 5. Run the actual benchmark
        start_time = time.perf_counter()
        for _ in range(NUM_BENCHMARK_RUNS):
            model(dummy_input, verbose=False)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_inference_time_ms = (total_time / NUM_BENCHMARK_RUNS) * 1000
        fps = NUM_BENCHMARK_RUNS / total_time

        print(f"{f'{width}x{height}':<15} {avg_inference_time_ms:<20.2f} {fps:<10.2f}")


if __name__ == "__main__":
    run_benchmark()
