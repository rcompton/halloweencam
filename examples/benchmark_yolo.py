import time
from ultralytics import YOLO
import torch

# --- Configuration ---
MODEL_PATH = "yolov8n-seg.pt"
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
NUM_WARMUP_RUNS = 10
NUM_BENCHMARK_RUNS = 100

def run_benchmark():
    """Loads a TensorRT model and benchmarks its inference speed."""

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

    # 3. Create a dummy input image on the GPU
    dummy_input = torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), device="cuda")
    print(f"\nCreated a dummy input tensor of size {dummy_input.shape}.")

    # 4. Perform warm-up runs
    print(f"Performing {NUM_WARMUP_RUNS} warm-up runs...")
    for _ in range(NUM_WARMUP_RUNS):
        model(dummy_input, verbose=False)
    print("Warm-up complete.")

    # 5. Run the actual benchmark
    print(f"\nRunning benchmark for {NUM_BENCHMARK_RUNS} iterations...")
    start_time = time.perf_counter()
    for _ in range(NUM_BENCHMARK_RUNS):
        model(dummy_input, verbose=False)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / NUM_BENCHMARK_RUNS) * 1000
    fps = NUM_BENCHMARK_RUNS / total_time

    print("\n--- BENCHMARK RESULTS ---")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Frames Per Second (FPS): {fps:.2f}")

if __name__ == "__main__":
    run_benchmark()

