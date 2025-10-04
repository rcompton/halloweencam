import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- Particle Simulation Constants ---
NUM_PARTICLES = 25000
GRAVITY = 0.01
REPULSION_STRENGTH = 20.8

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Initialize OpenCV to capture video from your webcam
cap = cv2.VideoCapture(0)
print("Starting combined ghost and particle effect. Press 'q' to quit.")

# Initialize Particles
success, frame = cap.read()
if not success:
    print("Could not read from camera. Exiting.")
    exit()

frame = cv2.flip(frame, 1)
frame_h, frame_w, _ = frame.shape

particles_pos = np.random.rand(NUM_PARTICLES, 2) * [frame_w, frame_h]
particles_vel = np.random.randn(NUM_PARTICLES, 2) * 0.5
particles_vel[:, 1] += 1

# --- FPS counter variables ---
start_time = time.time()
frame_count = 0
fps_display = "FPS: 0"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    results = model(frame, classes=0, verbose=False)
    
    output_image = np.zeros(frame.shape, dtype=np.uint8)

    final_mask = None
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        combined_mask_low_res = np.any(masks, axis=0)
        final_mask = cv2.resize(combined_mask_low_res.astype(np.uint8), (frame_w, frame_h)).astype(bool)

        # --- 1. DRAW THE GHOST FIRST ---
        # This creates the blurred, green background silhouette
        ghost_color = (0, 150, 0) # Dim green for the ghost
        output_image[final_mask] = ghost_color
        output_image = cv2.GaussianBlur(output_image, (55, 55), 0)

    # --- Run Particle Simulation Step ---
    particles_vel[:, 1] += GRAVITY
    
    if final_mask is not None:
        int_coords = particles_pos.astype(int)
        int_coords[:, 0] = np.clip(int_coords[:, 0], 0, frame_w - 1)
        int_coords[:, 1] = np.clip(int_coords[:, 1], 0, frame_h - 1)
        is_inside = final_mask[int_coords[:, 1], int_coords[:, 0]]
        particles_vel[is_inside, 1] -= REPULSION_STRENGTH

    particles_pos += particles_vel
    
    particles_pos[:, 0] %= frame_w
    particles_pos[:, 1] %= frame_h

    # --- 2. DRAW THE PARTICLES ON TOP ---
    # The particles are drawn over the existing ghost image
    draw_coords = particles_pos.astype(int)
    draw_coords[:, 0] = np.clip(draw_coords[:, 0], 0, frame_w - 1)
    draw_coords[:, 1] = np.clip(draw_coords[:, 1], 0, frame_h - 1)
    
    particle_color = (100, 255, 100) # Bright green for particles
    output_image[draw_coords[:, 1], draw_coords[:, 0]] = particle_color
    
    # --- FPS Calculation and Display ---
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:
        fps = frame_count / elapsed_time
        fps_display = f"FPS: {fps:.2f}"
        frame_count = 0
        start_time = time.time()
    
    cv2.putText(output_image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # --- Final Display ---
    cv2.imshow('Combined Ghost Effect', output_image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()