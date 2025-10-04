import cv2
import numpy as np
from ultralytics import YOLO
import time # <-- 1. Import the time library

# Load the YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Initialize OpenCV to capture video from your webcam
cap = cv2.VideoCapture(0)
print("Starting YOLOv8 ghost feed. Press 'q' to quit.")

# --- 2. Initialize FPS counter variables ---
start_time = time.time()
frame_count = 0
fps_display = "FPS: 0"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    # --- YOLOv8 DETECTION AND MASKING ---

    # Run the model on the frame
    results = model(frame, classes=0, verbose=False)

    # Create a black background to draw our ghosts on
    output_image = np.zeros(frame.shape, dtype=np.uint8)

    # Check if any masks were found in the results
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        combined_mask_low_res = np.any(masks, axis=0)

        frame_h, frame_w, _ = frame.shape
        final_mask = cv2.resize(combined_mask_low_res.astype(np.uint8), (frame_w, frame_h)).astype(bool)
        
        # --- CREATING THE SPOOKY EFFECT ---
        ghost_color = (0, 150, 0)
        output_image[final_mask] = ghost_color
        output_image = cv2.GaussianBlur(output_image, (55, 55), 0)

    # --- 3. FPS Calculation and Display ---
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0: # Update the FPS reading every second
        fps = frame_count / elapsed_time
        fps_display = f"FPS: {fps:.2f}"
        # Reset the counter and timer
        frame_count = 0
        start_time = time.time()
    
    # Draw the FPS text on the output image
    cv2.putText(output_image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # --- DISPLAYING THE OUTPUT ---
    cv2.imshow('YOLOv8 Ghost Effect', output_image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()