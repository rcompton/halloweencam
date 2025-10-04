# Import the necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
# model_selection=0 is for the general-purpose landscape model
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Initialize OpenCV to capture video from your webcam
# 0 is usually the default webcam. If you have multiple cameras,
# you might need to try 1, 2, etc.
cap = cv2.VideoCapture(0)

print("Starting camera feed. Press 'q' to quit.")

# Main loop to process video frames
while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # --- PROCESSING ---

    # Flip the frame horizontally for a more intuitive selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the segmentation mask
    results = segmenter.process(rgb_frame)
    mask = results.segmentation_mask

    # --- CREATING THE GHOST EFFECT ---

    # Create a condition where the mask is greater than a threshold (e.g., 0.5)
    # This means "select all pixels that are part of the person"
    condition = mask > 0.5

    # Create a black background image
    output_image = np.zeros(frame.shape, dtype=np.uint8)

    # Where the condition is true, set the output image pixels to white
    # This creates the solid white ghost effect
    output_image[condition] = [255, 255, 255] # White color

    # --- DISPLAYING THE OUTPUT ---

    # Show the original camera feed
    cv2.imshow('Original Feed', frame)
    
    # Show the final ghost effect
    cv2.imshow('Ghost Effect', output_image)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()
segmenter.close()