# Import the necessary libraries
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

# Initialize OpenCV to capture video from your webcam
cap = cv2.VideoCapture(0)

print("Starting spooky camera feed. Press 'q' to quit.")

# --- (Optional) Set up for full-screen projection ---
cv2.namedWindow('Ghost Effect', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Ghost Effect', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Main loop to process video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip the frame horizontally for a more intuitive selfie-view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the segmentation mask
    results = segmenter.process(rgb_frame)
    mask = results.segmentation_mask

    # --- CREATING THE SPOOKY EFFECT ---

    # 1. Create a condition from the mask
    condition = mask > 0.5

    # 2. Create a black background
    output_image = np.zeros(frame.shape, dtype=np.uint8)
    
    # 3. Define a spooky color (a dim, ghostly green)
    # Using lower values (e.g., 150 instead of 255) makes it look transparent.
    ghost_color = (0, 150, 0) # BGR format for OpenCV

    # 4. Apply the color to the silhouette
    output_image[condition] = ghost_color

    # 5. Add a blur to make the edges "wispy"
    # The (55, 55) tuple is the kernel size. Larger numbers = more blur.
    output_image = cv2.GaussianBlur(output_image, (55, 55), 0)


    # --- DISPLAYING THE OUTPUT ---

    # We don't need to see the original feed anymore, just the effect
    # cv2.imshow('Original Feed', frame)
    
    cv2.imshow('Ghost Effect', output_image)

    # Check for the 'q' key to quit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- CLEANUP ---
print("Closing application.")
cap.release()
cv2.destroyAllWindows()
segmenter.close()