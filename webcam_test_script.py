import cv2

# --- Configuration ---
# You can change these values
WINDOW_NAME = "Webcam Test"
CAMERA_INDEX = 0  # 0 is usually the default webcam. Change if you have multiple cameras.
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60

def main():
    """
    Captures video from the webcam and displays it in a window.
    Press 'q' to quit.
    """
    # Initialize the video capture object
    # We use CAP_V4L2 to ensure we're using the Video4Linux2 backend, which is standard for Linux.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    # --- Set Camera Properties ---
    # It's good practice to explicitly set the resolution and frame rate.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # --- Print Actual Camera Settings ---
    # Verify what the camera is actually outputting.
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Attempting to capture at {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps")
    print(f"Camera started at: {actual_width}x{actual_height} @ {actual_fps:.2f}fps")


    # Create a window that can be resized
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # If the frame was not successfully read, ret will be False
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame in the window
        cv2.imshow(WINDOW_NAME, frame)

        # Wait for 1ms and check if the 'q' key was pressed
        # The 0xFF is a bitmask for 64-bit systems
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, closing window.")
            break

    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
