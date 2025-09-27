import cv2

# --- Configuration ---
WINDOW_NAME = "Webcam Test"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60

def main():
    """
    Captures video from the webcam and displays it in a window.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    # --- SET THE VIDEO FORMAT (THE FIX) ---
    # We must set the Four Character Code (FOURCC) to MJPEG to unlock high resolutions and frame rates.
    # This needs to be done BEFORE setting width, height, and FPS.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # --- Set Camera Properties ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # --- Print Actual Camera Settings ---
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"Requesting: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FRAME_RATE}fps in MJPG format")
    print(f"Actual Capture: {actual_width}x{actual_height} @ {actual_fps:.2f}fps in {fourcc_str} format")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' pressed, closing window.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()