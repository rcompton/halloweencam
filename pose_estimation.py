import cv2
import mediapipe as mp
import time

# --- Configuration ---
WINDOW_NAME = "Pose Estimation Test"
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    """
    Captures video, performs pose estimation, and displays the skeleton.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        return

    # --- Set Camera Properties ---
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # --- Print Actual Camera Settings ---
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera started at: {actual_width}x{actual_height} @ {actual_fps:.2f}fps")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # To calculate FPS
    prev_frame_time = 0
    new_frame_time = 0

    # Initialize MediaPipe Pose
    # min_detection_confidence is the threshold for detecting a person
    # min_tracking_confidence is the threshold for tracking the pose landmarks
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            # --- POSE ESTIMATION ---
            # 1. Convert the BGR image to RGB for MediaPipe.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Process the image and find the pose.
            results = pose.process(image_rgb)

            # 3. Draw the pose annotation on the original frame.
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

            # --- FPS CALCULATION ---
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # --- Display the frame ---
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, closing window.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    