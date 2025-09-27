import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

# --- Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
# Define the connections for the skeleton
SKELETON_CONNECTIONS = [
    # Torso
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    # Left Arm
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    # Right Arm
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    # Left Leg
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    # Right Leg
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

def setup_projection():
    """Sets up the OpenGL perspective."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_WIDTH / WINDOW_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    # Move the camera back a bit so we can see the model
    glTranslatef(0.0, -0.5, -3.0) 

def draw_skeleton(landmarks):
    """Draws lines between the landmark points to form a 3D skeleton."""
    glLineWidth(5.0)
    glBegin(GL_LINES)
    glColor3f(1.0, 1.0, 1.0)  # White skeleton

    for connection in SKELETON_CONNECTIONS:
        start_landmark = landmarks[connection[0].value]
        end_landmark = landmarks[connection[1].value]

        # Check if landmarks are visible enough to be drawn
        if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
            # MediaPipe's Y is downward, OpenGL's Y is upward, so we negate Y.
            # MediaPipe's Z is towards the camera, OpenGL's is away, so we negate Z.
            glVertex3f(start_landmark.x, -start_landmark.y, -start_landmark.z)
            glVertex3f(end_landmark.x, -end_landmark.y, -end_landmark.z)
    
    glEnd()

def main():
    """Main loop for capturing video, processing pose, and rendering in 3D."""
    # --- OpenCV Setup ---
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    # --- Pygame & OpenGL Setup ---
    pygame.init()
    # --- THE FIX ---
    # Set OpenGL version attributes BEFORE creating the display.
    # X11 forwarding often only supports older, more basic OpenGL contexts.
    # By explicitly requesting a 2.1 compatibility profile, we increase our chances of success.
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)

    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Spooky Pose")
    setup_projection()

    # --- MediaPipe Pose Initialization ---
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1, # Use a more complex model for better 3D landmarks
        enable_segmentation=False,
        smooth_landmarks=True
    ) as pose:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_q):
                    running = False

            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame.")
                continue

            # Process with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            # --- OpenGL Drawing ---
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if results.pose_world_landmarks:
                # Use pose_world_landmarks for 3D coordinates in meters
                draw_skeleton(results.pose_world_landmarks.landmark)

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()