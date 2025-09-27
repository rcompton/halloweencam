import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

# --- Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FULLSCREEN = True # Set to True for the final display, False for testing
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
# Define the connections for the skeleton
SKELETON_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
]

def setup_projection():
    """Sets up the OpenGL perspective."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    display_width = pygame.display.Info().current_w
    display_height = pygame.display.Info().current_h
    gluPerspective(45, (display_width / display_height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glTranslatef(0.0, -0.5, -3.0) 

def draw_bone(p1, p2, width=0.03):
    """Draws a 3D cuboid representing a bone between two points."""
    # Vector from p1 to p2
    vector = p2 - p1
    length = np.linalg.norm(vector)
    if length < 1e-6: return  # Avoid division by zero for zero-length bones

    # The default orientation for our cuboid is along the Z-axis
    default_axis = np.array([0, 0, 1])
    
    # Normalize the bone vector
    vector_norm = vector / length

    # Calculate the rotation axis (cross product)
    axis = np.cross(default_axis, vector_norm)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        # Vectors are parallel, no rotation needed or 180 degree rotation
        axis = np.array([1, 0, 0]) # Arbitrary axis
        angle = 0 if np.dot(default_axis, vector_norm) > 0 else 180
    else:
        axis = axis / axis_len
        # Calculate the rotation angle (dot product)
        angle = math.degrees(math.acos(np.dot(default_axis, vector_norm)))

    glPushMatrix()
    # 1. Translate to the midpoint of the bone
    glTranslatef(p1[0] + vector[0] / 2, p1[1] + vector[1] / 2, p1[2] + vector[2] / 2)
    # 2. Rotate to align with the bone's direction
    glRotatef(angle, *axis)
    # 3. Scale to the bone's length and width
    glScalef(width, width, length)
    
    # Draw a simple cube primitive
    glBegin(GL_QUADS)
    # Front Face
    glVertex3f(-0.5, -0.5, 0.5); glVertex3f(0.5, -0.5, 0.5); glVertex3f(0.5, 0.5, 0.5); glVertex3f(-0.5, 0.5, 0.5)
    # Back Face
    glVertex3f(-0.5, -0.5, -0.5); glVertex3f(-0.5, 0.5, -0.5); glVertex3f(0.5, 0.5, -0.5); glVertex3f(0.5, -0.5, -0.5)
    # Top Face
    glVertex3f(-0.5, 0.5, -0.5); glVertex3f(-0.5, 0.5, 0.5); glVertex3f(0.5, 0.5, 0.5); glVertex3f(0.5, 0.5, -0.5)
    # Bottom Face
    glVertex3f(-0.5, -0.5, -0.5); glVertex3f(0.5, -0.5, -0.5); glVertex3f(0.5, -0.5, 0.5); glVertex3f(-0.5, -0.5, 0.5)
    # Right face
    glVertex3f(0.5, -0.5, -0.5); glVertex3f(0.5, 0.5, -0.5); glVertex3f(0.5, 0.5, 0.5); glVertex3f(0.5, -0.5, 0.5)
    # Left Face
    glVertex3f(-0.5, -0.5, -0.5); glVertex3f(-0.5, -0.5, 0.5); glVertex3f(-0.5, 0.5, 0.5); glVertex3f(-0.5, 0.5, -0.5)
    glEnd()

    glPopMatrix()


def draw_skeleton(landmarks):
    """Draws solid bones between landmark points."""
    glColor3f(1.0, 1.0, 1.0)  # White skeleton
    for connection in SKELETON_CONNECTIONS:
        start_landmark = landmarks[connection[0].value]
        end_landmark = landmarks[connection[1].value]

        if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
            p1 = np.array([start_landmark.x, -start_landmark.y, -start_landmark.z])
            p2 = np.array([end_landmark.x, -end_landmark.y, -end_landmark.z])
            draw_bone(p1, p2)

def main():
    """Main loop for capturing video, processing pose, and rendering in 3D."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)

    pygame.init()
    display_flags = DOUBLEBUF | OPENGL
    if FULLSCREEN:
        display_flags |= FULLSCREEN
        pygame.display.set_mode((0, 0), display_flags)
        pygame.mouse.set_visible(False)
    else:
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), display_flags)
        
    pygame.display.set_caption("3D Spooky Pose")
    setup_projection()
    glEnable(GL_DEPTH_TEST) # Enable depth testing for proper 3D rendering

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True
    ) as pose:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and (event.key == K_q or event.key == K_ESCAPE)):
                    running = False

            ret, frame = cap.read()
            if not ret: continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            if results.pose_world_landmarks:
                draw_skeleton(results.pose_world_landmarks.landmark)

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()

