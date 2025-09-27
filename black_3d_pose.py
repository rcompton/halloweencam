import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import trimesh # Use trimesh for modern 3D model loading

# --- Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FULLSCREEN = True
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60
MODEL_FILE = 'skeleton.glb' # CHANGED: Now expects a .glb or .gltf file

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose

def setup_opengl():
    """Sets up OpenGL perspective and lighting."""
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    display_width = pygame.display.Info().current_w
    display_height = pygame.display.Info().current_h
    gluPerspective(45, (display_width / display_height), 0.1, 100.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Set up basic lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 2, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)
    glTranslatef(0.0, -1.2, -5.0) # Move camera back and down

def draw_mesh(mesh):
    """Renders a trimesh object using efficient vertex arrays."""
    glEnableClientState(GL_VERTEX_ARRAY)
    # Check if the model has vertex normals for lighting
    if hasattr(mesh.visual, 'vertex_normals'):
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, mesh.visual.vertex_normals)
    
    glVertexPointer(3, GL_FLOAT, 0, mesh.vertices)
    glDrawElements(GL_TRIANGLES, len(mesh.faces.flatten()), GL_UNSIGNED_INT, mesh.faces.flatten())
    
    glDisableClientState(GL_VERTEX_ARRAY)
    if hasattr(mesh.visual, 'vertex_normals'):
        glDisableClientState(GL_NORMAL_ARRAY)


def main():
    # --- Load 3D Model using Trimesh ---
    try:
        # force='mesh' combines all geometries into a single mesh
        mesh = trimesh.load(MODEL_FILE, force='mesh')
    except Exception as e:
        print(f"ERROR: Could not load model file '{MODEL_FILE}'. Make sure it's in the same directory.")
        print(f"Trimesh error: {e}")
        return

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
    display_flags = DOUBLEBUF | OPENGL
    if FULLSCREEN:
        display_flags |= FULLSCREEN
        pygame.display.set_mode((0, 0), display_flags)
        pygame.mouse.set_visible(False)
    else:
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), display_flags)
    setup_opengl()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
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
                landmarks = results.pose_world_landmarks.landmark
                
                # --- Calculate Torso Orientation ---
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Center of torso
                center_x = (left_shoulder.x + right_shoulder.x) / 2
                center_y = (left_shoulder.y + left_hip.y) / 2
                center_z = (left_shoulder.z + right_shoulder.z) / 2

                # Rotation (angle of the shoulders)
                angle = -math.degrees(math.atan2(right_shoulder.y - left_shoulder.y,
                                                 right_shoulder.x - left_shoulder.x))

                glPushMatrix()
                # --- Apply Transformations ---
                # 1. Translate the model to the person's torso center
                glTranslatef(center_x, -center_y, -center_z)
                # 2. Rotate the model to match the shoulder tilt
                glRotatef(angle, 0, 0, 1)

                # --- Render the Model ---
                draw_mesh(mesh)
                glPopMatrix()

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()

