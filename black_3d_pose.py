import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import trimesh

# --- Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FULLSCREEN = True
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 60
MODEL_FILE = "skeleton.glb"

# --- Smoothing Factor ---
# Lower value = smoother but more "laggy" movement. 0.15 is a good start.
LERP_FACTOR = 0.15
# --- Confidence Threshold ---
# Only draw the model if the shoulders are detected with this confidence.
VISIBILITY_THRESHOLD = 0.8

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
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 5, 1))  # Light further away
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)
    glTranslatef(0.0, -1.0, -5.0)  # Adjusted camera position


def draw_mesh(mesh):
    """Renders a trimesh object using efficient vertex arrays."""
    glEnableClientState(GL_VERTEX_ARRAY)
    if hasattr(mesh.visual, "vertex_normals"):
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, mesh.visual.vertex_normals)

    if hasattr(mesh.visual, "vertex_colors"):
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, mesh.visual.vertex_colors)
    else:
        glColor3f(0.8, 0.8, 0.8)  # Slightly brighter default gray

    glVertexPointer(3, GL_FLOAT, 0, mesh.vertices)
    glDrawElements(
        GL_TRIANGLES, len(mesh.faces.flatten()), GL_UNSIGNED_INT, mesh.faces.flatten()
    )

    glDisableClientState(GL_VERTEX_ARRAY)
    if hasattr(mesh.visual, "vertex_normals"):
        glDisableClientState(GL_NORMAL_ARRAY)
    if hasattr(mesh.visual, "vertex_colors"):
        glDisableClientState(GL_COLOR_ARRAY)


def main():
    try:
        mesh = trimesh.load(MODEL_FILE, force="mesh")
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        max_extent = np.max(mesh.extents)
        mesh.apply_scale(2.0 / max_extent)  # Made the model a bit bigger

    except Exception as e:
        print(f"ERROR: Could not load model file '{MODEL_FILE}'.")
        print(f"Trimesh error: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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
    setup_opengl()

    smooth_pos = np.array([0.0, 0.0, 0.0])
    smooth_angle = 0.0

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1
    ) as pose:
        running = True
        person_visible = False  # Track if a person is currently visible
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == KEYDOWN
                    and (event.key == K_q or event.key == K_ESCAPE)
                ):
                    running = False

            ret, frame = cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # --- THE MAIN FIX: CONFIDENCE CHECK ---
            landmarks_detected = False
            if results.pose_world_landmarks:
                landmarks = results.pose_world_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # Check if the key landmarks are confidently detected
                if (
                    left_shoulder.visibility > VISIBILITY_THRESHOLD
                    and right_shoulder.visibility > VISIBILITY_THRESHOLD
                ):
                    landmarks_detected = True

            if landmarks_detected:
                if not person_visible:  # First time seeing the person
                    # Instantly snap to the person's position to avoid sliding in from nowhere
                    smooth_pos = np.array(
                        [
                            (left_shoulder.x + right_shoulder.x) / 2,
                            -(
                                left_shoulder.y
                                + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                            )
                            / 2,
                            -(left_shoulder.z + right_shoulder.z) / 2,
                        ]
                    )
                    person_visible = True

                # Calculate target position and angle for smoothing
                target_pos = np.array(
                    [
                        (left_shoulder.x + right_shoulder.x) / 2,
                        -(
                            left_shoulder.y
                            + landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
                        )
                        / 2,
                        -(left_shoulder.z + right_shoulder.z) / 2,
                    ]
                )
                target_angle = -math.degrees(
                    math.atan2(
                        right_shoulder.y - left_shoulder.y,
                        right_shoulder.x - left_shoulder.x,
                    )
                )

                # Apply Smoothing
                smooth_pos += (target_pos - smooth_pos) * LERP_FACTOR
                smooth_angle += (target_angle - smooth_angle) * LERP_FACTOR

                glPushMatrix()
                glTranslatef(smooth_pos[0], smooth_pos[1], smooth_pos[2])
                glRotatef(smooth_angle, 0, 0, 1)  # Torso tilt
                glRotatef(-90, 1, 0, 0)  # Stand upright
                draw_mesh(mesh)
                glPopMatrix()
            else:
                person_visible = False  # Person is no longer visible

            pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
