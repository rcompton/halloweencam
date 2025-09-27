import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh

# --- Configuration ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
FULLSCREEN = False # Start in a window for easier testing
MODEL_FILE = 'skeleton.glb'

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
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 1, 5, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)
    glTranslatef(0.0, -1.0, -5.0) # Start camera position

def draw_mesh(mesh):
    """Renders a trimesh object using efficient vertex arrays."""
    glEnableClientState(GL_VERTEX_ARRAY)
    if hasattr(mesh.visual, 'vertex_normals'):
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, mesh.visual.vertex_normals)
    
    if hasattr(mesh.visual, 'vertex_colors'):
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_UNSIGNED_BYTE, 0, mesh.visual.vertex_colors)
    else:
        glColor3f(0.8, 0.8, 0.8)
    
    glVertexPointer(3, GL_FLOAT, 0, mesh.vertices)
    glDrawElements(GL_TRIANGLES, len(mesh.faces.flatten()), GL_UNSIGNED_INT, mesh.faces.flatten())
    
    glDisableClientState(GL_VERTEX_ARRAY)
    if hasattr(mesh.visual, 'vertex_normals'):
        glDisableClientState(GL_NORMAL_ARRAY)
    if hasattr(mesh.visual, 'vertex_colors'):
        glDisableClientState(GL_COLOR_ARRAY)

def main():
    try:
        mesh = trimesh.load(MODEL_FILE, force='mesh')
        center = mesh.bounds.mean(axis=0)
        mesh.apply_translation(-center)
        max_extent = np.max(mesh.extents)
        mesh.apply_scale(2.0 / max_extent)
    except Exception as e:
        print(f"ERROR: Could not load model file '{MODEL_FILE}'.")
        print(f"Trimesh error: {e}")
        return

    pygame.init()
    display_flags = DOUBLEBUF | OPENGL
    if FULLSCREEN:
        display_flags |= FULLSCREEN
        pygame.display.set_mode((0, 0), display_flags)
        pygame.mouse.set_visible(False)
    else:
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), display_flags)
    setup_opengl()

    # --- Interactive Controls ---
    rot_x, rot_y = 0, 0
    zoom = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and (event.key == K_ESCAPE)):
                running = False
        
        # --- Keyboard Controls ---
        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:  rot_y -= 2
        if keys[K_RIGHT]: rot_y += 2
        if keys[K_UP]:    rot_x -= 2
        if keys[K_DOWN]:  rot_x += 2
        if keys[K_w]:     zoom += 0.1
        if keys[K_s]:     zoom -= 0.1

        # --- Rendering ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        # Apply interactive transformations
        glTranslatef(0, 0, zoom)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        # Stand the model up
        glRotatef(-90, 1, 0, 0) 
        
        draw_mesh(mesh)
        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10) # Small delay to not run at max CPU speed

    pygame.quit()

if __name__ == "__main__":
    main()
