import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

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
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 5, 5, 1))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_DEPTH_TEST)
    glTranslatef(0.0, -1.0, -5.0)

def draw_scene_graph(scene, node_name, transforms):
    """Recursively draws the scene graph, applying transformations."""
    # Get the transformation matrix for the current node
    transform = transforms[node_name]
    
    glPushMatrix()
    # Apply the node's transformation
    glMultMatrixf(transform.T) # OpenGL expects column-major, numpy is row-major

    # Draw the geometry associated with this node, if any
    if node_name in scene.geometry:
        mesh = scene.geometry[node_name]
        glEnableClientState(GL_VERTEX_ARRAY)
        if hasattr(mesh.visual, 'vertex_normals'):
            glEnableClientState(GL_NORMAL_ARRAY)
            glNormalPointer(GL_FLOAT, 0, mesh.visual.vertex_normals)
        
        glVertexPointer(3, GL_FLOAT, 0, mesh.vertices)
        glDrawElements(GL_TRIANGLES, len(mesh.faces.flatten()), GL_UNSIGNED_INT, mesh.faces.flatten())
        
        glDisableClientState(GL_VERTEX_ARRAY)
        if hasattr(mesh.visual, 'vertex_normals'):
            glDisableClientState(GL_NORMAL_ARRAY)
    
    # Recursively call for all children of this node
    children = scene.graph.get(node_name)[1]
    if children:
        for child_name in children:
            draw_scene_graph(scene, child_name, transforms)
            
    glPopMatrix()


def main():
    try:
        # Load the full scene, preserving the graph structure
        scene = trimesh.load(MODEL_FILE)
    except Exception as e:
        print(f"ERROR: Could not load model file '{MODEL_FILE}'.")
        print(f"Trimesh error: {e}")
        return

    # Store the original and current transforms for each node
    original_transforms = scene.graph.transforms.copy()
    current_transforms = scene.graph.transforms.copy()
    
    # Get the list of nodes that have geometry (the visible parts)
    nodes_with_geometry = list(scene.geometry.keys())
    selected_node_index = 0
    
    print("--- Movable Body Parts ---")
    for i, name in enumerate(nodes_with_geometry):
        print(f"{i}: {name}")
    print("--------------------------")
    
    pygame.init()
    # ... (pygame setup code remains the same) ...
    display_flags = DOUBLEBUF | OPENGL
    if FULLSCREEN:
        display_flags |= FULLSCREEN
        pygame.display.set_mode((0, 0), display_flags)
        pygame.mouse.set_visible(False)
    else:
        pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), display_flags)
    setup_opengl()

    rot_x, rot_y = 0, 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False
            if event.type == KEYDOWN:
                # Cycle through body parts with PAGE UP and PAGE DOWN
                if event.key == K_PAGEUP:
                    selected_node_index = (selected_node_index + 1) % len(nodes_with_geometry)
                    print(f"Selected: {nodes_with_geometry[selected_node_index]}")
                if event.key == K_PAGEDOWN:
                    selected_node_index = (selected_node_index - 1) % len(nodes_with_geometry)
                    print(f"Selected: {nodes_with_geometry[selected_node_index]}")

        keys = pygame.key.get_pressed()
        
        # --- Control the whole model ---
        if keys[K_w]: rot_x -= 2
        if keys[K_s]: rot_x += 2
        
        # --- Control the selected body part ---
        selected_node_name = nodes_with_geometry[selected_node_index]
        # Create rotation matrices from arrow key input
        if keys[K_LEFT]:
            rotation = R.from_euler('y', -5, degrees=True).as_matrix()
            current_transforms[selected_node_name][:3,:3] = current_transforms[selected_node_name][:3,:3] @ rotation
        if keys[K_RIGHT]:
            rotation = R.from_euler('y', 5, degrees=True).as_matrix()
            current_transforms[selected_node_name][:3,:3] = current_transforms[selected_node_name][:3,:3] @ rotation

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        
        # Apply global rotation for the camera view
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        
        # Start drawing from the root of the scene graph
        root_node = scene.graph.get_roots()[0]
        draw_scene_graph(scene, root_node, current_transforms)
        
        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()

if __name__ == "__main__":
    main()
