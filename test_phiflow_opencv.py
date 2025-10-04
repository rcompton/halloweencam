import cv2
import numpy as np
import phi.torch.flow as phi

# --- 1. Simulation Setup ---
print("Initializing advection-only simulation...")
WIDTH = 1280
HEIGHT = 800
RESOLUTION = dict(x=250, y=250)
DT = 1.0

DOMAIN_BOUNDS = phi.Box(x=(0, WIDTH), y=(0, HEIGHT))

# Walls are no longer needed for the solver, but we can keep them for future use
WALLS = [
    phi.Box(x=(-1, 1), y=None),
    phi.Box(x=(WIDTH - 1, WIDTH + 1), y=None),
    phi.Box(x=None, y=(-1, 1)),
    phi.Box(x=None, y=(HEIGHT - 1, HEIGHT + 1))
]

velocity = phi.StaggeredGrid(0, extrapolation=phi.extrapolation.ZERO, bounds=DOMAIN_BOUNDS, **RESOLUTION)
dye = phi.CenteredGrid(0, extrapolation=phi.extrapolation.BOUNDARY, bounds=DOMAIN_BOUNDS, **RESOLUTION)

# --- 2. OpenCV Window and Mouse Callback Setup ---
WINDOW_NAME = 'Interactive Advection'
cv2.namedWindow(WINDOW_NAME)

mouse_state = {'pos': (0, 0), 'prev_pos': (0, 0), 'left_down': False}

def mouse_callback(event, x, y, flags, param):
    global mouse_state
    mouse_state['pos'] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_state['left_down'] = True
        mouse_state['prev_pos'] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_state['left_down'] = False

cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# --- 3. Main Simulation & Display Loop ---
print("Simulation running. Drag mouse to paint velocity. Press 'q' to quit.")
while True:
    # --- Interaction Step ---
    if mouse_state['left_down']:
        pos_vec = phi.math.vec(x=mouse_state['pos'][0], y=mouse_state['pos'][1])
        prev_pos_vec = phi.math.vec(x=mouse_state['prev_pos'][0], y=mouse_state['prev_pos'][1])
        drag_velocity = (pos_vec - prev_pos_vec)
        
        dye += phi.CenteredGrid(phi.Sphere(center=pos_vec, radius=15), extrapolation=phi.extrapolation.BOUNDARY, bounds=DOMAIN_BOUNDS, **RESOLUTION)
        
        velocity_push = phi.CenteredGrid(phi.Sphere(center=pos_vec, radius=30), extrapolation=phi.extrapolation.ZERO, bounds=DOMAIN_BOUNDS, **RESOLUTION) * drag_velocity * 4
        velocity += velocity_push @ velocity
        
        mouse_state['prev_pos'] = mouse_state['pos']

    # --- Simplified Physics Step ---
    dye = phi.advect.semi_lagrangian(dye, velocity, DT) * 0.99
    velocity = phi.advect.semi_lagrangian(velocity, velocity, DT)
    
    # Add damping to make the velocity fade over time
    velocity *= 0.95
    
    # --- The pressure solve is now removed to guarantee stability ---
    # velocity, _ = phi.fluid.make_incompressible(velocity, obstacles=WALLS, solve=phi.Solve('CG', 1e-5))

    # --- Bridge: PhiFlow to OpenCV ---
    dye_values = dye.values.numpy(order='y,x')
    dye_values = np.expand_dims(dye_values, axis=-1)
    display_image = (np.clip(dye_values, 0, 1) * 255).astype(np.uint8)
    display_image = cv2.applyColorMap(display_image, cv2.COLORMAP_INFERNO)
    
    display_image = cv2.resize(display_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # --- Display Step ---
    cv2.imshow(WINDOW_NAME, display_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cv2.destroyAllWindows()