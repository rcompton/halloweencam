import cv2
import numpy as np
import phi.flow as phi
import time  # <-- Import the time library

# --- 1. Simulation Setup ---
print("Initializing fluid simulation...")
WIDTH = 1280
HEIGHT = 800
RESOLUTION = dict(x=128, y=80)
DT = 1.0

DOMAIN_BOUNDS = phi.Box(x=(0, WIDTH), y=(0, HEIGHT))

velocity = phi.StaggeredGrid(0, extrapolation=phi.extrapolation.ZERO, bounds=DOMAIN_BOUNDS, **RESOLUTION)
dye = phi.CenteredGrid(0, extrapolation=phi.extrapolation.BOUNDARY, bounds=DOMAIN_BOUNDS, **RESOLUTION)

# --- 2. OpenCV Window and Mouse Callback Setup ---
WINDOW_NAME = 'Interactive Fluid'
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

# --- Initialize FPS counter variables ---
start_time = time.time()
frame_count = 0
fps_display = "FPS: 0"

# --- 3. Main Simulation & Display Loop ---
print("Simulation running. Drag mouse to add smoke. Press 'q' to quit.")
while True:
    # --- Interaction Step ---
    if mouse_state['left_down']:
        pos_vec = phi.math.vec(x=mouse_state['pos'][0], y=mouse_state['pos'][1])
        prev_pos_vec = phi.math.vec(x=mouse_state['prev_pos'][0], y=mouse_state['prev_pos'][1])
        drag_velocity = (pos_vec - prev_pos_vec)
        
        dye += phi.CenteredGrid(phi.Sphere(center=pos_vec, radius=15), extrapolation=phi.extrapolation.BOUNDARY, bounds=DOMAIN_BOUNDS, **RESOLUTION) * 2.0
        
        velocity_push = phi.CenteredGrid(phi.Sphere(center=pos_vec, radius=30), extrapolation=phi.extrapolation.ZERO, bounds=DOMAIN_BOUNDS, **RESOLUTION) * drag_velocity * 10
        velocity += velocity_push @ velocity
        
        mouse_state['prev_pos'] = mouse_state['pos']

    # --- Physics Step ---
    buoyancy_force = (dye * phi.vec(x=0, y=0.2)) @ velocity
    dye = phi.advect.semi_lagrangian(dye, velocity, DT) * 0.995
    velocity = phi.advect.semi_lagrangian(velocity, velocity, DT) + buoyancy_force
    velocity *= 0.98

    # --- Bridge: PhiFlow to OpenCV ---
    dye_values = dye.values.numpy(order='y,x')
    dye_values = np.expand_dims(dye_values, axis=-1)
    display_image = (np.clip(dye_values, 0, 1) * 255).astype(np.uint8)
    display_image = cv2.applyColorMap(display_image, cv2.COLORMAP_INFERNO)
    
    display_image = cv2.resize(display_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    # --- FPS Calculation and Display ---
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1.0:  # Update the FPS reading every second
        fps = frame_count / elapsed_time
        fps_display = f"FPS: {fps:.2f}"
        # Reset counter and timer
        frame_count = 0
        start_time = time.time()

    # Draw the FPS text on the output image (in white color)
    cv2.putText(display_image, fps_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # --- Display Step ---
    cv2.imshow(WINDOW_NAME, display_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cv2.destroyAllWindows()