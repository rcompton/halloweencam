from ursina import *

# --- Configuration ---
MODEL_FILE = 'Duck.glb' # Use the standard test model first

# --- Ursina Application Setup ---
app = Ursina()

# --- Load the Model ---
# Ursina makes loading and displaying a model a single, simple line.
# It automatically handles centering and scaling.
model_entity = Entity(model=MODEL_FILE, scale=1, rotation_y=-90)

# --- Camera Controls ---
# Use Ursina's built-in, easy-to-use camera controls.
# Drag with the right mouse button to orbit. Scroll to zoom.
editor_camera = EditorCamera()

# --- Simple Lighting ---
# Add a light to make the model visible.
pivot = Entity()
DirectionalLight(parent=pivot, y=2, z=3, shadows=True)

# Start the application
app.run()
