import glfw
import moderngl
import numpy as np
import time
import cv2
import mediapipe as mp

# --- 1. Initialize Everything ---

# MediaPipe and OpenCV Setup
print("Initializing camera and MediaPipe...")
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Window and ModernGL Setup
print("Initializing ModernGL...")
if not glfw.init():
    raise Exception("GLFW can't be initialized")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

WIDTH, HEIGHT = 1280, 720
window = glfw.create_window(WIDTH, HEIGHT, "Ghost Fluid Effect", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.make_context_current(window)
ctx = moderngl.create_context()

# --- 2. Shader Program ---
program = ctx.program(
    vertex_shader="""
        #version 330
        in vec2 in_vert;
        out vec2 uv;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            uv = (in_vert + 1.0) / 2.0;
        }
    """,
    fragment_shader="""
        #version 330
        in vec2 uv;
        out vec4 fragColor;

        uniform float u_time;
        uniform sampler2D u_mask_texture;

        // 2D Noise function
        float noise(vec2 st) {
            vec2 i = floor(st);
            vec2 f = fract(st);
            float a = fract(sin(dot(i, vec2(12.9898,78.233))) * 43758.5453123);
            float b = fract(sin(dot(i + vec2(1.0, 0.0), vec2(12.9898,78.233))) * 43758.5453123);
            float c = fract(sin(dot(i + vec2(0.0, 1.0), vec2(12.9898,78.233))) * 43758.5453123);
            float d = fract(sin(dot(i + vec2(1.0, 1.0), vec2(12.9898,78.233))) * 43758.5453123);
            vec2 u = f * f * (3.0 - 2.0 * f);
            return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
        }

        void main() {
            // --- SIMULATION #1: Background Fluid ---
            float bg_zoom = 3.0;
            float bg_speed = 0.2;
            vec2 bg_uv1 = uv * bg_zoom + vec2(u_time * bg_speed, u_time * bg_speed * 0.5);
            vec2 bg_uv2 = uv * (bg_zoom * 1.5) - vec2(u_time * bg_speed * 0.3, u_time * bg_speed * 0.8);
            float bg_distort = noise(bg_uv2) * 1.0;
            float bg_val = noise(bg_uv1 + bg_distort);
            vec3 background_fluid_color = mix(vec3(0.0, 0.0, 0.0), vec3(0.3, 0.0, 0.4), smoothstep(0.0, 0.6, bg_val));
            background_fluid_color = mix(background_fluid_color, vec3(1.0, 0.5, 0.0), smoothstep(0.6, 1.0, bg_val));

            // --- SIMULATION #2: Person (Ghost) Fluid ---
            float person_zoom = 5.0;
            float person_speed = 0.1;
            vec2 p_uv1 = uv * person_zoom - vec2(u_time * person_speed * 0.8, u_time * person_speed);
            vec2 p_uv2 = uv * (person_zoom * 1.5) + vec2(u_time * person_speed * 0.5, u_time * person_speed * 0.3);
            float p_distort = noise(p_uv2) * 2.0;
            float p_val = noise(p_uv1 + p_distort);
            vec3 person_fluid_color = mix(vec3(0.1, 0.5, 0.1), vec3(0.7, 1.0, 0.5), p_val); // Dark green to bright green

            // --- Blending ---
            // Get the mask value (0.0 for background, 1.0 for person)
            float mask = texture(u_mask_texture, uv).r;

            // Mix between the two fluid simulations based on the mask
            vec3 final_color = mix(background_fluid_color, person_fluid_color, mask);
            
            fragColor = vec4(final_color, 1.0);
        }
    """,
)

# --- 3. Geometry and Texture Setup ---
vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype="f4")
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(program, vbo, "in_vert")
mask_texture = ctx.texture((WIDTH, HEIGHT), 1, dtype="f4")
start_time = time.time()

# --- 4. Main Render Loop ---
print("Running main loop. Step in front of the camera. Close the window to quit.")
while not glfw.window_should_close(window):
    # Part A: Get the mask from MediaPipe
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = segmenter.process(rgb_frame)
    mask = results.segmentation_mask
    if mask is None:
        continue

    mask = (mask > 0.5).astype("f4")
    mask_flipped = cv2.flip(mask, 0)
    mask_resized = cv2.resize(mask_flipped, (WIDTH, HEIGHT))

    # Part B: Run the Shader
    ctx.clear(0.1, 0.1, 0.1)

    mask_texture.write(mask_resized.tobytes())
    mask_texture.use(location=0)
    program["u_mask_texture"].value = 0
    program["u_time"].value = time.time() - start_time

    vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    glfw.swap_buffers(window)
    glfw.poll_events()

# --- Cleanup ---
cap.release()
segmenter.close()
glfw.terminate()
