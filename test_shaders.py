import glfw
import moderngl
import numpy as np
import time

# --- 1. Window and Context Setup ---
if not glfw.init():
    raise Exception("GLFW can't be initialized")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

window = glfw.create_window(1280, 720, "Procedural Noise Fluid", None, None)
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

        // 2D Noise function (a standard algorithm)
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
            // Define two layers of noise moving at different speeds for a swirling effect
            vec2 uv_layer1 = uv * 3.0 + vec2(u_time * 0.1, u_time * 0.05);
            vec2 uv_layer2 = uv * 4.0 - vec2(u_time * 0.03, u_time * 0.08);

            // Use noise to distort the texture coordinates (domain warping)
            float distortion = noise(uv_layer2) * 0.5;
            
            // Generate the final color based on the distorted coordinates
            float color_value = noise(uv_layer1 + distortion);

            // Map the noise value to a color palette (blue to orange)
            vec3 color = mix(vec3(0.1, 0.4, 0.8), vec3(1.0, 0.7, 0.2), color_value);
            
            fragColor = vec4(color, 1.0);
        }
    """
)

# --- 3. Geometry Setup ---
vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype='f4')
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(program, vbo, 'in_vert')

# Get the starting time
start_time = time.time()

# --- 4. Main Render Loop ---
print("Shader running. Close the window to quit.")
while not glfw.window_should_close(window):
    ctx.clear(0.1, 0.1, 0.1)

    # Pass the elapsed time to the shader
    program['u_time'].value = time.time() - start_time
    
    # Render the VAO
    vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    # Swap buffers
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()