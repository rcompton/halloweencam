import glfw
import moderngl
import numpy as np

# --- 1. Window Setup using GLFW ---
if not glfw.init():
    raise Exception("GLFW can't be initialized")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)

window = glfw.create_window(1280, 720, "ModernGL Advection Test", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.make_context_current(window)

# --- 2. GPU Program Setup using ModernGL ---
ctx = moderngl.create_context()

advection_program = ctx.program(
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
        uniform sampler2D DyeTexture;
        uniform float TimeStep = 0.01;
        void main() {
            vec2 velocity = vec2(0.5 - uv.y, uv.x - 0.5) * 2.0;
            vec2 source_uv = uv - velocity * TimeStep;
            fragColor = texture(DyeTexture, source_uv);
        }
    """,
)

display_program = ctx.program(
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
        uniform sampler2D TextureToDisplay;
        void main() {
            fragColor = texture(TextureToDisplay, uv);
        }
    """,
)

# --- 3. Geometry and Texture Setup ---
vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0], dtype="f4")
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(advection_program, vbo, "in_vert")
display_vao = ctx.simple_vertex_array(display_program, vbo, "in_vert")

width, height = 1280, 720
texture_a = ctx.texture((width, height), 4, dtype="f4")
texture_b = ctx.texture((width, height), 4, dtype="f4")

fbo_a = ctx.framebuffer(color_attachments=[texture_a])
fbo_b = ctx.framebuffer(color_attachments=[texture_b])

# --- 4. Initial State ---
# --- FIX: Create the initial image with NumPy and write it directly to the texture ---
# Create a black image with 4 channels (R, G, B, A)
initial_data = np.zeros((height, width, 4), dtype="f4")
# Draw a thick horizontal orange line in the middle
line_y = height // 2
line_thickness = 5
initial_data[
    line_y - line_thickness : line_y + line_thickness, width // 4 : width * 3 // 4
] = (1.0, 0.7, 0.2, 1.0)
# Write the NumPy array data directly to texture_a
texture_a.write(initial_data.tobytes())


# --- 5. Main Render Loop ---
print("Shader running. Close the window to quit.")
while not glfw.window_should_close(window):
    # PING: Render to texture B from texture A
    fbo_b.use()
    texture_a.use(location=0)
    advection_program["DyeTexture"].value = 0
    vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    # PONG: Render to the screen from texture B
    ctx.screen.use()
    texture_b.use(location=0)
    display_program["TextureToDisplay"].value = 0
    display_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)

    # Swap buffers
    glfw.swap_buffers(window)
    glfw.poll_events()

    # SWAP textures for the next frame
    texture_a, texture_b = texture_b, texture_a
    fbo_a, fbo_b = fbo_b, fbo_a

glfw.terminate()
