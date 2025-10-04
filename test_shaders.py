import glfw
import moderngl
import numpy as np

# --- 1. Window Setup using GLFW ---
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# --- FIX: Add window hints to request an OpenGL 3.3 Core Profile ---
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True) # Required on macOS

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(1280, 720, "ModernGL Shader Test", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

# Make the window's context current
glfw.make_context_current(window)

# --- 2. GPU Program Setup using ModernGL ---
ctx = moderngl.create_context()

# GLSL code for the shaders
vertex_shader_source = """
    #version 330
    in vec2 in_vert;
    in vec2 in_uv;
    out vec2 uv;
    void main() {
        gl_Position = vec4(in_vert, 0.0, 1.0);
        uv = in_uv;
    }
"""
fragment_shader_source = """
    #version 330
    in vec2 uv;
    out vec4 fragColor;
    void main() {
        fragColor = vec4(uv.x, uv.y, 0.5, 1.0);
    }
"""

prog = ctx.program(vertex_shader=vertex_shader_source, fragment_shader=fragment_shader_source)

# --- 3. Geometry Setup ---
vertices = np.array([
    -1.0, -1.0, 0.0, 0.0,
     1.0, -1.0, 1.0, 0.0,
    -1.0,  1.0, 0.0, 1.0,
     1.0,  1.0, 1.0, 1.0,
], dtype='f4')
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_uv')

# --- 4. Main Render Loop ---
print("Shader running. Close the window to quit.")
while not glfw.window_should_close(window):
    ctx.clear(0.1, 0.1, 0.1)
    vao.render(moderngl.TRIANGLE_STRIP)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()