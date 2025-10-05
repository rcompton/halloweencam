# step1_animated_gradient.py
# Expands on the FBO test — adds time-based UV distortion.
# Confirms dynamic sampling and time uniforms work before adding fluid logic.

import glfw, moderngl, numpy as np, time

WIDTH, HEIGHT = 800, 600

VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){
    gl_Position = vec4(in_vert,0.0,1.0);
    uv = (in_vert + 1.0)*0.5;
}
"""

# Animated gradient with sinusoidal distortion
FS_RED = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform float u_time;

void main(){
    vec2 p = uv;
    // apply a small swirl-like displacement
    p.x += 0.02 * sin(6.2831 * (p.y * 3.0 + u_time*0.4));
    p.y += 0.02 * cos(6.2831 * (p.x * 3.0 + u_time*0.3));

    vec3 color;
    color.r = 0.5 + 0.5*sin(6.2831*(p.x + u_time*0.1));
    color.g = 0.5 + 0.5*sin(6.2831*(p.y + u_time*0.13));
    color.b = 0.5 + 0.5*sin(6.2831*(p.x+p.y+u_time*0.09));

    fragColor = vec4(color, 1.0);
}
"""

FS_SHOW = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D tex;
void main(){
    fragColor = texture(tex, uv);
}
"""

def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Animated Gradient Base", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    verts = np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4')
    vbo = ctx.buffer(verts)
    prog_red = ctx.program(vertex_shader=VS, fragment_shader=FS_RED)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    vao_red = ctx.simple_vertex_array(prog_red, vbo, "in_vert")
    vao_show = ctx.simple_vertex_array(prog_show, vbo, "in_vert")

    tex = ctx.texture((WIDTH, HEIGHT), 4, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    fbo = ctx.framebuffer(color_attachments=[tex])

    start = time.time()

    while not glfw.window_should_close(win):
        glfw.poll_events()
        now = time.time() - start

        # draw animated gradient into FBO
        fbo.use()
        ctx.clear(0,0,0,1)
        prog_red["u_time"].value = now
        vao_red.render(moderngl.TRIANGLE_STRIP)

        # blit to screen
        ctx.screen.use()
        ctx.clear(0,0,0,1)
        tex.use(location=0)
        prog_show["tex"].value = 0
        vao_show.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(win)

    glfw.terminate()

if __name__ == "__main__":
    main()
