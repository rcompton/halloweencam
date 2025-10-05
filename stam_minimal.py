# step2_smooth_swirl.py
# Stable swirl advection (small incremental step from your working version)

import glfw, moderngl, numpy as np, time, math

# ---------------- Config ----------------
WIDTH, HEIGHT = 800, 600
TARGET_FPS = 60
DYE_DISSIPATION = 0.998
SUBSTEPS = 4              # smaller moves per frame (2–6)
CLAMP_EDGES = True
SPLAT_INTERVAL = 0.7

# ---------------- Shaders ----------------
VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){
    gl_Position = vec4(in_vert, 0.0, 1.0);
    uv = (in_vert + 1.0) * 0.5;
}
"""

FS_ADVECT = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D dye_src;
uniform float dt;
uniform float dissipation;

// analytic swirl velocity (like rotation)
vec2 swirl(vec2 p){
    vec2 c = p - vec2(0.5);
    float r = length(c);
    vec2 v = vec2(-c.y, c.x) * (0.6 * exp(-2.2*r));
    return v;
}

void main(){
    vec2 v = swirl(uv);
    vec2 prev = uv - v * dt;         // small move
    prev = clamp(prev, vec2(0.001), vec2(0.999));  // avoid wrap seams
    vec4 col = texture(dye_src, prev);
    fragColor = col * dissipation;
}
"""

FS_SPLAT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye_src;
uniform vec2 point;
uniform vec3 color;
uniform float radius;

void main(){
    vec4 base = texture(dye_src, uv);
    vec2 d = uv - point;
    float fall = exp(-dot(d,d)/(radius*radius));
    fragColor = base + vec4(color * fall, 1.0);
}
"""

FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
void main(){ fragColor = texture(dye, uv); }
"""

# ---------------- Utils ----------------
def make_tex(ctx, size, comps=4):
    tex = ctx.texture(size, comps, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex

def fullscreen_quad(ctx):
    verts = np.array([-1,-1,1,-1,-1,1,1,1], dtype='f4')
    return ctx.buffer(verts)

# ---------------- Main ----------------
def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Step 2: Smooth Swirl", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # programs
    prog_adv = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    vbo = fullscreen_quad(ctx)
    vao_adv = ctx.simple_vertex_array(prog_adv, vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_show = ctx.simple_vertex_array(prog_show, vbo, "in_vert")

    # textures and fbos
    dye_a = make_tex(ctx, (WIDTH, HEIGHT))
    dye_b = make_tex(ctx, (WIDTH, HEIGHT))
    fbo_a = ctx.framebuffer([dye_a])
    fbo_b = ctx.framebuffer([dye_b])
    for fbo in (fbo_a, fbo_b):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # seed center blob
    fbo_b.use()
    dye_a.use(location=0)
    prog_splat["dye_src"].value = 0
    prog_splat["point"].value = (0.5,0.5)
    prog_splat["color"].value = (0.8,0.4,0.9)
    prog_splat["radius"].value = 0.1
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a, dye_b = dye_b, dye_a
    fbo_a, fbo_b = fbo_b, fbo_a

    prev = time.time()
    start = prev
    print("Running. ESC to quit.")

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        now = time.time()
        dt_frame = min(0.033, now - prev)
        prev = now

        # --- advect (multiple small substeps) ---
        sub_dt = dt_frame / SUBSTEPS
        for _ in range(SUBSTEPS):
            fbo_b.use()
            dye_a.use(location=0)
            prog_adv["dye_src"].value = 0
            prog_adv["dt"].value = sub_dt
            prog_adv["dissipation"].value = DYE_DISSIPATION
            vao_adv.render(moderngl.TRIANGLE_STRIP)
            dye_a, dye_b = dye_b, dye_a
            fbo_a, fbo_b = fbo_b, fbo_a

        # --- show ---
        ctx.screen.use()
        ctx.clear(0,0,0,1)
        dye_a.use(location=0)
        prog_show["dye"].value = 0
        vao_show.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(win)
        time.sleep(max(0, 1.0/TARGET_FPS - (time.time()-now)))

    glfw.terminate()

if __name__ == "__main__":
    main()
