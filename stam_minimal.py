# step2_swirl_advection.py
# Minimal stable-fluid style dye advection using an analytic swirl velocity field.

import glfw, moderngl, numpy as np, time, math

WIDTH, HEIGHT = 800, 600
TARGET_FPS = 60
DYE_DISSIPATION = 0.998     # <1.0 -> slow fade
DT = 1.0 / TARGET_FPS       # fixed timestep
SPLAT_INTERVAL = 0.7        # seconds between color injections

VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){
    gl_Position = vec4(in_vert,0.0,1.0);
    uv = (in_vert + 1.0)*0.5;
}
"""

# Advection pass: move dye along a fixed swirling velocity field
FS_ADVECT = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D dye_src;
uniform float dt;
uniform float dissipation;

// Analytic velocity field swirling around the center
vec2 velocity(vec2 p){
    vec2 c = p - vec2(0.5);
    // rotate 90 degrees (perpendicular) for swirl
    vec2 v = vec2(-c.y, c.x);
    float r = length(c);
    // speed decreases with distance
    v *= 0.25 * exp(-3.0 * r);
    return v;
}

void main(){
    vec2 p = uv;
    vec2 v = velocity(p);
    // backtrace in UV space
    vec2 prev = p - dt * v;
    prev = fract(prev); // wrap edges
    vec4 col = texture(dye_src, prev);
    fragColor = col * dissipation;
}
"""

# Inject dye (splat) pass
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

# Display pass
FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
void main(){ fragColor = texture(dye, uv); }
"""

def make_tex(ctx, size, comps=4):
    tex = ctx.texture(size, comps, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = True
    tex.repeat_y = True
    return tex

def fullscreen_quad(ctx):
    verts = np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4')
    return ctx.buffer(verts)

def main():
    # ---- window ----
    if not glfw.init(): raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Step 2: Swirl Advection", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # ---- programs ----
    prog_adv = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    vbo = fullscreen_quad(ctx)
    vao_adv = ctx.simple_vertex_array(prog_adv, vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_show = ctx.simple_vertex_array(prog_show, vbo, "in_vert")

    # ---- textures ----
    dye_a = make_tex(ctx, (WIDTH, HEIGHT))
    dye_b = make_tex(ctx, (WIDTH, HEIGHT))
    fbo_a = ctx.framebuffer([dye_a])
    fbo_b = ctx.framebuffer([dye_b])
    for fbo in (fbo_a, fbo_b):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # seed a small blob in the center
    fbo_b.use()
    dye_a.use(location=0)
    prog_splat["dye_src"].value = 0
    prog_splat["point"].value = (0.5,0.5)
    prog_splat["color"].value = (0.7,0.3,0.9)
    prog_splat["radius"].value = 0.1
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a, dye_b = dye_b, dye_a
    fbo_a, fbo_b = fbo_b, fbo_a

    start = time.time()
    last_splat = start

    # ---- loop ----
    print("Running. ESC to quit.")
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        now = time.time()

        # 1) Advect dye: read dye_a -> write dye_b
        fbo_b.use()
        ctx.clear(0,0,0,1)
        dye_a.use(location=0)
        prog_adv["dye_src"].value = 0
        prog_adv["dt"].value = DT
        prog_adv["dissipation"].value = DYE_DISSIPATION
        vao_adv.render(moderngl.TRIANGLE_STRIP)
        dye_a, dye_b = dye_b, dye_a
        fbo_a, fbo_b = fbo_b, fbo_a

        # 2) Periodically inject new dye around a circle
        if now - last_splat > SPLAT_INTERVAL:
            last_splat = now
            t = now - start
            u = 0.5 + 0.25*math.cos(t*0.6)
            v = 0.5 + 0.25*math.sin(t*0.6)
            col = (
                0.5 + 0.5*math.sin(2.3*t + 0.0),
                0.5 + 0.5*math.sin(2.3*t + 2.1),
                0.5 + 0.5*math.sin(2.3*t + 4.2)
            )
            fbo_b.use()
            ctx.clear(0,0,0,1)
            dye_a.use(location=0)
            prog_splat["dye_src"].value = 0
            prog_splat["point"].value = (u,v)
            prog_splat["color"].value = col
            prog_splat["radius"].value = 0.05
            vao_splat.render(moderngl.TRIANGLE_STRIP)
            dye_a, dye_b = dye_b, dye_a
            fbo_a, fbo_b = fbo_b, fbo_a

        # 3) Show
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
