# step3_mouse_swirl.py
# Mouse-interactive swirl advection (small extension of the working swirl build).
# - Click & drag (LMB): inject dye + locally push the flow
# - [ / ] : swirl strength -/+
# - SPACE : toggle advection
# - C     : clear dye
# - P     : toggle palette

import glfw, moderngl, numpy as np, time, math

# ---------------- Config ----------------
WIDTH, HEIGHT = 800, 600
TARGET_FPS = 60
SUBSTEPS = 4                   # smaller moves per frame (2–6)
DYE_DISSIPATION = 0.999          # keep mass; look is controlled in display
DYE_BLUR = True                # mild post blur to reduce pixel noise
# Micro source to keep things alive visually (disable if you want pure mouse)
ORBIT_SOURCE = True
ORBIT_R = 0.18
ORBIT_SPEED = 0.35
MICRO_SPLATS_PER_FRAME = 4
MICRO_SPLAT_RADIUS = 0.018
MICRO_SPLAT_STRENGTH = 0.06

# Mouse “force” parameters
MOUSE_RADIUS = 0.10           # influence radius (UV units)
MOUSE_FORCE_SCALE = 2.2       # multiply mouse velocity (UV/sec) -> advection force

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

# Advection with analytic swirl + local mouse push (Gaussian)
FS_ADVECT = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D dye_src;
uniform float dt;
uniform float dissipation;
uniform float swirl_strength;

uniform vec2  mouse_pos;      // UV
uniform vec2  mouse_force;    // UV/sec
uniform float mouse_radius;   // UV (Gaussian sigma-ish)

vec2 swirl(vec2 p){
    vec2 c = p - vec2(0.5);
    float r = length(c);
    // analytic swirl baseline
    vec2 v = vec2(-c.y, c.x) * (swirl_strength * exp(-2.2*r));
    return v;
}

void main(){
    vec2 v = swirl(uv);

    // local mouse push: Gaussian falloff around mouse_pos
    vec2 d = uv - mouse_pos;
    float fall = exp(-dot(d,d) / (mouse_radius*mouse_radius));
    v += mouse_force * fall;

    // small backtrace
    vec2 prev = uv - v * dt;
    prev = clamp(prev, vec2(0.001), vec2(0.999));  // container walls (no wrap)
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

# mild 9-tap blur (visual only)
FS_BLUR = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D src;
uniform vec2 texel;
void main(){
    vec4 s = vec4(0.0);
    for(int j=-1;j<=1;++j)
        for(int i=-1;i<=1;++i)
            s += texture(src, uv + vec2(i,j)*texel);
    fragColor = s / 9.0;
}
"""

# palette display: map dye rgb -> luminance then colorize orange→purple
FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
uniform int palette_on;

vec3 palette(float t){
    t = clamp(t, 0.0, 1.0);
    if (t < 0.5) {
        float k = t / 0.5;
        return mix(vec3(0.0), vec3(1.0,0.5,0.0), k);    // black -> orange
    } else {
        float k = (t - 0.5) / 0.5;
        return mix(vec3(1.0,0.5,0.0), vec3(0.45,0.0,0.6), k); // orange -> purple
    }
}

void main(){
    vec3 c = texture(dye, uv).rgb;
    if (palette_on == 0){
        fragColor = vec4(c,1.0);
    } else {
        float l = dot(c, vec3(0.299, 0.587, 0.114));
        l = 1.0 - exp(-2.5*l);
        fragColor = vec4(palette(l), 1.0);
    }
}
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

# ---------------- Mouse state ----------------
class Mouse:
    def __init__(self, w, h):
        self.win_w = w
        self.win_h = h
        self.x = w * 0.5
        self.y = h * 0.5
        self.px = self.x
        self.py = self.y
        self.down = False
        self.uv = (0.5, 0.5)
        self.puv = (0.5, 0.5)

    def set_pos(self, x, y):
        self.px, self.py = self.x, self.y
        self.x, self.y = x, y
        # window coords -> UV (note y flip)
        u = np.clip(x / self.win_w, 0.0, 1.0)
        v = np.clip(1.0 - y / self.win_h, 0.0, 1.0)
        self.puv = self.uv
        self.uv = (float(u), float(v))

    def set_button(self, down):
        self.down = down

# ---------------- Main ----------------
def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Mouse Swirl Fluid", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # Mouse handlers
    mouse = Mouse(WIDTH, HEIGHT)
    def on_cursor(_, x, y): mouse.set_pos(x, y)
    def on_button(_, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse.set_button(action == glfw.PRESS)
    glfw.set_cursor_pos_callback(win, on_cursor)
    glfw.set_mouse_button_callback(win, on_button)

    # programs
    prog_adv  = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat= ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_blur = ctx.program(vertex_shader=VS, fragment_shader=FS_BLUR)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    vbo = fullscreen_quad(ctx)
    vao_adv   = ctx.simple_vertex_array(prog_adv,   vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_blur  = ctx.simple_vertex_array(prog_blur,  vbo, "in_vert")
    vao_show  = ctx.simple_vertex_array(prog_show,  vbo, "in_vert")

    # textures
    dye_a = make_tex(ctx, (WIDTH, HEIGHT))
    dye_b = make_tex(ctx, (WIDTH, HEIGHT))
    tmp   = make_tex(ctx, (WIDTH, HEIGHT))
    fbo_a = ctx.framebuffer([dye_a])
    fbo_b = ctx.framebuffer([dye_b])
    fbo_t = ctx.framebuffer([tmp])
    for fbo in (fbo_a, fbo_b, fbo_t):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # seed: soft puff
    fbo_b.use()
    dye_a.use(location=0)
    prog_splat["dye_src"].value = 0
    prog_splat["point"].value = (0.5, 0.5)
    prog_splat["color"].value = (0.9, 0.4, 0.9)
    prog_splat["radius"].value = 0.12
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a, dye_b = dye_b, dye_a
    fbo_a, fbo_b = fbo_b, fbo_a

    # uniforms
    prog_blur["texel"].value = (1.0/WIDTH, 1.0/HEIGHT)
    palette_on = 1
    swirl_strength = 0.6
    advection_enabled = True

    start = prev = time.time()
    print("Controls: [ ] = swirl +/-   SPACE=toggle advect   C=clear   P=palette toggle   LMB drag = draw & push")
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # hotkeys
        if glfw.get_key(win, glfw.KEY_LEFT_BRACKET) == glfw.PRESS:
            swirl_strength = max(0.05, swirl_strength - 0.01)
        if glfw.get_key(win, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS:
            swirl_strength = min(3.0, swirl_strength + 0.01)
        if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS:
            advection_enabled = not advection_enabled; time.sleep(0.12)
        if glfw.get_key(win, glfw.KEY_C) == glfw.PRESS:
            fbo_a.use(); ctx.clear(0,0,0,1); fbo_b.use(); ctx.clear(0,0,0,1); ctx.screen.use(); time.sleep(0.12)
        if glfw.get_key(win, glfw.KEY_P) == glfw.PRESS:
            palette_on = 1 - palette_on; time.sleep(0.12)

        now = time.time()
        dt_frame = min(0.033, now - prev)             # ~30 fps clamp
        prev = now

        # compute mouse-based force in UV/sec
        du = (mouse.uv[0] - mouse.puv[0]) / max(dt_frame, 1e-5)
        dv = (mouse.uv[1] - mouse.puv[1]) / max(dt_frame, 1e-5)
        mforce = (du * MOUSE_FORCE_SCALE, dv * MOUSE_FORCE_SCALE) if mouse.down else (0.0, 0.0)

        # --- advect (substeps) ---
        if advection_enabled:
            sub_dt = dt_frame / SUBSTEPS
            for _ in range(SUBSTEPS):
                fbo_b.use()
                dye_a.use(location=0)
                prog_adv["dye_src"].value   = 0
                prog_adv["dt"].value        = sub_dt
                prog_adv["dissipation"].value = DYE_DISSIPATION
                prog_adv["swirl_strength"].value = swirl_strength
                prog_adv["mouse_pos"].value   = mouse.uv
                prog_adv["mouse_force"].value = mforce
                prog_adv["mouse_radius"].value= MOUSE_RADIUS
                vao_adv.render(moderngl.TRIANGLE_STRIP)
                dye_a, dye_b = dye_b, dye_a
                fbo_a, fbo_b = fbo_b, fbo_a

        # --- continuous tiny source (optional) ---
        if ORBIT_SOURCE:
            t = now - start
            base_u = 0.5 + ORBIT_R * math.cos(t * ORBIT_SPEED)
            base_v = 0.5 + ORBIT_R * math.sin(t * ORBIT_SPEED)
        else:
            base_u, base_v = 0.5, 0.5

        for i in range(MICRO_SPLATS_PER_FRAME):
            # jitter ring around base
            ang = 2*math.pi * (i / MICRO_SPLATS_PER_FRAME + 0.1 * math.sin((now-start)*0.4))
            u = base_u + 0.010 * math.cos(ang)
            v = base_v + 0.010 * math.sin(ang)
            # halloween-ish orange/purple blend
            k = (i / MICRO_SPLATS_PER_FRAME + 0.2 * math.sin((now-start)*0.7)) % 1.0
            orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
            purple = np.array([0.45, 0.0, 0.6], dtype=np.float32)
            col = (orange * (1.0 - k) + purple * k) * MICRO_SPLAT_STRENGTH

            fbo_b.use()
            dye_a.use(location=0)
            prog_splat["dye_src"].value = 0
            prog_splat["point"].value = (float(u), float(v))
            prog_splat["color"].value = (float(col[0]), float(col[1]), float(col[2]))
            prog_splat["radius"].value = MICRO_SPLAT_RADIUS
            vao_splat.render(moderngl.TRIANGLE_STRIP)
            dye_a, dye_b = dye_b, dye_a
            fbo_a, fbo_b = fbo_b, fbo_a

        # --- mouse painting (dye) when dragging ---
        if mouse.down:
            fbo_b.use()
            dye_a.use(location=0)
            prog_splat["dye_src"].value = 0
            prog_splat["point"].value = mouse.uv
            # bright dye but small radius so it doesn't blow out
            prog_splat["color"].value = (0.9, 0.9, 0.9)
            prog_splat["radius"].value = 0.022
            vao_splat.render(moderngl.TRIANGLE_STRIP)
            dye_a, dye_b = dye_b, dye_a
            fbo_a, fbo_b = fbo_b, fbo_a

        # --- optional mild blur ---
        if DYE_BLUR:
            fbo_t.use()
            dye_a.use(location=0)
            prog_blur["src"].value = 0
            vao_blur.render(moderngl.TRIANGLE_STRIP)
            dye_a, tmp = tmp, dye_a
            fbo_a, fbo_t = fbo_t, fbo_a

        # --- display ---
        ctx.screen.use()
        dye_a.use(location=0)
        prog_show["dye"].value = 0
        prog_show["palette_on"].value = 1
        vao_show.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(win)
        time.sleep(max(0, 1.0/TARGET_FPS - (time.time()-now)))

    glfw.terminate()

if __name__ == "__main__":
    main()
