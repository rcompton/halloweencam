# step2_swirl_confirm.py
# Same stable swirl advection as your working version, plus:
# - runtime swirl strength control ([ and ])
# - many micro-splats for wispy look (instead of 1 big blob)
# - non-linear display mapping (orange→purple palette)
# - space toggles advection; C clears dye

import glfw, moderngl, numpy as np, time, math

# ---------------- Config ----------------
WIDTH, HEIGHT = 800, 600
TARGET_FPS = 60
SUBSTEPS = 4  # smaller moves per frame (2–6)
DYE_DISSIPATION = 1.0  # keep mass; we’ll control look in display
MICRO_SPLATS_PER_FRAME = 6  # more = smoother, smokier
MICRO_SPLAT_RADIUS = 0.02  # small puffs
MICRO_SPLAT_STRENGTH = 0.08  # per-splat color intensity
ORBIT_SOURCE = True  # True = slow orbit; False = center source
ORBIT_R = 0.22
ORBIT_SPEED = 0.35
DYE_BLUR = True  # mild blur to hide pixel noise

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

# advection with uniform swirl strength, clamped edges, small per-substep move
FS_ADVECT = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D dye_src;
uniform float dt;
uniform float dissipation;
uniform float swirl_strength;   // runtime adjustable

vec2 swirl(vec2 p){
    vec2 c = p - vec2(0.5);
    float r = length(c);
    // swirl_strength is multiplied by a radial falloff
    vec2 v = vec2(-c.y, c.x) * (swirl_strength * exp(-2.2*r));
    return v;
}

void main(){
    vec2 v = swirl(uv);
    vec2 prev = uv - v * dt;
    prev = clamp(prev, vec2(0.001), vec2(0.999));
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
    // halloween-ish: black->orange->magenta->purple
    // t in [0,1]
    t = clamp(t, 0.0, 1.0);
    if (t < 0.5) {
        float k = t / 0.5;                 // 0..1
        return mix(vec3(0.0), vec3(1.0,0.5,0.0), k);   // black -> orange
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
        // luminance then soft tone-map
        float l = dot(c, vec3(0.299, 0.587, 0.114));
        l = 1.0 - exp(-2.5*l);             // soft roll-off
        fragColor = vec4(palette(l), 1.0);
    }
}
"""


# ---------------- Utils ----------------
def make_tex(ctx, size, comps=4):
    tex = ctx.texture(size, comps, dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def fullscreen_quad(ctx):
    verts = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    return ctx.buffer(verts)


# ---------------- Main ----------------
def main():
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(WIDTH, HEIGHT, "Swirl Confirm", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # programs
    prog_adv = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_blur = ctx.program(vertex_shader=VS, fragment_shader=FS_BLUR)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    vbo = fullscreen_quad(ctx)
    vao_adv = ctx.simple_vertex_array(prog_adv, vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_blur = ctx.simple_vertex_array(prog_blur, vbo, "in_vert")
    vao_show = ctx.simple_vertex_array(prog_show, vbo, "in_vert")

    # textures
    dye_a = make_tex(ctx, (WIDTH, HEIGHT))
    dye_b = make_tex(ctx, (WIDTH, HEIGHT))
    tmp = make_tex(ctx, (WIDTH, HEIGHT))
    fbo_a = ctx.framebuffer([dye_a])
    fbo_b = ctx.framebuffer([dye_b])
    fbo_t = ctx.framebuffer([tmp])
    for fbo in (fbo_a, fbo_b, fbo_t):
        fbo.use()
        ctx.clear(0, 0, 0, 1)
    ctx.screen.use()

    # seed: a soft puff so there's motion immediately
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
    prog_blur["texel"].value = (1.0 / WIDTH, 1.0 / HEIGHT)
    palette_on = 1
    swirl_strength = 0.6  # runtime adjustable
    advection_enabled = True

    start = prev = time.time()
    print(
        "Controls: [ ] = swirl +/-   SPACE=toggle advect   C=clear   P=palette toggle"
    )

    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
            break

        # controls
        if glfw.get_key(win, glfw.KEY_LEFT_BRACKET) == glfw.PRESS:
            swirl_strength = max(0.1, swirl_strength - 0.01)
        if glfw.get_key(win, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS:
            swirl_strength = min(2.5, swirl_strength + 0.01)
        if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS:
            advection_enabled = not advection_enabled
            time.sleep(0.12)
        if glfw.get_key(win, glfw.KEY_C) == glfw.PRESS:
            fbo_a.use()
            ctx.clear(0, 0, 0, 1)
            fbo_b.use()
            ctx.clear(0, 0, 0, 1)
            ctx.screen.use()
            time.sleep(0.12)
        if glfw.get_key(win, glfw.KEY_P) == glfw.PRESS:
            palette_on = 1 - palette_on
            time.sleep(0.12)

        now = time.time()
        dt_frame = min(0.033, now - prev)  # ~30fps clamp
        prev = now

        # --- advect (substeps) ---
        if advection_enabled:
            sub_dt = dt_frame / SUBSTEPS
            for _ in range(SUBSTEPS):
                fbo_b.use()
                dye_a.use(location=0)
                prog_adv["dye_src"].value = 0
                prog_adv["dt"].value = sub_dt
                prog_adv["dissipation"].value = DYE_DISSIPATION
                prog_adv["swirl_strength"].value = swirl_strength
                vao_adv.render(moderngl.TRIANGLE_STRIP)
                dye_a, dye_b = dye_b, dye_a
                fbo_a, fbo_b = fbo_b, fbo_a

        # --- micro-splats each frame (wispy source) ---
        t = now - start
        base_u, base_v = 0.5, 0.5
        if ORBIT_SOURCE:
            base_u = 0.5 + ORBIT_R * math.cos(t * ORBIT_SPEED)
            base_v = 0.5 + ORBIT_R * math.sin(t * ORBIT_SPEED)

        for i in range(MICRO_SPLATS_PER_FRAME):
            # jitter around base point
            ang = 6.28318 * (i / MICRO_SPLATS_PER_FRAME)
            jitter = 0.012 * np.array(
                [math.cos(ang + t * 0.5), math.sin(ang + t * 0.5)]
            )
            u = float(base_u + jitter[0])
            v = float(base_v + jitter[1])

            # halloween-ish alternating colors
            hue = (i / MICRO_SPLATS_PER_FRAME + 0.2 * math.sin(t * 0.7)) % 1.0
            # quick 2-color blend: orange<->purple
            orange = np.array([1.0, 0.5, 0.0])
            purple = np.array([0.45, 0.0, 0.6])
            col = orange * (1.0 - hue) + purple * hue
            col = (col * MICRO_SPLAT_STRENGTH).astype("f4")

            fbo_b.use()
            dye_a.use(location=0)
            prog_splat["dye_src"].value = 0
            prog_splat["point"].value = (u, v)
            prog_splat["color"].value = (float(col[0]), float(col[1]), float(col[2]))
            prog_splat["radius"].value = MICRO_SPLAT_RADIUS
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
        prog_show["palette_on"].value = palette_on
        vao_show.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(win)
        time.sleep(max(0, 1.0 / TARGET_FPS - (time.time() - now)))

    glfw.terminate()


if __name__ == "__main__":
    main()
