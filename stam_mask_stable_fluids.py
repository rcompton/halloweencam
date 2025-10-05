# step6_mask_edges_fluids.py
# Stable fluids with edge-based forces from a MediaPipe segmentation mask.
# - GPU computes mask gradient (edge normals) on sim grid
# - Adds velocity along edge normals + tangential component (vortices)
# - Optionally inject dye along edges (to "see" your silhouette)
# Keys: SPACE pause, C clear, [/] vorticity -/+, P palette toggle, M toggle mask/mouse

import glfw, moderngl, numpy as np, time, math, cv2
import mediapipe as mp

# --- YOLO instance segmentation (multi-person) ---
import torch
from ultralytics import YOLO

_YOLO_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# small and fast; you can try 'yolov8s-seg.pt' for better edges if you have headroom
_yolo = YOLO("yolov8n-seg.pt")  # downloads on first run
_yolo.to(_YOLO_DEVICE)


def people_mask_yolo(frame_bgr, out_w, out_h, conf=0.25):
    """
    Returns a float32 mask in [0,1] of size (out_h, out_w) that is the UNION
    of all 'person' instance masks (class id 0 in COCO).
    """
    # YOLO expects RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # run at a reasonable size (you can raise imgsz for cleaner edges)
    res = _yolo.predict(rgb, imgsz=640, device=_YOLO_DEVICE, conf=conf, verbose=False)[
        0
    ]

    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return np.zeros((out_h, out_w), np.float32)

    # select only class==0 (person)
    cls = res.boxes.cls.cpu().numpy().astype(int)
    keep = np.where(cls == 0)[0]
    if keep.size == 0:
        return np.zeros((out_h, out_w), np.float32)

    # res.masks.data is (N, mh, mw) float masks in 0..1
    m = res.masks.data[keep].cpu().numpy()  # (K, mh, mw)
    # union of selected masks
    union = np.clip(np.max(m, axis=0), 0, 1)

    # resize to your SIM grid (or window) and FLIP V so UV↑ matches image↓
    union = cv2.resize(
        union.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    union = cv2.flip(union, 0).astype(np.float32)
    return union


# ---------------- Config ----------------
WIDTH, HEIGHT = 1024, 576
SIM_SCALE = 0.9  # sim grid = SIM_SCALE * window
SUBSTEPS = 5
DT_CLAMP = 0.033
JACOBI_ITERS = 60
VEL_DISSIPATION = 0.999
DYE_DISSIPATION = 0.99
VORTICITY = 5.0
PALETTE_ON = 1

# Mask / CV
CAMERA_INDEX = 0
MASK_THRESHOLD = 0.15  # for centroid fallback & noise trimming (not needed for force)
MASK_MIN_AREA = 0.02

# Edge-based force params (primary knobs)
EDGE_THRESH = 0.01  # minimum gradient magnitude to be considered an edge
EDGE_NORMAL_AMP = 1.0  # push along normals (higher = stronger "shove")
EDGE_TANGENTIAL_AMP = 1.0  # spin along edge (vortex seeding)
EDGE_USE_TEMPORAL = True  # scale normal push by mask growth/shrink
EDGE_DYE_STRENGTH = 0.10  # dye added along edges each step
EDGE_DYE_RADIUS = (
    0.001  # 0 => per-pixel add; >0 => soften via small blur pass (we keep 0)
)

# Fallback ambient source when mask is missing
USE_FALLBACK_ORBIT = True
ORBIT_R = 0.18
ORBIT_SPEED = 0.35
MICRO_SPLATS_PER_FRAME = 2
MICRO_SPLAT_RADIUS = 0.010
MICRO_SPLAT_STRENGTH = 0.02

# ---------------- Shaders ----------------
VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){ gl_Position = vec4(in_vert,0.0,1.0); uv = (in_vert + 1.0)*0.5; }
"""

# Generic advection (vel advects vel or dye)
FS_ADVECT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform sampler2D src;
uniform float dt;
uniform float dissipation;
void main(){
    vec2 v = texture(vel, uv).xy;
    vec2 prev = uv - v * dt;
    prev = clamp(prev, vec2(0.001), vec2(0.999));
    fragColor = texture(src, prev) * dissipation;
}
"""

# Splat adds to a field (vec2 for velocity in xy, vec3 for dye rgb)
FS_SPLAT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec2 point;
uniform vec3 value;
uniform float radius;
void main(){
    vec4 base = texture(field, uv);
    vec2 d = uv - point;
    float fall = exp(-dot(d,d)/(radius*radius));
    fragColor = base + vec4(value * fall, 0.0);
}
"""

# Divergence -> Pressure Jacobi -> Subtract gradient
FS_DIVERGENCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform vec2 texel;
void main(){
    vec2 L = texture(vel, uv - vec2(texel.x,0)).xy;
    vec2 R = texture(vel, uv + vec2(texel.x,0)).xy;
    vec2 B = texture(vel, uv - vec2(0,texel.y)).xy;
    vec2 T = texture(vel, uv + vec2(0,texel.y)).xy;
    float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
    fragColor = vec4(div,0,0,1);
}
"""

FS_JACOBI = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D prs;
uniform sampler2D div;
uniform vec2 texel;
void main(){
    float L = texture(prs, uv - vec2(texel.x,0)).r;
    float R = texture(prs, uv + vec2(texel.x,0)).r;
    float B = texture(prs, uv - vec2(0,texel.y)).r;
    float T = texture(prs, uv + vec2(0,texel.y)).r;
    float b = texture(div, uv).r;
    float p = 0.25 * (L + R + B + T - b);
    fragColor = vec4(p,0,0,1);
}
"""

FS_GRADIENT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform sampler2D prs;
uniform vec2 texel;
void main(){
    float L = texture(prs, uv - vec2(texel.x,0)).r;
    float R = texture(prs, uv + vec2(texel.x,0)).r;
    float B = texture(prs, uv - vec2(0,texel.y)).r;
    float T = texture(prs, uv + vec2(0,texel.y)).r;
    vec2 grad = 0.5 * vec2(R - L, T - B);
    vec2 v = texture(vel, uv).xy - grad;
    fragColor = vec4(v,0,1);
}
"""

# Curl and vorticity confinement
FS_CURL = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform vec2 texel;
void main(){
    vec2 L = texture(vel, uv - vec2(texel.x,0)).xy;
    vec2 R = texture(vel, uv + vec2(texel.x,0)).xy;
    vec2 B = texture(vel, uv - vec2(0,texel.y)).xy;
    vec2 T = texture(vel, uv + vec2(0,texel.y)).xy;
    float curl = 0.5 * ((T.x - B.x) - (R.y - L.y));
    fragColor = vec4(curl,0,0,1);
}
"""

FS_VORTICITY = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform sampler2D curlTex;
uniform vec2 texel;
uniform float eps;
uniform float dt;
void main(){
    float L = texture(curlTex, uv - vec2(texel.x,0)).r;
    float R = texture(curlTex, uv + vec2(texel.x,0)).r;
    float B = texture(curlTex, uv - vec2(0,texel.y)).r;
    float T = texture(curlTex, uv + vec2(0,texel.y)).r;
    float c = texture(curlTex, uv).r;

    vec2 grad = 0.5 * vec2(abs(R) - abs(L), abs(T) - abs(B));
    grad += 1e-5;
    vec2 N = normalize(grad);

    vec2 force = eps * vec2(N.y, -N.x) * c;
    vec2 v = texture(vel, uv).xy + dt * force;
    fragColor = vec4(v,0,1);
}
"""

# NEW: Edge-based velocity force from mask (current & previous mask)
FS_MASK_FORCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel_in;
uniform sampler2D mask_curr;
uniform sampler2D mask_prev;
uniform vec2 texel;          // sim grid texel
uniform float dt;
uniform float edge_thresh;
uniform float amp_normal;    // push along normals
uniform float amp_tangent;   // spin along edge
uniform int   use_temporal;  // 1 to modulate by mask growth rate

void main(){
    float M  = texture(mask_curr, uv).r;
    float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
    float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
    vec2 g = 0.5 * vec2(Mx, My);            // central difference
    float gmag = length(g);

    vec2 v = texture(vel_in, uv).xy;

    if (gmag > edge_thresh){
        vec2 n = g / (gmag + 1e-6);         // outward normal (bg->fg)
        vec2 t = vec2(-n.y, n.x);           // tangent

        float growth = 1.0;
        if (use_temporal == 1){
            float Mp = texture(mask_prev, uv).r;
            growth = (M - Mp) / max(dt, 1e-4);   // positive when mask expands
            growth = clamp(growth * 0.5 + 0.5, 0.0, 1.0); // remap to 0..1
        }

        // force scales with edge strength and optional temporal growth
        vec2 add = amp_normal * n * growth + amp_tangent * t;
        add *= smoothstep(edge_thresh, edge_thresh*3.0, gmag);

        v += dt * add;
    }

    fragColor = vec4(v,0,1);
}
"""

# NEW: Dye deposition along edges so the outline is visible
FS_MASK_DYE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye_in;
uniform sampler2D mask_curr;
uniform vec2 texel;
uniform float edge_thresh;
uniform vec3 edge_color;
uniform float strength;

void main(){
    float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
    float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
    float gmag = length(0.5 * vec2(Mx, My));
    vec3 base = texture(dye_in, uv).rgb;

    float k = smoothstep(edge_thresh, edge_thresh*3.0, gmag);
    vec3 add = edge_color * (strength * k);
    fragColor = vec4(base + add, 1.0);
}
"""

# Display
FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
uniform int palette_on;
vec3 palette(float t){
    t = clamp(t,0.0,1.0);
    if(t<0.5){ float k=t/0.5; return mix(vec3(0.0), vec3(1.0,0.5,0.0), k); }
    float k=(t-0.5)/0.5; return mix(vec3(1.0,0.5,0.0), vec3(0.45,0.0,0.6), k);
}
void main(){
    vec3 c = texture(dye, uv).rgb;
    if(palette_on==0){ fragColor = vec4(c,1.0); }
    else{
        float l = dot(c, vec3(0.299,0.587,0.114));
        l = 1.0 - exp(-2.3*l);
        fragColor = vec4(palette(l),1.0);
    }
}
"""


# ---------------- Utils ----------------
def make_tex(ctx, size, comps, clamp=True):
    tex = ctx.texture(size, comps, dtype="f4")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        tex.repeat_x = False
        tex.repeat_y = False
    return tex


def fullscreen_quad(ctx):
    v = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
    return ctx.buffer(v.tobytes())


# ---------------- Main ----------------
def main():
    # Window
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(
        WIDTH, HEIGHT, "Step 6: Edge-Driven Stable Fluids", None, None
    )
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # MediaPipe
    mp_seg = mp.solutions.selfie_segmentation
    segmenter = mp_seg.SelfieSegmentation(model_selection=0)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Programs
    prog_adv = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_div = ctx.program(vertex_shader=VS, fragment_shader=FS_DIVERGENCE)
    prog_jac = ctx.program(vertex_shader=VS, fragment_shader=FS_JACOBI)
    prog_grad = ctx.program(vertex_shader=VS, fragment_shader=FS_GRADIENT)
    prog_curl = ctx.program(vertex_shader=VS, fragment_shader=FS_CURL)
    prog_vort = ctx.program(vertex_shader=VS, fragment_shader=FS_VORTICITY)
    prog_maskF = ctx.program(vertex_shader=VS, fragment_shader=FS_MASK_FORCE)
    prog_maskD = ctx.program(vertex_shader=VS, fragment_shader=FS_MASK_DYE)
    prog_show = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)

    vbo = fullscreen_quad(ctx)
    vao_adv = ctx.simple_vertex_array(prog_adv, vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_div = ctx.simple_vertex_array(prog_div, vbo, "in_vert")
    vao_jac = ctx.simple_vertex_array(prog_jac, vbo, "in_vert")
    vao_grad = ctx.simple_vertex_array(prog_grad, vbo, "in_vert")
    vao_curl = ctx.simple_vertex_array(prog_curl, vbo, "in_vert")
    vao_vort = ctx.simple_vertex_array(prog_vort, vbo, "in_vert")
    vao_maskF = ctx.simple_vertex_array(prog_maskF, vbo, "in_vert")
    vao_maskD = ctx.simple_vertex_array(prog_maskD, vbo, "in_vert")
    vao_show = ctx.simple_vertex_array(prog_show, vbo, "in_vert")

    # Sizes
    sim_w = max(32, int(WIDTH * SIM_SCALE))
    sim_h = max(32, int(HEIGHT * SIM_SCALE))
    texel = (1.0 / sim_w, 1.0 / sim_h)

    # Textures
    vel_a = make_tex(ctx, (sim_w, sim_h), 2)
    vel_b = make_tex(ctx, (sim_w, sim_h), 2)
    dye_a = make_tex(ctx, (WIDTH, HEIGHT), 4)
    dye_b = make_tex(ctx, (WIDTH, HEIGHT), 4)
    prs = make_tex(ctx, (sim_w, sim_h), 1)
    prs_b = make_tex(ctx, (sim_w, sim_h), 1)
    div = make_tex(ctx, (sim_w, sim_h), 1)
    curl = make_tex(ctx, (sim_w, sim_h), 1)
    # mask textures at sim resolution (clean gradients)
    mask_curr = make_tex(ctx, (sim_w, sim_h), 1)
    mask_prev = make_tex(ctx, (sim_w, sim_h), 1)

    fbo_vel_a = ctx.framebuffer([vel_a])
    fbo_vel_b = ctx.framebuffer([vel_b])
    fbo_dye_a = ctx.framebuffer([dye_a])
    fbo_dye_b = ctx.framebuffer([dye_b])
    fbo_prs = ctx.framebuffer([prs])
    fbo_prs_b = ctx.framebuffer([prs_b])
    fbo_div = ctx.framebuffer([div])
    fbo_curl = ctx.framebuffer([curl])

    for fbo in (
        fbo_vel_a,
        fbo_vel_b,
        fbo_dye_a,
        fbo_dye_b,
        fbo_prs,
        fbo_prs_b,
        fbo_div,
        fbo_curl,
    ):
        fbo.use()
        ctx.clear(0, 0, 0, 1)
    ctx.screen.use()

    # Static uniforms
    for prog in (
        prog_div,
        prog_jac,
        prog_grad,
        prog_curl,
        prog_vort,
        prog_maskF,
        prog_maskD,
    ):
        pass
    prog_div["texel"].value = texel
    prog_jac["texel"].value = texel
    prog_grad["texel"].value = texel
    prog_curl["texel"].value = texel
    prog_vort["texel"].value = texel
    prog_vort["eps"].value = VORTICITY
    prog_maskF["texel"].value = texel
    prog_maskF["edge_thresh"].value = EDGE_THRESH
    prog_maskF["amp_normal"].value = EDGE_NORMAL_AMP
    prog_maskF["amp_tangent"].value = EDGE_TANGENTIAL_AMP
    prog_maskF["use_temporal"].value = 1 if EDGE_USE_TEMPORAL else 0
    prog_maskD["texel"].value = (1.0 / WIDTH, 1.0 / HEIGHT)  # dye is full-res
    prog_maskD["edge_thresh"].value = EDGE_THRESH
    # halloween-ish dye color on edges
    prog_maskD["edge_color"].value = (1.0, 0.45, 0.1)

    # Seed a soft puff
    fbo_dye_b.use()
    dye_a.use(location=0)
    prog_splat["field"].value = 0
    prog_splat["point"].value = (0.5, 0.5)
    prog_splat["value"].value = (0.9, 0.4, 0.9)
    prog_splat["radius"].value = 0.12
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a, dye_b = dye_b, dye_a
    fbo_dye_a, fbo_dye_b = fbo_dye_b, fbo_dye_a

    palette_on = PALETTE_ON
    running = True
    prev = time.time()
    have_mask = False

    print("SPACE pause  C clear  [/] vorticity -/+  P palette")
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            if glfw.get_key(win, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS:
                running = running if False else not running
                time.sleep(0.12)
            if glfw.get_key(win, glfw.KEY_C) == glfw.PRESS:
                for fbo in (
                    fbo_vel_a,
                    fbo_vel_b,
                    fbo_dye_a,
                    fbo_dye_b,
                    fbo_prs,
                    fbo_prs_b,
                    fbo_div,
                    fbo_curl,
                ):
                    fbo.use()
                    ctx.clear(0, 0, 0, 1)
                ctx.screen.use()
                time.sleep(0.12)
            if glfw.get_key(win, glfw.KEY_LEFT_BRACKET) == glfw.PRESS:
                prog_vort["eps"].value = max(0.0, prog_vort["eps"].value - 0.1)
                time.sleep(0.05)
            if glfw.get_key(win, glfw.KEY_RIGHT_BRACKET) == glfw.PRESS:
                prog_vort["eps"].value = min(5.0, prog_vort["eps"].value + 0.1)
                time.sleep(0.05)
            if glfw.get_key(win, glfw.KEY_P) == glfw.PRESS:
                palette_on = 1 - palette_on
                time.sleep(0.12)

            now = time.time()
            dt = min(DT_CLAMP, now - prev)
            prev = now

            # --- Camera & mask upload (to sim resolution) ---
            ret, frame = cap.read()
            have_mask = False
            if ret:
                frame = cv2.flip(frame, 1)  # mirror horizontally like before

                # --- YOLO multi-person mask (sim grid size) ---
                m_small = people_mask_yolo(frame, sim_w, sim_h, conf=0.25)
                mask_area = float(
                    (
                        cv2.resize(
                            m_small, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR
                        )
                        > 0.5
                    ).mean()
                )

                have_mask = mask_area > 0.005  # be forgiving; tune as you like

                if have_mask:
                    # write to GPU textures for the edge-force pass exactly like before
                    # keep a CPU copy as "previous" for temporal effects if you use them
                    try:
                        last_mask_small
                    except NameError:
                        last_mask_small = m_small.copy()

                    mask_prev.write(last_mask_small.tobytes())
                    mask_curr.write(m_small.tobytes())
                    last_mask_small = m_small.copy()

            if running:
                sdt = dt / SUBSTEPS
                for _ in range(SUBSTEPS):
                    # 1) Advect velocity
                    fbo_vel_b.use()
                    vel_a.use(location=0)
                    vel_a.use(location=1)
                    prog_adv["vel"].value = 0
                    prog_adv["src"].value = 1
                    prog_adv["dt"].value = sdt
                    prog_adv["dissipation"].value = VEL_DISSIPATION
                    vao_adv.render(moderngl.TRIANGLE_STRIP)
                    vel_a, vel_b = vel_b, vel_a
                    fbo_vel_a, fbo_vel_b = fbo_vel_b, fbo_vel_a

                    # 2) Edge-based forces from mask (if available)
                    if have_mask:
                        fbo_vel_b.use()
                        vel_a.use(location=0)
                        mask_curr.use(location=1)
                        mask_prev.use(location=2)
                        prog_maskF["vel_in"].value = 0
                        prog_maskF["mask_curr"].value = 1
                        prog_maskF["mask_prev"].value = 2
                        prog_maskF["dt"].value = sdt
                        vao_maskF.render(moderngl.TRIANGLE_STRIP)
                        vel_a, vel_b = vel_b, vel_a
                        fbo_vel_a, fbo_vel_b = fbo_vel_b, fbo_vel_a

                    # 3) Vorticity confinement
                    if VORTICITY > 0.0:
                        fbo_curl.use()
                        vel_a.use(location=0)
                        prog_curl["vel"].value = 0
                        vao_curl.render(moderngl.TRIANGLE_STRIP)

                        fbo_vel_b.use()
                        vel_a.use(location=0)
                        curl.use(location=1)
                        prog_vort["vel"].value = 0
                        prog_vort["curlTex"].value = 1
                        prog_vort["dt"].value = sdt
                        vao_vort.render(moderngl.TRIANGLE_STRIP)
                        vel_a, vel_b = vel_b, vel_a
                        fbo_vel_a, fbo_vel_b = fbo_vel_b, fbo_vel_a

                    # 4) Projection
                    fbo_div.use()
                    vel_a.use(location=0)
                    prog_div["vel"].value = 0
                    vao_div.render(moderngl.TRIANGLE_STRIP)

                    for _j in range(JACOBI_ITERS):
                        fbo_prs_b.use()
                        prs.use(location=0)
                        div.use(location=1)
                        prog_jac["prs"].value = 0
                        prog_jac["div"].value = 1
                        vao_jac.render(moderngl.TRIANGLE_STRIP)
                        prs, prs_b = prs_b, prs
                        fbo_prs, fbo_prs_b = fbo_prs_b, fbo_prs

                    fbo_vel_b.use()
                    vel_a.use(location=0)
                    prs.use(location=1)
                    prog_grad["vel"].value = 0
                    prog_grad["prs"].value = 1
                    vao_grad.render(moderngl.TRIANGLE_STRIP)
                    vel_a, vel_b = vel_b, vel_a
                    fbo_vel_a, fbo_vel_b = fbo_vel_b, fbo_vel_a

                    # 5) Advect dye
                    fbo_dye_b.use()
                    vel_a.use(location=0)
                    dye_a.use(location=1)
                    prog_adv["vel"].value = 0
                    prog_adv["src"].value = 1
                    prog_adv["dt"].value = sdt
                    prog_adv["dissipation"].value = DYE_DISSIPATION
                    vao_adv.render(moderngl.TRIANGLE_STRIP)
                    dye_a, dye_b = dye_b, dye_a
                    fbo_dye_a, fbo_dye_b = fbo_dye_b, fbo_dye_a

                    # 6) (Optional) dye along edges to visualize silhouette
                    if have_mask and EDGE_DYE_STRENGTH > 0.0:
                        fbo_dye_b.use()
                        dye_a.use(location=0)
                        mask_curr.use(location=1)
                        prog_maskD["dye_in"].value = 0
                        prog_maskD["mask_curr"].value = 1
                        prog_maskD["strength"].value = EDGE_DYE_STRENGTH
                        vao_maskD.render(moderngl.TRIANGLE_STRIP)
                        dye_a, dye_b = dye_b, dye_a
                        fbo_dye_a, fbo_dye_b = fbo_dye_b, fbo_dye_a

                    # 7) Fallback ambient source when no mask
                    if USE_FALLBACK_ORBIT and not have_mask:
                        t = now
                        base_u = 0.5 + ORBIT_R * math.cos(t * ORBIT_SPEED)
                        base_v = 0.5 + ORBIT_R * math.sin(t * ORBIT_SPEED)
                        for i in range(MICRO_SPLATS_PER_FRAME):
                            ang = (
                                2
                                * math.pi
                                * (i / MICRO_SPLATS_PER_FRAME + 0.1 * math.sin(t * 0.4))
                            )
                            u = base_u + 0.010 * math.cos(ang)
                            v = base_v + 0.010 * math.sin(ang)
                            # dye only
                            orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
                            purple = np.array([0.45, 0.0, 0.6], dtype=np.float32)
                            k = (
                                i / MICRO_SPLATS_PER_FRAME + 0.2 * math.sin(t * 0.7)
                            ) % 1.0
                            col = (
                                orange * (1.0 - k) + purple * k
                            ) * MICRO_SPLAT_STRENGTH
                            fbo_dye_b.use()
                            dye_a.use(location=0)
                            prog_splat["field"].value = 0
                            prog_splat["point"].value = (float(u), float(v))
                            prog_splat["value"].value = (
                                float(col[0]),
                                float(col[1]),
                                float(col[2]),
                            )
                            prog_splat["radius"].value = MICRO_SPLAT_RADIUS
                            vao_splat.render(moderngl.TRIANGLE_STRIP)
                            dye_a, dye_b = dye_b, dye_a
                            fbo_dye_a, fbo_dye_b = fbo_dye_b, fbo_dye_a

            # Present
            ctx.screen.use()
            dye_a.use(location=0)
            prog_show["dye"].value = 0
            prog_show["palette_on"].value = palette_on
            vao_show.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(win)

    finally:
        cap.release()
        segmenter.close()
        glfw.terminate()


if __name__ == "__main__":
    main()
