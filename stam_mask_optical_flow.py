# step7b_mask_flow_fluids_debug.py
# Stable fluids driven by optical flow of the MediaPipe mask region.
# This version relaxes thresholds and adds a flow debug overlay (press F).

import glfw, moderngl, numpy as np, time, math, cv2
import mediapipe as mp

# ---------------- Config ----------------
WIDTH, HEIGHT = 1024, 576
SIM_SCALE = 0.7                 # sim grid = SIM_SCALE * window
SUBSTEPS = 3
DT_CLAMP = 0.033
JACOBI_ITERS = 28
VEL_DISSIPATION = 0.999
DYE_DISSIPATION = 0.999
VORTICITY = 2.4
PALETTE_ON = 1

# Camera / segmentation (more forgiving)
CAMERA_INDEX = 0
MASK_THRESHOLD = 0.20           # was 0.35
MASK_MIN_AREA = 0.005           # was 0.02

# Optical flow params (relaxed + stronger)
FLOW_AMP = 10.0                 # was 5.0 (adjust live with -/=)
FLOW_EDGE_WEIGHT = False        # start off; toggle with E if you want
FLOW_MIN = 0.0005               # was 0.0015
FLOW_CLAMP = 0.5
FLOW_SMOOTH_SIGMA = 1.5

# Fallback ambient source when mask missing
USE_FALLBACK_ORBIT = True
ORBIT_R = 0.18
ORBIT_SPEED = 0.35
MICRO_SPLATS_PER_FRAME = 3
MICRO_SPLAT_RADIUS = 0.018
MICRO_SPLAT_STRENGTH = 0.05

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

# Add optical-flow force: v += dt * amp * flow(uv)
FS_FLOW_FORCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel_in;
uniform sampler2D flowTex;   // RG = UV/sec flow
uniform float dt;
uniform float amp;
void main(){
    vec2 v = texture(vel_in, uv).xy;
    vec2 f = texture(flowTex, uv).xy;
    v += dt * amp * f;
    fragColor = vec4(v,0,1);
}
"""

# Splat for dye seeding (fallback ambient only)
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

# Display dye
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

# Debug: view flow magnitude (grayscale)
FS_SHOW_FLOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D flowTex;
void main(){
    vec2 f = texture(flowTex, uv).xy;
    float m = clamp(length(f) * 2.0, 0.0, 1.0); // scale for visibility
    fragColor = vec4(vec3(m), 1.0);
}
"""

# ---------------- Utils ----------------
def make_tex(ctx, size, comps, clamp=True):
    tex = ctx.texture(size, comps, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        tex.repeat_x = False; tex.repeat_y = False
    return tex

def fullscreen_quad(ctx):
    v = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')
    return ctx.buffer(v.tobytes())

# ---------------- Main ----------------
def main():
    # Window
    if not glfw.init(): raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Step 7b: Optical-Flow Fluids (Debug)", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # MediaPipe
    mp_seg = mp.solutions.selfie_segmentation
    segmenter = mp_seg.SelfieSegmentation(model_selection=0)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Programs
    prog_adv   = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_div   = ctx.program(vertex_shader=VS, fragment_shader=FS_DIVERGENCE)
    prog_jac   = ctx.program(vertex_shader=VS, fragment_shader=FS_JACOBI)
    prog_grad  = ctx.program(vertex_shader=VS, fragment_shader=FS_GRADIENT)
    prog_curl  = ctx.program(vertex_shader=VS, fragment_shader=FS_CURL)
    prog_vort  = ctx.program(vertex_shader=VS, fragment_shader=FS_VORTICITY)
    prog_flowF = ctx.program(vertex_shader=VS, fragment_shader=FS_FLOW_FORCE)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_show  = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)
    prog_show_flow = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW_FLOW)

    vbo = fullscreen_quad(ctx)
    vao_adv   = ctx.simple_vertex_array(prog_adv,   vbo, "in_vert")
    vao_div   = ctx.simple_vertex_array(prog_div,   vbo, "in_vert")
    vao_jac   = ctx.simple_vertex_array(prog_jac,   vbo, "in_vert")
    vao_grad  = ctx.simple_vertex_array(prog_grad,  vbo, "in_vert")
    vao_curl  = ctx.simple_vertex_array(prog_curl,  vbo, "in_vert")
    vao_vort  = ctx.simple_vertex_array(prog_vort,  vbo, "in_vert")
    vao_flowF = ctx.simple_vertex_array(prog_flowF, vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_show  = ctx.simple_vertex_array(prog_show,  vbo, "in_vert")
    vao_show_flow = ctx.simple_vertex_array(prog_show_flow, vbo, "in_vert")

    # Sizes
    sim_w = max(32, int(WIDTH * SIM_SCALE))
    sim_h = max(32, int(HEIGHT * SIM_SCALE))
    texel = (1.0/sim_w, 1.0/sim_h)

    # Textures
    vel_a = make_tex(ctx, (sim_w,sim_h), 2)
    vel_b = make_tex(ctx, (sim_w,sim_h), 2)
    dye_a = make_tex(ctx, (WIDTH,HEIGHT), 4)
    dye_b = make_tex(ctx, (WIDTH,HEIGHT), 4)
    prs   = make_tex(ctx, (sim_w,sim_h), 1)
    prs_b = make_tex(ctx, (sim_w,sim_h), 1)
    div   = make_tex(ctx, (sim_w,sim_h), 1)
    curl  = make_tex(ctx, (sim_w,sim_h), 1)
    flow_tex = make_tex(ctx, (sim_w,sim_h), 2)  # RG = UV/sec

    fbo_vel_a = ctx.framebuffer([vel_a])
    fbo_vel_b = ctx.framebuffer([vel_b])
    fbo_dye_a = ctx.framebuffer([dye_a])
    fbo_dye_b = ctx.framebuffer([dye_b])
    fbo_prs   = ctx.framebuffer([prs])
    fbo_prs_b = ctx.framebuffer([prs_b])
    fbo_div   = ctx.framebuffer([div])
    fbo_curl  = ctx.framebuffer([curl])

    for fbo in (fbo_vel_a,fbo_vel_b,fbo_dye_a,fbo_dye_b,fbo_prs,fbo_prs_b,fbo_div,fbo_curl):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # Static uniforms
    prog_div['texel'].value = texel
    prog_jac['texel'].value = texel
    prog_grad['texel'].value = texel
    prog_curl['texel'].value = texel
    prog_vort['texel'].value = texel
    prog_vort['eps'].value = VORTICITY

    # Seed a soft puff
    fbo_dye_b.use()
    dye_a.use(location=0)
    prog_splat['field'].value = 0
    prog_splat['point'].value = (0.5,0.5)
    prog_splat['value'].value = (0.9,0.4,0.9)
    prog_splat['radius'].value = 0.12
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a,dye_b = dye_b,dye_a
    fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

    # Flow CPU state
    have_prev = False
    prev_gray_small = None
    flow_amp = FLOW_AMP
    palette_on = PALETTE_ON
    debug_flow_view = False
    running = True
    prev = time.time()
    frame_i = 0

    print("SPACE pause  C clear  [/] vorticity -/+  P palette  -/= flow amp  F flow debug  E edge-weight toggle")
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            if glfw.get_key(win,glfw.KEY_ESCAPE)==glfw.PRESS: break
            if glfw.get_key(win,glfw.KEY_SPACE)==glfw.PRESS:
                running = not running; time.sleep(0.12)
            if glfw.get_key(win,glfw.KEY_C)==glfw.PRESS:
                for fbo in (fbo_vel_a,fbo_vel_b,fbo_dye_a,fbo_dye_b,fbo_prs,fbo_prs_b,fbo_div,fbo_curl):
                    fbo.use(); ctx.clear(0,0,0,1)
                ctx.screen.use(); time.sleep(0.12)
            if glfw.get_key(win,glfw.KEY_LEFT_BRACKET)==glfw.PRESS:
                prog_vort['eps'].value = max(0.0, prog_vort['eps'].value - 0.1); time.sleep(0.05)
            if glfw.get_key(win,glfw.KEY_RIGHT_BRACKET)==glfw.PRESS:
                prog_vort['eps'].value = min(5.0, prog_vort['eps'].value + 0.1); time.sleep(0.05)
            if glfw.get_key(win,glfw.KEY_MINUS)==glfw.PRESS:
                flow_amp = max(0.0, flow_amp - 0.5); time.sleep(0.05)
            if glfw.get_key(win,glfw.KEY_EQUAL)==glfw.PRESS:
                flow_amp = min(30.0, flow_amp + 0.5); time.sleep(0.05)
            if glfw.get_key(win,glfw.KEY_P)==glfw.PRESS:
                palette_on = 1 - palette_on; time.sleep(0.12)
            if glfw.get_key(win,glfw.KEY_F)==glfw.PRESS:
                debug_flow_view = not debug_flow_view; time.sleep(0.12)
            if glfw.get_key(win,glfw.KEY_E)==glfw.PRESS:
                # toggle edge weight live
                global FLOW_EDGE_WEIGHT
                FLOW_EDGE_WEIGHT = not FLOW_EDGE_WEIGHT
                print("FLOW_EDGE_WEIGHT =", FLOW_EDGE_WEIGHT); time.sleep(0.12)

            now = time.time()
            dt = min(DT_CLAMP, now - prev)
            prev = now

            # --- Camera & segmentation ---
            ret, frame = cap.read()
            have_mask = False
            mask_area = 0.0
            if ret:
                frame = cv2.flip(frame, 1)  # mirror horizontally
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = segmenter.process(rgb)
                mask = res.segmentation_mask
                if mask is not None:
                    # window-size area check
                    m_big = cv2.resize(mask.astype(np.float32), (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
                    mask_area = (m_big > MASK_THRESHOLD).sum() / float(WIDTH*HEIGHT)
                    have_mask = mask_area > MASK_MIN_AREA

                    # sim-size, vertically flipped (so UV up matches image down)
                    m_small = cv2.flip(
                        cv2.resize(mask.astype(np.float32), (sim_w, sim_h), interpolation=cv2.INTER_LINEAR),
                        0
                    )

                    # sim-size grayscale (also flipped)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
                    gray_small = cv2.flip(
                        cv2.resize(gray, (sim_w, sim_h), interpolation=cv2.INTER_LINEAR),
                        0
                    )
                    # soft mask to focus on subject
                    gray_small *= cv2.GaussianBlur(m_small, (0,0), 1.0)

                    # Optical flow
                    if have_prev:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_gray_small, gray_small,
                            None, 0.5, 3, 15, 2, 5, 1.2, 0
                        )
                        fu = flow[:,:,0] / float(sim_w) / max(dt,1e-5)
                        fv = -flow[:,:,1] / float(sim_h) / max(dt,1e-5)  # invert y (already flipped)

                        if FLOW_SMOOTH_SIGMA > 0.0:
                            fu = cv2.GaussianBlur(fu, (0,0), FLOW_SMOOTH_SIGMA)
                            fv = cv2.GaussianBlur(fv, (0,0), FLOW_SMOOTH_SIGMA)

                        if FLOW_EDGE_WEIGHT:
                            gx = 0.5*(np.roll(m_small,-1,axis=1) - np.roll(m_small,1,axis=1))
                            gy = 0.5*(np.roll(m_small,-1,axis=0) - np.roll(m_small,1,axis=0))
                            gmag = np.sqrt(gx*gx + gy*gy)
                            w = np.clip((gmag - 0.005) / 0.02, 0.0, 1.0)  # softer thresholds
                            fu *= w; fv *= w
                        else:
                            fu *= m_small; fv *= m_small

                        mag = np.sqrt(fu*fu + fv*fv)
                        fu = np.where(mag < FLOW_MIN, 0.0, np.clip(fu, -FLOW_CLAMP, FLOW_CLAMP))
                        fv = np.where(mag < FLOW_MIN, 0.0, np.clip(fv, -FLOW_CLAMP, FLOW_CLAMP))

                        flow_rg = np.dstack([fu, fv]).astype(np.float32, copy=True)
                        flow_tex.write(flow_rg.tobytes())

                        if frame_i % 30 == 0:
                            print(f"[flow] mask_area={mask_area:.3f}  max_mag={mag.max():.4f}  amp={flow_amp:.1f}")
                    prev_gray_small = gray_small.copy()
                    have_prev = True
                else:
                    have_prev = False

            # --- Simulation ---
            if running:
                sdt = dt / SUBSTEPS
                for _ in range(SUBSTEPS):
                    # 1) Advect velocity
                    fbo_vel_b.use()
                    vel_a.use(location=0); vel_a.use(location=1)
                    prog_adv['vel'].value = 0
                    prog_adv['src'].value = 1
                    prog_adv['dt'].value = sdt
                    prog_adv['dissipation'].value = VEL_DISSIPATION
                    vao_adv.render(moderngl.TRIANGLE_STRIP)
                    vel_a,vel_b = vel_b,vel_a
                    fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                    # 2) Add optical-flow force if available
                    if have_prev and have_mask:
                        fbo_vel_b.use()
                        vel_a.use(location=0); flow_tex.use(location=1)
                        prog_flowF['vel_in'].value = 0
                        prog_flowF['flowTex'].value = 1
                        prog_flowF['dt'].value = sdt
                        prog_flowF['amp'].value = flow_amp
                        vao_flowF.render(moderngl.TRIANGLE_STRIP)
                        vel_a,vel_b = vel_b,vel_a
                        fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                    # 3) Vorticity confinement
                    if VORTICITY > 0.0:
                        fbo_curl.use()
                        vel_a.use(location=0)
                        prog_curl['vel'].value = 0
                        vao_curl.render(moderngl.TRIANGLE_STRIP)

                        fbo_vel_b.use()
                        vel_a.use(location=0); curl.use(location=1)
                        prog_vort['vel'].value = 0
                        prog_vort['curlTex'].value = 1
                        prog_vort['dt'].value = sdt
                        vao_vort.render(moderngl.TRIANGLE_STRIP)
                        vel_a,vel_b = vel_b,vel_a
                        fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                    # 4) Projection
                    fbo_div.use()
                    vel_a.use(location=0)
                    prog_div['vel'].value = 0
                    vao_div.render(moderngl.TRIANGLE_STRIP)

                    for _j in range(JACOBI_ITERS):
                        fbo_prs_b.use()
                        prs.use(location=0); div.use(location=1)
                        prog_jac['prs'].value = 0
                        prog_jac['div'].value = 1
                        vao_jac.render(moderngl.TRIANGLE_STRIP)
                        prs,prs_b = prs_b,prs
                        fbo_prs,fbo_prs_b = fbo_prs_b,fbo_prs

                    fbo_vel_b.use()
                    vel_a.use(location=0); prs.use(location=1)
                    prog_grad['vel'].value = 0
                    prog_grad['prs'].value = 1
                    vao_grad.render(moderngl.TRIANGLE_STRIP)
                    vel_a,vel_b = vel_b,vel_a
                    fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                    # 5) Advect dye
                    fbo_dye_b.use()
                    vel_a.use(location=0); dye_a.use(location=1)
                    prog_adv['vel'].value = 0
                    prog_adv['src'].value = 1
                    prog_adv['dt'].value = sdt
                    prog_adv['dissipation'].value = DYE_DISSIPATION
                    vao_adv.render(moderngl.TRIANGLE_STRIP)
                    dye_a,dye_b = dye_b,dye_a
                    fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

                    # 6) Fallback ambient dye when no mask
                    if USE_FALLBACK_ORBIT and not (have_prev and have_mask):
                        t = now
                        base_u = 0.5 + ORBIT_R * math.cos(t * ORBIT_SPEED)
                        base_v = 0.5 + ORBIT_R * math.sin(t * ORBIT_SPEED)
                        for i in range(MICRO_SPLATS_PER_FRAME):
                            ang = 2*math.pi * (i / MICRO_SPLATS_PER_FRAME + 0.1 * math.sin(t*0.4))
                            u = base_u + 0.010 * math.cos(ang)
                            v = base_v + 0.010 * math.sin(ang)
                            # dye only
                            orange = np.array([1.0, 0.5, 0.0], dtype=np.float32)
                            purple = np.array([0.45, 0.0, 0.6], dtype=np.float32)
                            k = (i / MICRO_SPLATS_PER_FRAME + 0.2 * math.sin(t*0.7)) % 1.0
                            col = (orange * (1.0 - k) + purple * k) * MICRO_SPLAT_STRENGTH
                            fbo_dye_b.use()
                            dye_a.use(location=0)
                            prog_splat['field'].value = 0
                            prog_splat['point'].value = (float(u), float(v))
                            prog_splat['value'].value = (float(col[0]), float(col[1]), float(col[2]))
                            prog_splat['radius'].value = MICRO_SPLAT_RADIUS
                            vao_splat.render(moderngl.TRIANGLE_STRIP)
                            dye_a,dye_b = dye_b,dye_a
                            fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

            # Present
            ctx.screen.use()
            if debug_flow_view:
                flow_tex.use(location=0)
                prog_show_flow['flowTex'].value = 0
                vao_show_flow.render(moderngl.TRIANGLE_STRIP)
            else:
                dye_a.use(location=0)
                prog_show['dye'].value = 0
                prog_show['palette_on'].value = palette_on
                vao_show.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(win)

            frame_i += 1

    finally:
        cap.release()
        segmenter.close()
        glfw.terminate()

if __name__ == "__main__":
    main()
