# single_file_fluid_mask.py
# Real-time 2D Stable Fluids (Stam) driven by MediaPipe selfie segmentation mask motion.
# One-file script: ModernGL shaders are embedded as strings.
# pip install moderngl glfw opencv-python mediapipe numpy

import glfw
import moderngl
import numpy as np
import time
import cv2
import mediapipe as mp

# ---------------------
# Config
# ---------------------
WIDTH, HEIGHT = 1280, 720          # window size
SIM_SCALE = 1.0                     # 1.0 = full res, 0.5 = half res (much faster)
JACOBI_ITERS = 30                   # 20–60 is typical
DYE_DISSIPATION = 0.998             # 0.995–0.999
VEL_DISSIPATION = 1.0               # usually 1.0 (no decay)
SPLAT_RADIUS = 0.03                 # in UV space
FORCE_GAIN = 220.0                  # scales optical-flow driven force
DYE_GAIN = 0.8                      # dye intensity for the mask
FLOW_DOWNSAMPLE = 4                 # compute optical flow on smaller grid for speed
MASK_EDGE_THRESH = (50, 150)        # Canny thresholds to splat along edges
TARGET_FPS = 60

# ---------------------
# Utility: create texture/FBO
# ---------------------
def make_tex(ctx, size, components, dtype='f2', filter=(moderngl.LINEAR, moderngl.LINEAR), repeat=False):
    tex = ctx.texture(size, components, dtype=dtype)
    tex.filter = filter
    tex.repeat_x = repeat
    tex.repeat_y = repeat
    return tex

# ---------------------
# Shaders (GLSL 330)
# ---------------------
VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){
    gl_Position = vec4(in_vert,0.0,1.0);
    uv = (in_vert + 1.0) * 0.5;
}
"""

# Advect: semi-Lagrangian
FS_ADVECT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;     // velocity field
uniform sampler2D src;     // field to advect (vel or dye)
uniform float dt;
uniform float dissipation;
uniform vec2 invSize;      // 1.0 / simSize
void main(){
    vec2 v = texture(vel, uv).xy;
    // backtrace in UV space; v is in "pixels per second" if you like—
    // multiplying by invSize keeps behavior roughly res-invariant
    vec2 prev = uv - dt * v * invSize;
    vec4 val = texture(src, prev);
    fragColor = val * dissipation;
}
"""

# Splat: add value around 'point' with Gaussian falloff
FS_SPLAT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec2 point;
uniform vec3 value;     // xy0 for velocity, rgb for dye
uniform float radius;   // in UV
void main(){
    vec2 p = uv - point;
    float falloff = exp(-dot(p,p)/(radius*radius));
    vec4 base = texture(field, uv);
    fragColor = base + vec4(value * falloff, 0.0);
}
"""

# Divergence of velocity
FS_DIVERGENCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform vec2 texel; // (1/width, 1/height)
void main(){
    vec2 L = texture(vel, uv - vec2(texel.x,0)).xy;
    vec2 R = texture(vel, uv + vec2(texel.x,0)).xy;
    vec2 B = texture(vel, uv - vec2(0,texel.y)).xy;
    vec2 T = texture(vel, uv + vec2(0,texel.y)).xy;
    float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
    fragColor = vec4(div,0,0,1);
}
"""

# Jacobi pressure solve
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

# Subtract gradient: v := v - grad(p)
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

# Display dye (optionally composite background)
FS_DISPLAY = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
void main(){
    fragColor = texture(dye, uv);
}
"""

# Compose a spooky palette on the CPU and splat as dye when mask is present.
def halloween_palette(bg_val, person_val):
    # background: black -> orange
    bg = np.array([0.0, 0.0, 0.0]) * (1.0-bg_val) + np.array([1.0, 0.5, 0.0]) * bg_val
    # person: green -> purple
    pr = np.array([0.1, 0.7, 0.2]) * (1.0-person_val) + np.array([0.4, 0.0, 0.5]) * person_val
    return bg, pr

# ---------------------
# Fullscreen quad
# ---------------------
def make_fullscreen(ctx):
    verts = np.array([-1.0,-1.0,  1.0,-1.0,  -1.0,1.0,  1.0,1.0], dtype='f4')
    vbo = ctx.buffer(verts)
    return vbo

# ---------------------
# Main
# ---------------------
def main():
    # Camera + MediaPipe
    print("Initializing camera and MediaPipe...")
    mp_selfie_seg = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_seg.SelfieSegmentation(model_selection=0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Window + GL
    print("Initializing ModernGL...")
    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    window = glfw.create_window(WIDTH, HEIGHT, "Ghost Fluid (Mask-Driven)", None, None)
    if not window:
        glfw.terminate(); raise RuntimeError("Window creation failed")
    glfw.make_context_current(window)
    ctx = moderngl.create_context()

    # Compile shader programs
    prog_advect   = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat    = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_div      = ctx.program(vertex_shader=VS, fragment_shader=FS_DIVERGENCE)
    prog_jacobi   = ctx.program(vertex_shader=VS, fragment_shader=FS_JACOBI)
    prog_gradient = ctx.program(vertex_shader=VS, fragment_shader=FS_GRADIENT)
    prog_display  = ctx.program(vertex_shader=VS, fragment_shader=FS_DISPLAY)

    # Geometry
    vbo = make_fullscreen(ctx)
    vao_advect   = ctx.simple_vertex_array(prog_advect, vbo, "in_vert")
    vao_splat    = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_div      = ctx.simple_vertex_array(prog_div, vbo, "in_vert")
    vao_jacobi   = ctx.simple_vertex_array(prog_jacobi, vbo, "in_vert")
    vao_gradient = ctx.simple_vertex_array(prog_gradient, vbo, "in_vert")
    vao_display  = ctx.simple_vertex_array(prog_display, vbo, "in_vert")

    # Simulation sizes
    sim_w = max(32, int(WIDTH * SIM_SCALE))
    sim_h = max(32, int(HEIGHT * SIM_SCALE))
    invSize = (1.0 / sim_w, 1.0 / sim_h)
    texel = (1.0 / sim_w, 1.0 / sim_h)

    # Fields
    vel_a = make_tex(ctx, (sim_w, sim_h), 2, dtype='f2')
    vel_b = make_tex(ctx, (sim_w, sim_h), 2, dtype='f2')
    dye_a = make_tex(ctx, (sim_w, sim_h), 4, dtype='f1')  # RGBA8
    dye_b = make_tex(ctx, (sim_w, sim_h), 4, dtype='f1')
    prs   = make_tex(ctx, (sim_w, sim_h), 1, dtype='f2')
    prs_b = make_tex(ctx, (sim_w, sim_h), 1, dtype='f2')
    div   = make_tex(ctx, (sim_w, sim_h), 1, dtype='f2')

    # FBOs
    fbo_vel_a = ctx.framebuffer(color_attachments=[vel_a])
    fbo_vel_b = ctx.framebuffer(color_attachments=[vel_b])
    fbo_dye_a = ctx.framebuffer(color_attachments=[dye_a])
    fbo_dye_b = ctx.framebuffer(color_attachments=[dye_b])
    fbo_prs   = ctx.framebuffer(color_attachments=[prs])
    fbo_prs_b = ctx.framebuffer(color_attachments=[prs_b])
    fbo_div   = ctx.framebuffer(color_attachments=[div])

    # Clear fields
    for fbo in (fbo_vel_a, fbo_vel_b, fbo_dye_a, fbo_dye_b, fbo_prs, fbo_prs_b, fbo_div):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # Timing
    start = time.time()
    prev_time = start
    dt = 1.0 / TARGET_FPS

    # Optical flow cache
    prev_mask_small = None

    print("Running main loop. Wave in front of the camera. Close window to quit.")
    while not glfw.window_should_close(window):
        frame_ok, frame = cap.read()
        if not frame_ok:
            glfw.poll_events(); continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Segment
        res = segmenter.process(rgb)
        mask = res.segmentation_mask
        if mask is None:
            glfw.poll_events(); continue

        # Prepare masks
        mask = cv2.resize(mask, (WIDTH, HEIGHT))
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        # Downsample for optical flow
        m_small = cv2.resize(mask, (WIDTH//FLOW_DOWNSAMPLE, HEIGHT//FLOW_DOWNSAMPLE))
        if prev_mask_small is None:
            prev_mask_small = m_small.copy()

        # Optical flow on mask (scalar to scalar flow)
        flow = cv2.calcOpticalFlowFarneback(
            prev_mask_small, m_small, None,
            pyr_scale=0.5, levels=2, winsize=9,
            iterations=2, poly_n=5, poly_sigma=1.2, flags=0
        )
        prev_mask_small = m_small

        # Choose splat points along edges of current mask (in small space)
        small_u8 = (m_small * 255).astype(np.uint8)
        edges = cv2.Canny(small_u8, MASK_EDGE_THRESH[0], MASK_EDGE_THRESH[1]) > 0
        mag = np.linalg.norm(flow, axis=2)
        ys, xs = np.where((edges) & (mag > 0.2))
        # Throttle number of splats
        max_splats = 250
        if len(xs) > max_splats:
            idx = np.linspace(0, len(xs)-1, max_splats).astype(np.int32)
            xs = xs[idx]; ys = ys[idx]

        # Advect velocity
        fbo_vel_b.use()
        vel_a.use(location=0)  # vel sampler
        vel_a.use(location=1)  # src sampler (advect self)
        prog_advect['vel'].value = 0
        prog_advect['src'].value = 1
        prog_advect['dt'].value = dt
        prog_advect['dissipation'].value = VEL_DISSIPATION
        prog_advect['invSize'].value = invSize
        vao_advect.render(moderngl.TRIANGLE_STRIP)
        vel_a, vel_b = vel_b, vel_a

        # Add forces from mask motion (splat into velocity)
        for x, y in zip(xs, ys):
            u = (x + 0.5) / m_small.shape[1]
            v = (y + 0.5) / m_small.shape[0]
            fx, fy = flow[y, x]
            # scale to UV direction; flip Y (GL's up is +y in UV)
            fx /= m_small.shape[1]; fy /= m_small.shape[0]
            fbo_vel_b.use()
            vel_a.use(location=0)
            prog_splat['field'].value = 0
            prog_splat['point'].value = (u, 1.0 - v)
            prog_splat['value'].value = (fx * FORCE_GAIN, -fy * FORCE_GAIN, 0.0)
            prog_splat['radius'].value = SPLAT_RADIUS
            vao_splat.render(moderngl.TRIANGLE_STRIP)
            vel_a, vel_b = vel_b, vel_a

        # Compute divergence
        fbo_div.use()
        vel_a.use(location=0)
        prog_div['vel'].value = 0
        prog_div['texel'].value = (texel[0], texel[1])
        vao_div.render(moderngl.TRIANGLE_STRIP)

        # Jacobi pressure solve
        # zero prs on the first few frames is fine; iterative updates converge
        for _ in range(JACOBI_ITERS):
            fbo_prs_b.use()
            prs.use(location=0)
            div.use(location=1)
            prog_jacobi['prs'].value = 0
            prog_jacobi['div'].value = 1
            prog_jacobi['texel'].value = (texel[0], texel[1])
            vao_jacobi.render(moderngl.TRIANGLE_STRIP)
            prs, prs_b = prs_b, prs

        # Subtract gradient from velocity
        fbo_vel_b.use()
        vel_a.use(location=0)
        prs.use(location=1)
        prog_gradient['vel'].value = 0
        prog_gradient['prs'].value = 1
        prog_gradient['texel'].value = (texel[0], texel[1])
        vao_gradient.render(moderngl.TRIANGLE_STRIP)
        vel_a, vel_b = vel_b, vel_a

        # Advect dye
        fbo_dye_b.use()
        vel_a.use(location=0)
        dye_a.use(location=1)
        prog_advect['vel'].value = 0
        prog_advect['src'].value = 1
        prog_advect['dt'].value = dt
        prog_advect['dissipation'].value = DYE_DISSIPATION
        prog_advect['invSize'].value = invSize
        vao_advect.render(moderngl.TRIANGLE_STRIP)
        dye_a, dye_b = dye_b, dye_a

        # Also splat dye where mask is strong (gives the “ghost fog”)
        # Sample a sparse grid of mask points to color
        stride = 16
        h_idx = range(0, HEIGHT, stride)
        w_idx = range(0, WIDTH, stride)
        for yy in h_idx:
            for xx in w_idx:
                m = mask[yy, xx]
                if m < 0.5: 
                    continue
                # uv in window space -> sim space
                u = (xx + 0.5) / WIDTH
                v = 1.0 - ((yy + 0.5) / HEIGHT)
                # Pick a person color intensity
                person_val = 0.6 + 0.4*np.random.rand()
                _, person_col = halloween_palette(0.0, person_val)
                rgb = (person_col * DYE_GAIN).astype(np.float32)
                fbo_dye_b.use()
                dye_a.use(location=0)
                prog_splat['field'].value = 0
                prog_splat['point'].value = (u, v)
                prog_splat['value'].value = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
                prog_splat['radius'].value = SPLAT_RADIUS * 0.75
                vao_splat.render(moderngl.TRIANGLE_STRIP)
                dye_a, dye_b = dye_b, dye_a

        # Present dye to screen
        ctx.screen.use()
        dye_a.use(location=0)
        prog_display['dye'].value = 0
        vao_display.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(window)
        glfw.poll_events()

        # simple fixed timestep feel (optional)
        now = time.time()
        sleep = (1.0/TARGET_FPS) - (now - prev_time)
        if sleep > 0:
            time.sleep(sleep)
        prev_time = now

    # Cleanup
    cap.release()
    segmenter.close()
    glfw.terminate()

if __name__ == "__main__":
    main()
