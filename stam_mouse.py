# step4_mouse_stable_fluids_min.py
# Minimal "real" stable fluids:
# - velocity field (RG float) + dye
# - mouse injects velocity (throw) and dye (paint)
# - advect vel -> vorticity -> divergence -> Jacobi pressure -> subtract grad
# - advect dye with projected vel
# Keys: SPACE pause, C clear, [/] vorticity -, +, P palette toggle

import glfw, moderngl, numpy as np, time, math

# ------------- Config -------------
WIDTH, HEIGHT = 1024, 576      # window
MOUSE_VEL_SCALE = 0.8
MOUSE_RADIUS    = 0.01
VORTICITY       = 3.0
VEL_DISSIPATION = 0.9995
DYE_DISSIPATION = 0.99
SUBSTEPS        = 6
JACOBI_ITERS    = 28
SIM_SCALE       = 0.7
DT_CLAMP       = 0.033
PALETTE_ON = 1                 # 1 = orange→purple palette

# ------------- Shaders -------------
VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){
  gl_Position = vec4(in_vert,0.0,1.0);
  uv = (in_vert + 1.0)*0.5;
}
"""

# advect any field (src=vel or dye) using velocity "vel"
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

# add a Gaussian "splat" to a field (vec2 to velocity, vec3 to dye)
FS_SPLAT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec2 point;      // UV
uniform vec3 value;      // xy0 for velocity, rgb for dye
uniform float radius;    // UV
void main(){
  vec4 base = texture(field, uv);
  vec2 d = uv - point;
  float fall = exp(-dot(d,d)/(radius*radius));
  fragColor = base + vec4(value * fall, 0.0);
}
"""

# compute divergence of velocity
FS_DIVERGENCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform vec2 texel;  // 1/W,1/H (sim grid)
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
uniform sampler2D prs;   // previous pressure
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

# subtract pressure gradient from velocity
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

# curl (scalar omega_z)
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

# vorticity confinement: F = eps * (N x omega_z)
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
  grad += 1e-5; // avoid zero
  vec2 N = normalize(grad);

  vec2 force = eps * vec2(N.y, -N.x) * c;
  vec2 v = texture(vel, uv).xy + dt * force;
  fragColor = vec4(v,0,1);
}
"""

# simple blur (optional if you want)
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

# palette visualization (orange->purple)
FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
uniform int palette_on;
vec3 palette(float t){
  t = clamp(t,0.0,1.0);
  if(t < 0.5){
    float k = t/0.5;
    return mix(vec3(0.0), vec3(1.0,0.5,0.0), k);
  }else{
    float k = (t-0.5)/0.5;
    return mix(vec3(1.0,0.5,0.0), vec3(0.45,0.0,0.6), k);
  }
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

# ------------- Utils -------------
def make_tex(ctx, size, comps, clamp=True):
    tex = ctx.texture(size, comps, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    if clamp:
        tex.repeat_x = False; tex.repeat_y = False
    return tex

def fullscreen_quad(ctx):
    v = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')
    return ctx.buffer(v.tobytes())

class Mouse:
    def __init__(self,w,h):
        self.w=w; self.h=h
        self.x=w*0.5; self.y=h*0.5
        self.px=self.x; self.py=self.y
        self.down=False
    def set_pos(self,x,y):
        self.px,self.py = self.x,self.y
        self.x,self.y = x,y
    def uv(self):
        return (np.clip(self.x/self.w,0,1), np.clip(1.0-self.y/self.h,0,1))
    def puv(self):
        return (np.clip(self.px/self.w,0,1), np.clip(1.0-self.py/self.h,0,1))

# ------------- Main -------------
def main():
    # Window / GL
    if not glfw.init(): raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR,3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR,3)
    glfw.window_hint(glfw.OPENGL_PROFILE,glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT,True)
    win = glfw.create_window(WIDTH, HEIGHT, "Step 4: Mouse Stable Fluids (min)", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()

    # Mouse
    mouse = Mouse(WIDTH, HEIGHT)
    def on_cursor(_,x,y): mouse.set_pos(x,y)
    def on_button(_,btn,action,mods):
        if btn==glfw.MOUSE_BUTTON_LEFT: mouse.down = (action==glfw.PRESS)
    glfw.set_cursor_pos_callback(win,on_cursor)
    glfw.set_mouse_button_callback(win,on_button)

    # Programs & VAOs
    prog_adv   = ctx.program(vertex_shader=VS, fragment_shader=FS_ADVECT)
    prog_splat = ctx.program(vertex_shader=VS, fragment_shader=FS_SPLAT)
    prog_div   = ctx.program(vertex_shader=VS, fragment_shader=FS_DIVERGENCE)
    prog_jac   = ctx.program(vertex_shader=VS, fragment_shader=FS_JACOBI)
    prog_grad  = ctx.program(vertex_shader=VS, fragment_shader=FS_GRADIENT)
    prog_curl  = ctx.program(vertex_shader=VS, fragment_shader=FS_CURL)
    prog_vort  = ctx.program(vertex_shader=VS, fragment_shader=FS_VORTICITY)
    prog_blur  = ctx.program(vertex_shader=VS, fragment_shader=FS_BLUR)
    prog_show  = ctx.program(vertex_shader=VS, fragment_shader=FS_SHOW)

    vbo = fullscreen_quad(ctx)
    vao_adv   = ctx.simple_vertex_array(prog_adv,   vbo, "in_vert")
    vao_splat = ctx.simple_vertex_array(prog_splat, vbo, "in_vert")
    vao_div   = ctx.simple_vertex_array(prog_div,   vbo, "in_vert")
    vao_jac   = ctx.simple_vertex_array(prog_jac,   vbo, "in_vert")
    vao_grad  = ctx.simple_vertex_array(prog_grad,  vbo, "in_vert")
    vao_curl  = ctx.simple_vertex_array(prog_curl,  vbo, "in_vert")
    vao_vort  = ctx.simple_vertex_array(prog_vort,  vbo, "in_vert")
    vao_blur  = ctx.simple_vertex_array(prog_blur,  vbo, "in_vert")
    vao_show  = ctx.simple_vertex_array(prog_show,  vbo, "in_vert")

    # Sizes
    sim_w = max(32, int(WIDTH * SIM_SCALE))
    sim_h = max(32, int(HEIGHT * SIM_SCALE))
    texel = (1.0/sim_w, 1.0/sim_h)

    # Textures/FBOs
    vel_a = make_tex(ctx, (sim_w,sim_h), 2)
    vel_b = make_tex(ctx, (sim_w,sim_h), 2)
    dye_a = make_tex(ctx, (WIDTH,HEIGHT), 4)
    dye_b = make_tex(ctx, (WIDTH,HEIGHT), 4)
    prs   = make_tex(ctx, (sim_w,sim_h), 1)
    prs_b = make_tex(ctx, (sim_w,sim_h), 1)
    div   = make_tex(ctx, (sim_w,sim_h), 1)
    curl  = make_tex(ctx, (sim_w,sim_h), 1)
    tmp   = make_tex(ctx, (WIDTH,HEIGHT), 4)

    fbo_vel_a = ctx.framebuffer([vel_a])
    fbo_vel_b = ctx.framebuffer([vel_b])
    fbo_dye_a = ctx.framebuffer([dye_a])
    fbo_dye_b = ctx.framebuffer([dye_b])
    fbo_prs   = ctx.framebuffer([prs])
    fbo_prs_b = ctx.framebuffer([prs_b])
    fbo_div   = ctx.framebuffer([div])
    fbo_curl  = ctx.framebuffer([curl])
    fbo_tmp   = ctx.framebuffer([tmp])

    for fbo in (fbo_vel_a,fbo_vel_b,fbo_dye_a,fbo_dye_b,fbo_prs,fbo_prs_b,fbo_div,fbo_curl,fbo_tmp):
        fbo.use(); ctx.clear(0,0,0,1)
    ctx.screen.use()

    # Const uniforms
    prog_div['texel'].value = texel
    prog_jac['texel'].value = texel
    prog_grad['texel'].value = texel
    prog_curl['texel'].value = texel
    prog_vort['texel'].value = texel
    prog_vort['eps'].value = VORTICITY

    # Seed a soft puff of dye
    fbo_dye_b.use()
    dye_a.use(location=0)
    prog_splat['field'].value = 0
    prog_splat['point'].value = (0.5,0.5)
    prog_splat['value'].value = (0.9,0.4,0.9)
    prog_splat['radius'].value = 0.12
    vao_splat.render(moderngl.TRIANGLE_STRIP)
    dye_a,dye_b = dye_b,dye_a
    fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

    # Loop
    palette_on = PALETTE_ON
    running = True
    prev = time.time()
    print("SPACE pause  C clear  [/] vorticity -/+  P palette")
    while not glfw.window_should_close(win):
        glfw.poll_events()
        if glfw.get_key(win,glfw.KEY_ESCAPE)==glfw.PRESS: break
        if glfw.get_key(win,glfw.KEY_SPACE)==glfw.PRESS:
            #running = !running if False else not running; time.sleep(0.12)
            running = running if False else not running; time.sleep(0.12)
        if glfw.get_key(win,glfw.KEY_C)==glfw.PRESS:
            for fbo in (fbo_vel_a,fbo_vel_b,fbo_dye_a,fbo_dye_b,fbo_prs,fbo_prs_b,fbo_div,fbo_curl):
                fbo.use(); ctx.clear(0,0,0,1)
            ctx.screen.use(); time.sleep(0.12)
        if glfw.get_key(win,glfw.KEY_LEFT_BRACKET)==glfw.PRESS:
            val = max(0.0, prog_vort['eps'].value - 0.1); prog_vort['eps'].value = val; time.sleep(0.05)
        if glfw.get_key(win,glfw.KEY_RIGHT_BRACKET)==glfw.PRESS:
            val = min(5.0, prog_vort['eps'].value + 0.1); prog_vort['eps'].value = val; time.sleep(0.05)
        if glfw.get_key(win,glfw.KEY_P)==glfw.PRESS:
            palette_on = 1 - palette_on; time.sleep(0.12)

        now = time.time()
        dt = min(DT_CLAMP, now - prev)
        prev = now

        # Mouse velocity (UV/sec)
        u,v   = mouse.uv()
        pu,pv = mouse.puv()
        mvel = ((u-pu)/max(dt,1e-5), (v-pv)/max(dt,1e-5))
        mforce = (mvel[0]*MOUSE_VEL_SCALE, mvel[1]*MOUSE_VEL_SCALE) if mouse.down else (0.0,0.0)

        if running:
            # Substeps
            sdt = dt / SUBSTEPS
            for _ in range(SUBSTEPS):
                # 1) Advect velocity by itself
                fbo_vel_b.use()
                vel_a.use(location=0); vel_a.use(location=1)
                prog_adv['vel'].value = 0
                prog_adv['src'].value = 1
                prog_adv['dt'].value = sdt
                prog_adv['dissipation'].value = VEL_DISSIPATION
                vao_adv.render(moderngl.TRIANGLE_STRIP)
                vel_a,vel_b = vel_b,vel_a
                fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                # 2) Add mouse velocity splat (throw)
                if mouse.down and (mforce[0]!=0.0 or mforce[1]!=0.0):
                    fbo_vel_b.use()
                    vel_a.use(location=0)
                    prog_splat['field'].value = 0
                    prog_splat['point'].value = (u,v)
                    prog_splat['value'].value = (mforce[0], mforce[1], 0.0)
                    prog_splat['radius'].value = MOUSE_RADIUS
                    vao_splat.render(moderngl.TRIANGLE_STRIP)
                    vel_a,vel_b = vel_b,vel_a
                    fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                # 3) Vorticity confinement (adds swirls)
                if VORTICITY > 0.0:
                    # curl
                    fbo_curl.use()
                    vel_a.use(location=0)
                    prog_curl['vel'].value = 0
                    vao_curl.render(moderngl.TRIANGLE_STRIP)
                    # add force
                    fbo_vel_b.use()
                    vel_a.use(location=0); curl.use(location=1)
                    prog_vort['vel'].value = 0
                    prog_vort['curlTex'].value = 1
                    prog_vort['dt'].value = sdt
                    vao_vort.render(moderngl.TRIANGLE_STRIP)
                    vel_a,vel_b = vel_b,vel_a
                    fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                # 4) Projection (divergence-free)
                # divergence
                fbo_div.use()
                vel_a.use(location=0)
                prog_div['vel'].value = 0
                vao_div.render(moderngl.TRIANGLE_STRIP)
                # Jacobi iterations
                for _j in range(JACOBI_ITERS):
                    fbo_prs_b.use()
                    prs.use(location=0); div.use(location=1)
                    prog_jac['prs'].value = 0
                    prog_jac['div'].value = 1
                    vao_jac.render(moderngl.TRIANGLE_STRIP)
                    prs,prs_b = prs_b,prs
                    fbo_prs,fbo_prs_b = fbo_prs_b,fbo_prs
                # subtract gradient
                fbo_vel_b.use()
                vel_a.use(location=0); prs.use(location=1)
                prog_grad['vel'].value = 0
                prog_grad['prs'].value = 1
                vao_grad.render(moderngl.TRIANGLE_STRIP)
                vel_a,vel_b = vel_b,vel_a
                fbo_vel_a,fbo_vel_b = fbo_vel_b,fbo_vel_a

                # 5) Advect dye with updated velocity
                fbo_dye_b.use()
                vel_a.use(location=0); dye_a.use(location=1)
                prog_adv['vel'].value = 0
                prog_adv['src'].value = 1
                prog_adv['dt'].value = sdt
                prog_adv['dissipation'].value = DYE_DISSIPATION
                vao_adv.render(moderngl.TRIANGLE_STRIP)
                dye_a,dye_b = dye_b,dye_a
                fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

                # 6) Paint dye under the mouse (optional, looks nice)
                if mouse.down:
                    fbo_dye_b.use()
                    dye_a.use(location=0)
                    prog_splat['field'].value = 0
                    prog_splat['point'].value = (u,v)
                    prog_splat['value'].value = (0.9, 0.9, 0.9)
                    prog_splat['radius'].value = 0.02
                    vao_splat.render(moderngl.TRIANGLE_STRIP)
                    dye_a,dye_b = dye_b,dye_a
                    fbo_dye_a,fbo_dye_b = fbo_dye_b,fbo_dye_a

        # Show
        ctx.screen.use()
        dye_a.use(location=0)
        prog_show['dye'].value = 0
        prog_show['palette_on'].value = palette_on
        vao_show.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(win)

    glfw.terminate()

if __name__ == "__main__":
    main()
