VS = """
#version 330
in vec2 in_vert;
out vec2 uv;
void main(){ gl_Position = vec4(in_vert,0.0,1.0); uv = (in_vert + 1.0)*0.5; }
"""

FS_ADVECT = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel;
uniform sampler2D src;
uniform float dt;
uniform float dissipation;
uniform float top_fade;  // height of the band (UV) at the top where values fade out; 0 = off
void main(){
    vec2 v = texture(vel, uv).xy;
    vec2 prev = clamp(uv - v*dt, vec2(0.001), vec2(0.999));
    float d = dissipation;
    if (top_fade > 0.0) d *= mix(0.97, 1.0, smoothstep(1.0, 1.0 - top_fade, uv.y));
    fragColor = texture(src, prev) * d;
}
"""

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
    fragColor = base + vec4(value*fall, 0.0);
}
"""

FS_FORCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec3 value;
void main(){
    vec4 base = texture(field, uv);
    fragColor = base + vec4(value, 0.0);
}
"""

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
uniform int open_top;  // 1: smoke leaves through the top edge (pressure 0 there)
void main(){
    float L = texture(prs, uv - vec2(texel.x,0)).r;
    float R = texture(prs, uv + vec2(texel.x,0)).r;
    float B = texture(prs, uv - vec2(0,texel.y)).r;
    float T = texture(prs, uv + vec2(0,texel.y)).r;
    if (open_top == 1 && uv.y + texel.y > 1.0) T = 0.0;
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
uniform int open_top;
void main(){
    float L = texture(prs, uv - vec2(texel.x,0)).r;
    float R = texture(prs, uv + vec2(texel.x,0)).r;
    float B = texture(prs, uv - vec2(0,texel.y)).r;
    float T = texture(prs, uv + vec2(0,texel.y)).r;
    if (open_top == 1 && uv.y + texel.y > 1.0) T = 0.0;
    vec2 grad = 0.5 * vec2(R - L, T - B);
    vec2 v = texture(vel, uv).xy - grad;
    fragColor = vec4(v,0,1);
}
"""

FS_MASK_FORCE_FULL = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel_in;
uniform sampler2D mask_curr;
uniform vec2 texel;
uniform float dt;
uniform float amp_normal;
void main(){
    float M = texture(mask_curr, uv).r;
    vec2 v = texture(vel_in, uv).xy;
    if (M > 0.01){
        float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
        float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
        vec2 g = 0.5*vec2(Mx,My);
        vec2 n = normalize(g + 1e-6);
        vec2 add = amp_normal * n * M;
        v += dt * add;
    }
    fragColor = vec4(v,0,1);
}
"""

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

FS_MASK_FORCE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D vel_in;
uniform sampler2D mask_curr;
uniform sampler2D mask_prev;
uniform vec2 texel;
uniform float dt;
uniform float edge_thresh;
uniform float amp_normal;
uniform float amp_tangent;
uniform int   use_temporal;
uniform float border;  // UV band at the frame edges where mask edges are ignored
float border_fade(vec2 p){
    vec2 e = min(p, 1.0 - p);
    return smoothstep(0.0, border, min(e.x, e.y));
}
void main(){
    float M  = texture(mask_curr, uv).r;
    float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
    float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
    vec2 g = 0.5*vec2(Mx,My);
    float gmag = length(g);
    vec2 v = texture(vel_in, uv).xy;
    if (gmag > edge_thresh){
        vec2 n = g / (gmag + 1e-6);
        vec2 t = vec2(-n.y, n.x);
        float growth = 1.0;
        if (use_temporal==1){
            float Mp = texture(mask_prev, uv).r;
            growth = (M - Mp) / max(dt, 1e-4);
            growth = clamp(growth * 0.5 + 0.5, 0.0, 1.0);
        }
        vec2 add = amp_normal*n*growth + amp_tangent*t*growth; // w/ tangent
        add *= smoothstep(edge_thresh, edge_thresh*3.0, gmag);
        add *= border_fade(uv);
        v += dt * add;
    }
    fragColor = vec4(v,0,1);
}
"""

FS_MASK_DYE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye_in;
uniform sampler2D mask_curr;
uniform vec2 texel;
uniform float edge_thresh;
uniform vec3 edge_color;
uniform float strength;
uniform float border;   // UV band at the frame edges where mask edges are ignored
uniform float dye_cap;  // outline stops adding dye once the dye here is this bright
void main(){
    float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
    float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
    float gmag = length(0.5*vec2(Mx,My));
    vec3 base = texture(dye_in, uv).rgb;
    float k = smoothstep(edge_thresh, edge_thresh*3.0, gmag);
    vec2 e = min(uv, 1.0 - uv);
    k *= smoothstep(0.0, border, min(e.x, e.y));
    k *= clamp(1.0 - dot(base, vec3(0.299, 0.587, 0.114)) / dye_cap, 0.0, 1.0);
    fragColor = vec4(base + strength*k*edge_color, 1.0);
}
"""

FS_GHOST = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D ghost_in;
uniform sampler2D mask_curr;
uniform float decay;     // per-frame multiplier on the old trail
uniform float stamp;     // 1 when a person mask is current, else 0
void main(){
    float g = texture(ghost_in, uv).r * decay;
    float m = texture(mask_curr, uv).r * stamp;
    fragColor = vec4(max(g, m), 0, 0, 1);
}
"""

FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
uniform sampler2D dye2;     // the next mode's output while crossfading between modes
uniform float mode_mix;     // 0 = all dye, 1 = all dye2
uniform int palette_on;
uniform int palette_id;   // current palette (0..N-1)
uniform int palette_id2;  // next palette for crossfade
uniform float palette_mix; // 0..1 crossfade amount
uniform float glow_strength; // 0 disables
uniform float glow_radius;   // in UV units
uniform float vignette;      // 0 disables, 1 = black corners
uniform float aspect;        // width / height
uniform sampler2D ghost;     // fading silhouette trail
uniform sampler2D mask_now;  // current silhouette (so the trail shows behind it)
uniform float ghost_strength;
uniform vec3 ghost_color;
#define MAX_EYES 4
uniform vec4 eyes[MAX_EYES];        // per pair: alpha (0 = hidden), centre u, centre v, openness
uniform float eyes_size[MAX_EYES];  // eye width as a fraction of screen height

// One almond-shaped eye with a slit pupil. p is in eye units (x: -1..1 across).
vec3 eye(vec2 p, float open_){
    float h = max(open_, 0.02) * 0.42 * (1.0 - p.x * p.x);
    float inside = smoothstep(0.02, -0.04, abs(p.y) - h) * step(abs(p.x), 1.0);
    float pupil = smoothstep(0.10, 0.04, abs(p.x) - 0.12 * (1.0 - abs(p.y) / 0.42));
    vec3 iris = mix(vec3(1.0, 0.85, 0.2), vec3(1.0, 0.15, 0.02), smoothstep(0.0, 0.8, length(p)));
    vec3 col = inside * iris * (1.0 - 0.9 * pupil);
    float glow = exp(-3.0 * dot(p * vec2(0.6, 2.0), p * vec2(0.6, 2.0))) * open_;
    return col + glow * vec3(0.9, 0.08, 0.02) * 0.6;
}

vec3 field(vec2 p){
    vec3 a = texture(dye, p).rgb;
    if (mode_mix <= 0.0) return a;
    return mix(a, texture(dye2, p).rgb, mode_mix);
}

// 3-stop gradient helper
vec3 grad3(float t, vec3 a, vec3 b, vec3 c){
    t = clamp(t, 0.0, 1.0);
    if (t < 0.5) return mix(a, b, t*2.0);
    return mix(b, c, (t-0.5)*2.0);
}

// ---- spooky palettes ----
// 0: Pumpkin (black -> orange -> purple)
vec3 pal_pumpkin(float t){ return grad3(t, vec3(0.0), vec3(1.0,0.5,0.0), vec3(0.45,0.0,0.6)); }
// 1: Slime (black -> toxic green -> magenta)
vec3 pal_slime(float t){ return grad3(t, vec3(0.0), vec3(0.0,0.78,0.20), vec3(0.9,0.0,0.9)); }
// 2: Ember (deep red -> orange -> yellow)
vec3 pal_ember(float t){ return grad3(t, vec3(0.15,0.0,0.0), vec3(1.0,0.35,0.0), vec3(1.0,0.82,0.2)); }
// 3: Midnight (blue -> purple -> magenta)
vec3 pal_midnight(float t){ return grad3(t, vec3(0.0,0.1,0.3), vec3(0.25,0.0,0.45), vec3(0.9,0.0,0.6)); }
// 4: Ectoplasm (teal -> neon green -> white)
vec3 pal_ecto(float t){  return grad3(t, vec3(0.0,0.30,0.30), vec3(0.16,0.85,0.34), vec3(0.8)); }
// 5: Blood Moon (black -> crimson -> amber)
vec3 pal_blood(float t){ return grad3(t, vec3(0.0), vec3(0.6,0.02,0.06), vec3(1.0,0.5,0.0)); }

vec3 pick_palette(int id, float t){
    if(id==0) return pal_pumpkin(t);
    if(id==1) return pal_slime(t);
    if(id==2) return pal_ember(t);
    if(id==3) return pal_midnight(t);
    if(id==4) return pal_ecto(t);
    if(id==5) return pal_blood(t);
    return pal_pumpkin(t);
}

void main(){
    vec3 dye_rgb = field(uv);
    if (palette_on == 0){
        fragColor = vec4(dye_rgb, 1.0);
        return;
    }
    // luminance → perceptual curve (same as before, just named)
    const vec3 LUMA = vec3(0.299, 0.587, 0.114);
    float l = dot(dye_rgb, LUMA);
    l = 1.0 - exp(-2.3 * l);

    float pm = clamp(palette_mix, 0.0, 1.0);
    vec3 outc = mix(pick_palette(palette_id, l), pick_palette(palette_id2, l), pm);

    // Glow: blur the dye with two rings of taps and add it back, palette-coloured.
    if (glow_strength > 0.0){
        float gl = 0.0;
        for (int i = 0; i < 12; i++){
            float ang = 6.2831853 * (float(i) + 0.5) / 12.0;
            vec2 dir = vec2(cos(ang) / aspect, sin(ang));
            gl += dot(field(uv + dir * glow_radius * 0.4), LUMA);
            gl += dot(field(uv + dir * glow_radius), LUMA);
        }
        gl = 1.0 - exp(-2.3 * gl / 24.0);
        vec3 gc = mix(pick_palette(palette_id, gl), pick_palette(palette_id2, gl), pm);
        outc += glow_strength * gc * gl;
    }

    // Ghost trail: pale cold after-image, strongest just behind a moving person.
    if (ghost_strength > 0.0){
        float g = texture(ghost, uv).r;
        float now_ = texture(mask_now, uv).r;
        float trail = max(g - 0.6 * now_, 0.0);
        outc += ghost_strength * ghost_color * (trail + 0.25 * g);
    }

    const float tilt = 0.28;  // angled inward: angry
    const mat2 rot_l = mat2(cos(tilt), sin(tilt), -sin(tilt), cos(tilt));
    const mat2 rot_r = mat2(cos(tilt), -sin(tilt), sin(tilt), cos(tilt));
    for (int i = 0; i < MAX_EYES; i++){
        if (eyes[i].x <= 0.0) continue;
        vec2 d = (uv - eyes[i].yz) * vec2(aspect, 1.0) / eyes_size[i];
        if (dot(d, d) > 16.0) continue;  // far from this pair
        vec2 l = rot_l * (d - vec2(-1.25, 0.0));
        vec2 r = rot_r * (d - vec2(1.25, 0.0));
        outc += eyes[i].x * (eye(l, eyes[i].w) + eye(r, eyes[i].w));
    }

    if (vignette > 0.0){
        vec2 d = (uv - 0.5) * vec2(aspect, 1.0) / (0.5 * aspect);
        outc *= 1.0 - vignette * smoothstep(0.35, 1.25, length(d));
    }
    fragColor = vec4(outc, 1.0);
}
"""

# ---- Slime mold (physarum) ----
# Agents live in a texture: rg = position (UV), b = heading (radians, pixel space).
FS_SLIME_AGENTS = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D agents;
uniform sampler2D trail;
uniform sampler2D mask;
uniform sampler2D mask_mips;  // the mask again, mipmapped: people smelled from afar
uniform vec2 texel;
uniform float sensor_angle;
uniform float sensor_dist;  // pixels
uniform float turn;
uniform float speed;        // pixels per step
uniform float food;         // how strongly people attract the slime
uniform float outline;      // extra pull toward people's outlines (vs. their insides)
uniform float seed;

float rand(){
    uint h = uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uint(seed * 1000.0) * 26699u;
    h = (h ^ (h >> 16)) * 0x7feb352du;
    h = (h ^ (h >> 15)) * 0x846ca68bu;
    h ^= h >> 16;
    return float(h) / 4294967295.0;
}

float sense(vec2 pos, float a){
    vec2 q = pos + vec2(cos(a), sin(a)) * texel * sensor_dist;
    float e = textureLod(mask_mips, q, 2.0).r;
    float smell = texture(mask, q).r + 3.0 * textureLod(mask_mips, q, 3.0).r + 6.0 * textureLod(mask_mips, q, 5.0).r
                + outline * 4.0 * e * (1.0 - e);
    return texture(trail, q).r + food * smell;
}

void main(){
    vec4 ag = texelFetch(agents, ivec2(gl_FragCoord.xy), 0);
    vec2 pos = ag.xy;
    float a = ag.z;
    float r = rand();
    float F = sense(pos, a);
    float L = sense(pos, a + sensor_angle);
    float R = sense(pos, a - sensor_angle);
    if (F >= L && F >= R) {
    } else if (F < L && F < R) {
        a += (r < 0.5 ? turn : -turn);
    } else if (L > R) {
        a += turn;
    } else {
        a -= turn;
    }
    pos += vec2(cos(a), sin(a)) * texel * speed;
    if (pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0){
        pos = clamp(pos, vec2(0.0), vec2(1.0));
        a = r * 6.2831853;  // bounce off in a random direction
    }
    fragColor = vec4(pos, mod(a, 6.2831853), 0.0);
}
"""

FS_COPY_R = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D src;
void main(){ fragColor = vec4(texture(src, uv).r); }
"""

# One point per agent, drawn with additive blending into the trail.
VS_SLIME_DEPOSIT = """
#version 330
uniform sampler2D agents;
uniform sampler2D mask_mips;
uniform int agent_w;
uniform float amount;
uniform float feed_glow;  // slime on a person leaves this many times more trail
out float strength;
void main(){
    vec2 p = texelFetch(agents, ivec2(gl_VertexID % agent_w, gl_VertexID / agent_w), 0).xy;
    strength = amount * (1.0 + feed_glow * textureLod(mask_mips, p, 1.0).r);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

FS_SLIME_DEPOSIT = """
#version 330
in float strength;
out vec4 fragColor;
void main(){ fragColor = vec4(strength); }
"""

FS_SLIME_DIFFUSE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D trail;
uniform vec2 texel;
uniform float decay;    // per step
uniform float diffuse;  // 0..1 blend toward the 3x3 blur
void main(){
    float c = texture(trail, uv).r;
    float s = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            s += texture(trail, uv + vec2(x, y) * texel).r;
    fragColor = vec4(mix(c, s / 9.0, diffuse) * decay);
}
"""

# ---- Ripples (2D wave equation) ----
# field: r = height now, g = height one step ago.
FS_WAVE_STEP = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform sampler2D mask_curr;
uniform sampler2D mask_prev;
uniform vec2 texel;
uniform float c2;         // (wave speed)^2 in pixels per step; < 0.5 to stay stable
uniform float damp;       // per step, applied to the surface's velocity
uniform int have_mask;
uniform float move_amp;   // pushes the water where someone moves
uniform float pulse_amp;  // outlines throb with this force
uniform float pulse;      // -1..1, the outline's current phase

// Mask blurred over a few pixels: hard, jittery mask edges make grainy noise.
float soft(sampler2D m){
    float s = texture(m, uv).r;
    for (int i = 0; i < 8; i++){
        float a = 0.785398 * float(i);
        s += texture(m, uv + vec2(cos(a), sin(a)) * texel * 4.0).r;
    }
    return s / 9.0;
}

void main(){
    vec2 f = texture(field, uv).rg;
    float h = f.r, hp = f.g;
    float n4 = texture(field, uv - vec2(texel.x, 0.0)).r + texture(field, uv + vec2(texel.x, 0.0)).r
             + texture(field, uv - vec2(0.0, texel.y)).r + texture(field, uv + vec2(0.0, texel.y)).r;
    float hn = h + (h - hp) * damp + c2 * (n4 - 4.0 * h);
    hn = mix(hn, n4 * 0.25, 0.005);  // a little viscosity to keep pixel-scale ripples down
    if (have_mask == 1){
        float m = soft(mask_curr);
        float edge = clamp(4.0 * m * (1.0 - m), 0.0, 1.0);
        hn += move_amp * (m - soft(mask_prev));       // moving pushes water ahead of you
        hn += pulse_amp * pulse * edge;               // outlines throb out rings
        hn *= 1.0 - 0.3 * smoothstep(0.6, 0.95, m);   // calm inside people
    }
    fragColor = vec4(clamp(hn * 0.9995, -4.0, 4.0), h, 0.0, 0.0);
}
"""

FS_WAVE_DROP = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec2 point;
uniform float amount;
uniform float radius;
uniform float aspect;
void main(){
    vec2 d = (uv - point) * vec2(aspect, 1.0);
    float b = amount * exp(-dot(d, d) / (radius * radius));
    fragColor = texture(field, uv) + vec4(b, b, 0.0, 0.0);  // raise both: no initial velocity
}
"""

FS_WAVE_SHADE = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D field;
uniform vec2 texel;
uniform float gain;
void main(){
    float h = texture(field, uv).r;
    vec2 g = vec2(
        texture(field, uv + vec2(texel.x, 0.0)).r - texture(field, uv - vec2(texel.x, 0.0)).r,
        texture(field, uv + vec2(0.0, texel.y)).r - texture(field, uv - vec2(0.0, texel.y)).r);
    float v = gain * (max(h, 0.0) + 1.5 * length(g));  // bright crests and steep slopes
    fragColor = vec4(vec3(v), 1.0);
}
"""

FS_SHOW_CAM = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D cam;
void main(){ fragColor = texture(cam, uv); }
"""

VS_DEBUGOVERLAY = """
#version 330
in vec2 in_vert;
in vec2 in_uv;
out vec2 uv;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    uv = in_uv;
}
"""

FS_DEBUGOVERLAY = """
#version 330
in vec2 uv;
out vec4 fragColor;
uniform sampler2D fontTexture;
uniform vec3 textColor;
void main() {
    float alpha = texture(fontTexture, uv).r;
    fragColor = vec4(textColor, alpha);
}
"""
