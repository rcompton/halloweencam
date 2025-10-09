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
void main(){
    vec2 v = texture(vel, uv).xy;
    vec2 prev = clamp(uv - v*dt, vec2(0.001), vec2(0.999));
    fragColor = texture(src, prev) * dissipation;
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
        vec2 add = amp_normal*n*growth + amp_tangent*t;
        add *= smoothstep(edge_thresh, edge_thresh*3.0, gmag);
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
void main(){
    float Mx = texture(mask_curr, uv + vec2(texel.x,0)).r - texture(mask_curr, uv - vec2(texel.x,0)).r;
    float My = texture(mask_curr, uv + vec2(0,texel.y)).r - texture(mask_curr, uv - vec2(0,texel.y)).r;
    float gmag = length(0.5*vec2(Mx,My));
    vec3 base = texture(dye_in, uv).rgb;
    float k = smoothstep(edge_thresh, edge_thresh*3.0, gmag);
    fragColor = vec4(base + strength*k*edge_color, 1.0);
}
"""

FS_SHOW = """
#version 330
in vec2 uv; out vec4 fragColor;
uniform sampler2D dye;
uniform int palette_on;
uniform int palette_id;   // current palette (0..N-1)
uniform int palette_id2;  // next palette for crossfade
uniform float palette_mix; // 0..1 crossfade amount

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
    vec3 dye_rgb = texture(dye, uv).rgb;
    if (palette_on == 0){
        fragColor = vec4(dye_rgb, 1.0);
        return;
    }
    // luminance → perceptual curve (same as before, just named)
    float l = dot(dye_rgb, vec3(0.299, 0.587, 0.114));
    l = 1.0 - exp(-2.3 * l);

    vec3 a = pick_palette(palette_id, l);
    vec3 b = pick_palette(palette_id2, l);
    vec3 outc = mix(a, b, clamp(palette_mix, 0.0, 1.0));
    fragColor = vec4(outc, 1.0);
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
