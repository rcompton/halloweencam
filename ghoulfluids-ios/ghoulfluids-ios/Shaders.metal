#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float2 position [[attribute(0)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
    VertexOut out;
    out.position = float4(in.position, 0.0, 1.0);
    // Map -1..1 to 0..1
    out.uv = (in.position + 1.0) * 0.5;
    // Flip Y for Metal texture coordinates vs normalized device coordinates if needed
    // But usually quad vertices are set up to match.
    // Assuming standard full screen quad: (-1, -1) to (1, 1)
    // out.uv.y = 1.0 - out.uv.y; // Flip if textures are upside down
    return out;
}

// -- Advection --
fragment float4 advect_main(VertexOut in [[stage_in]],
                            texture2d<float> vel [[texture(0)]],
                            texture2d<float> src [[texture(1)]],
                            constant float &dt [[buffer(0)]],
                            constant float &dissipation [[buffer(1)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 v = vel.sample(s, in.uv).xy;
    float2 prev = clamp(in.uv - v * dt, float2(0.001), float2(0.999));
    return src.sample(s, prev) * dissipation;
}

// -- Splat --
fragment float4 splat_main(VertexOut in [[stage_in]],
                           texture2d<float> field [[texture(0)]],
                           constant float2 &point [[buffer(0)]],
                           constant float3 &value [[buffer(1)]],
                           constant float &radius [[buffer(2)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 base = field.sample(s, in.uv);
    float2 d = in.uv - point;
    // Correct aspect ratio in distance? Assuming square-ish splat or just texture space distance
    float fall = exp(-dot(d, d) / (radius * radius));
    return base + float4(value * fall, 0.0);
}

// -- Force --
fragment float4 force_main(VertexOut in [[stage_in]],
                           texture2d<float> field [[texture(0)]],
                           constant float3 &value [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 base = field.sample(s, in.uv);
    return base + float4(value, 0.0);
}

// -- Divergence --
fragment float4 divergence_main(VertexOut in [[stage_in]],
                                texture2d<float> vel [[texture(0)]],
                                constant float2 &texel [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float L = vel.sample(s, in.uv - float2(texel.x, 0)).x;
    float R = vel.sample(s, in.uv + float2(texel.x, 0)).x;
    float B = vel.sample(s, in.uv - float2(0, texel.y)).y;
    float T = vel.sample(s, in.uv + float2(0, texel.y)).y;
    float div = 0.5 * ((R - L) + (T - B));
    return float4(div, 0, 0, 1);
}

// -- Jacobi --
fragment float4 jacobi_main(VertexOut in [[stage_in]],
                            texture2d<float> prs [[texture(0)]],
                            texture2d<float> div [[texture(1)]],
                            constant float2 &texel [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float L = prs.sample(s, in.uv - float2(texel.x, 0)).r;
    float R = prs.sample(s, in.uv + float2(texel.x, 0)).r;
    float B = prs.sample(s, in.uv - float2(0, texel.y)).r;
    float T = prs.sample(s, in.uv + float2(0, texel.y)).r;
    float b = div.sample(s, in.uv).r;
    float p = 0.25 * (L + R + B + T - b);
    return float4(p, 0, 0, 1);
}

// -- Gradient --
fragment float4 gradient_main(VertexOut in [[stage_in]],
                              texture2d<float> vel [[texture(0)]],
                              texture2d<float> prs [[texture(1)]],
                              constant float2 &texel [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float L = prs.sample(s, in.uv - float2(texel.x, 0)).r;
    float R = prs.sample(s, in.uv + float2(texel.x, 0)).r;
    float B = prs.sample(s, in.uv - float2(0, texel.y)).r;
    float T = prs.sample(s, in.uv + float2(0, texel.y)).r;
    float2 grad = 0.5 * float2(R - L, T - B);
    float2 v = vel.sample(s, in.uv).xy - grad;
    return float4(v, 0, 1);
}

// -- Curl --
fragment float4 curl_main(VertexOut in [[stage_in]],
                          texture2d<float> vel [[texture(0)]],
                          constant float2 &texel [[buffer(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float L = vel.sample(s, in.uv - float2(texel.x, 0)).y;
    float R = vel.sample(s, in.uv + float2(texel.x, 0)).y;
    float B = vel.sample(s, in.uv - float2(0, texel.y)).x;
    float T = vel.sample(s, in.uv + float2(0, texel.y)).x;
    float curl = 0.5 * ((T - B) - (R - L));
    return float4(curl, 0, 0, 1);
}

// -- Vorticity --
fragment float4 vorticity_main(VertexOut in [[stage_in]],
                               texture2d<float> vel [[texture(0)]],
                               texture2d<float> curlTex [[texture(1)]],
                               constant float2 &texel [[buffer(0)]],
                               constant float &eps [[buffer(1)]],
                               constant float &dt [[buffer(2)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float L = curlTex.sample(s, in.uv - float2(texel.x, 0)).r;
    float R = curlTex.sample(s, in.uv + float2(texel.x, 0)).r;
    float B = curlTex.sample(s, in.uv - float2(0, texel.y)).r;
    float T = curlTex.sample(s, in.uv + float2(0, texel.y)).r;
    float c = curlTex.sample(s, in.uv).r;
    float2 grad = 0.5 * float2(abs(R) - abs(L), abs(T) - abs(B));
    grad += 1e-5;
    float2 N = normalize(grad);
    float2 force = eps * float2(N.y, -N.x) * c;
    float2 v = vel.sample(s, in.uv).xy + dt * force;
    return float4(v, 0, 1);
}

// -- Mask Force (Edges) --
fragment float4 mask_force_main(VertexOut in [[stage_in]],
                                texture2d<float> vel_in [[texture(0)]],
                                texture2d<float> mask_curr [[texture(1)]],
                                texture2d<float> mask_prev [[texture(2)]],
                                constant float2 &texel [[buffer(0)]],
                                constant float &dt [[buffer(1)]],
                                constant float &edge_thresh [[buffer(2)]],
                                constant float &amp_normal [[buffer(3)]],
                                constant float &amp_tangent [[buffer(4)]],
                                constant int &use_temporal [[buffer(5)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float M = mask_curr.sample(s, in.uv).r;
    float Mx = mask_curr.sample(s, in.uv + float2(texel.x, 0)).r - mask_curr.sample(s, in.uv - float2(texel.x, 0)).r;
    float My = mask_curr.sample(s, in.uv + float2(0, texel.y)).r - mask_curr.sample(s, in.uv - float2(0, texel.y)).r;
    float2 g = 0.5 * float2(Mx, My);
    float gmag = length(g);
    float2 v = vel_in.sample(s, in.uv).xy;

    if (gmag > edge_thresh) {
        float2 n = g / (gmag + 1e-6);
        float2 t = float2(-n.y, n.x);
        float growth = 1.0;
        if (use_temporal == 1) {
            float Mp = mask_prev.sample(s, in.uv).r;
            growth = (M - Mp) / max(dt, 1e-4);
            growth = clamp(growth * 0.5 + 0.5, 0.0, 1.0);
        }
        float2 add = amp_normal * n * growth + amp_tangent * t * growth;
        add *= smoothstep(edge_thresh, edge_thresh * 3.0, gmag);
        v += dt * add;
    }
    return float4(v, 0, 1);
}

// -- Mask Dye --
fragment float4 mask_dye_main(VertexOut in [[stage_in]],
                              texture2d<float> dye_in [[texture(0)]],
                              texture2d<float> mask_curr [[texture(1)]],
                              constant float2 &texel [[buffer(0)]],
                              constant float &edge_thresh [[buffer(1)]],
                              constant float3 &edge_color [[buffer(2)]],
                              constant float &strength [[buffer(3)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float Mx = mask_curr.sample(s, in.uv + float2(texel.x, 0)).r - mask_curr.sample(s, in.uv - float2(texel.x, 0)).r;
    float My = mask_curr.sample(s, in.uv + float2(0, texel.y)).r - mask_curr.sample(s, in.uv - float2(0, texel.y)).r;
    float gmag = length(0.5 * float2(Mx, My));
    float3 base = dye_in.sample(s, in.uv).rgb;
    float k = smoothstep(edge_thresh, edge_thresh * 3.0, gmag);
    return float4(base + strength * k * edge_color, 1.0);
}

// -- Palettes --
float3 grad3(float t, float3 a, float3 b, float3 c){
    t = clamp(t, 0.0, 1.0);
    if (t < 0.5) return mix(a, b, t*2.0);
    return mix(b, c, (t-0.5)*2.0);
}

float3 pal_pumpkin(float t){ return grad3(t, float3(0.0), float3(1.0,0.5,0.0), float3(0.45,0.0,0.6)); }
float3 pal_slime(float t){ return grad3(t, float3(0.0), float3(0.0,0.78,0.20), float3(0.9,0.0,0.9)); }
float3 pal_ember(float t){ return grad3(t, float3(0.15,0.0,0.0), float3(1.0,0.35,0.0), float3(1.0,0.82,0.2)); }
float3 pal_midnight(float t){ return grad3(t, float3(0.0,0.1,0.3), float3(0.25,0.0,0.45), float3(0.9,0.0,0.6)); }
float3 pal_ecto(float t){  return grad3(t, float3(0.0,0.30,0.30), float3(0.16,0.85,0.34), float3(0.8)); }
float3 pal_blood(float t){ return grad3(t, float3(0.0), float3(0.6,0.02,0.06), float3(1.0,0.5,0.0)); }

float3 pick_palette(int id, float t){
    if(id==0) return pal_pumpkin(t);
    if(id==1) return pal_slime(t);
    if(id==2) return pal_ember(t);
    if(id==3) return pal_midnight(t);
    if(id==4) return pal_ecto(t);
    if(id==5) return pal_blood(t);
    return pal_pumpkin(t);
}

fragment float4 show_main(VertexOut in [[stage_in]],
                          texture2d<float> dye [[texture(0)]],
                          constant int &palette_on [[buffer(0)]],
                          constant int &palette_id [[buffer(1)]],
                          constant int &palette_id2 [[buffer(2)]],
                          constant float &palette_mix [[buffer(3)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float3 dye_rgb = dye.sample(s, in.uv).rgb;

    if (palette_on == 0) {
        return float4(dye_rgb, 1.0);
    }

    float l = dot(dye_rgb, float3(0.299, 0.587, 0.114));
    l = 1.0 - exp(-2.3 * l);

    float3 a = pick_palette(palette_id, l);
    float3 b = pick_palette(palette_id2, l);
    float3 outc = mix(a, b, clamp(palette_mix, 0.0, 1.0));

    // Simple alpha for blending: if dye is dark, it's transparent
    float alpha = length(outc);
    // Or use luminance
    float lum = dot(outc, float3(0.299, 0.587, 0.114));
    return float4(outc, smoothstep(0.0, 0.2, lum));
}

fragment float4 show_cam_main(VertexOut in [[stage_in]],
                              texture2d<float> cam [[texture(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    // Camera often comes in BGRA from AVFoundation, but Metal texture loader handles swizzle usually?
    // If manual CVMetalTextureCache is .bgra8Unorm, we just sample.
    return cam.sample(s, in.uv);
}

fragment float4 copy_main(VertexOut in [[stage_in]],
                          texture2d<float> src [[texture(0)]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    return src.sample(s, in.uv);
}
