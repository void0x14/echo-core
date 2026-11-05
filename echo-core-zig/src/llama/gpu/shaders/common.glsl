// Ortak tanımlar. Tüm ofsetler push constant'ta 32 bit kelime birimindedir; vec4/uvec4 bölgeler 16 bayt hizalıdır.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(std430, set = 0, binding = 0) buffer AF { float f[]; };
layout(std430, set = 0, binding = 0) buffer AV { vec4 v[]; };
layout(std430, set = 0, binding = 0) buffer AU { uint u[]; };
layout(std430, set = 0, binding = 0) buffer AQ { uvec4 uq[]; };
layout(std430, set = 0, binding = 1) buffer PR { uint token; uint head; uint n_cells; uint mode; float pos; uint pad1[11]; uint ix; } par;

// işaretli byte: ACO extract_i8 + cvt'yi tek SDWA komutuna indirir
vec4 u8v(uint w) { int i = int(w); return vec4(float(bitfieldExtract(i, 0, 8)), float(bitfieldExtract(i, 8, 8)), float(bitfieldExtract(i, 16, 8)), float(i >> 24)); }
// ggml quantize_row_q8_0: en yakına yuvarla
uint pack8(vec4 a, float id) {
    uvec4 i = uvec4(ivec4(roundEven(a * id))) & 255u;
    return i.x | (i.y << 8) | (i.z << 16) | (i.w << 24);
}
// ggml ölçeği f16 saklar; aynı yuvarlama
float dq16(float d) { return unpackHalf2x16(packHalf2x16(vec2(d, 0.0))).x; }
float silu(float x) { return x / (1.0 + exp(-x)); }
vec4 silu4(vec4 x) { return x / (1.0 + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
vec4 sigmoid4(vec4 x) { return 1.0 / (1.0 + exp(-x)); }
float softplus(float x) { return x > 20.0 ? x : log(1.0 + exp(x)); }
float wscale(uint d_off, uint i) { return unpackHalf2x16(u[d_off + i / 2u] >> ((i & 1u) * 16u)).x; }
