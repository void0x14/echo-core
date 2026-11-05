//! AVX2/FMA/F16C intrinsic'leri. llama.cpp kernel'ları bunlarla birebir port edildi.
//! LLVM intrinsic extern'leri yalnız LLVM backend'de çalışır (build.zig: use_llvm).
const std = @import("std");
const builtin = @import("builtin");

pub const supported = builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHasAll(builtin.cpu.features, .{ .avx2, .fma, .f16c });

pub const F8 = @Vector(8, f32);
pub const F4 = @Vector(4, f32);
pub const D4 = @Vector(4, f64);
pub const B32 = @Vector(32, i8);
pub const U32 = @Vector(32, u8);
pub const B16 = @Vector(16, i8);
pub const W16 = @Vector(16, i16);
pub const W8 = @Vector(8, i16);
pub const I8 = @Vector(8, i32);
pub const I4 = @Vector(4, i32);

extern fn @"llvm.x86.avx2.pmadd.ub.sw"(a: B32, b: B32) W16;
extern fn @"llvm.x86.avx2.pmadd.wd"(a: W16, b: W16) I8;
extern fn @"llvm.x86.avx2.psign.b"(a: B32, b: B32) B32;
extern fn @"llvm.x86.avx2.pshuf.b"(a: B32, b: B32) B32;
extern fn @"llvm.x86.avx2.packssdw"(a: I8, b: I8) W16;
extern fn @"llvm.x86.avx2.packsswb"(a: W16, b: W16) B32;
extern fn @"llvm.x86.avx.cvt.ps2dq.256"(a: F8) I8;
extern fn @"llvm.x86.avx.max.ps.256"(a: F8, b: F8) F8;
extern fn @"llvm.x86.sse.max.ps"(a: F4, b: F4) F4;
extern fn @"llvm.x86.ssse3.pshuf.b.128"(a: B16, b: B16) B16;
extern fn @"llvm.x86.ssse3.phadd.w.128"(a: W8, b: W8) W8;
extern fn @"llvm.x86.ssse3.phadd.d.128"(a: I4, b: I4) I4;
extern fn @"llvm.x86.sse2.pmadd.wd"(a: W8, b: W8) I4;
extern fn @"llvm.x86.sse3.hadd.ps"(a: F4, b: F4) F4;

// glibc libm: llama.cpp ile aynı fonksiyonlar. Zig compiler_rt aynı adları binary içinde
// tanımlayıp extern'i gölgelediği için (farklı algoritma, farklı ulp) dlsym ile alınır.
const MathFn = *const fn (f32) callconv(.c) f32;
extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn powf(x: f32, y: f32) f32; // DT_NEEDED libm.so bunun sayesinde kalır
var expf_fn: MathFn = undefined;
var logf_fn: MathFn = undefined;
var sinf_fn: MathFn = undefined;
var cosf_fn: MathFn = undefined;

pub fn initLibm() void {
    std.mem.doNotOptimizeAway(powf(1.0, 1.0));
    expf_fn = @ptrCast(dlsym(null, "expf").?);
    logf_fn = @ptrCast(dlsym(null, "logf").?);
    sinf_fn = @ptrCast(dlsym(null, "sinf").?);
    cosf_fn = @ptrCast(dlsym(null, "cosf").?);
}
pub inline fn expf(x: f32) f32 {
    return expf_fn(x);
}
pub inline fn logf(x: f32) f32 {
    return logf_fn(x);
}
pub inline fn sinf(x: f32) f32 {
    return sinf_fn(x);
}
pub inline fn cosf(x: f32) f32 {
    return cosf_fn(x);
}
pub inline fn powfGlibc(x: f32, y: f32) f32 {
    return powf(x, y);
}

pub inline fn maddubs(a: B32, b: B32) W16 {
    return @"llvm.x86.avx2.pmadd.ub.sw"(a, b);
}
pub inline fn madd(a: W16, b: W16) I8 {
    return @"llvm.x86.avx2.pmadd.wd"(a, b);
}
pub inline fn sign8(a: B32, b: B32) B32 {
    return @"llvm.x86.avx2.psign.b"(a, b);
}
pub inline fn shuffle8(a: B32, b: B32) B32 {
    return @"llvm.x86.avx2.pshuf.b"(a, b);
}
pub inline fn packs32(a: I8, b: I8) W16 {
    return @"llvm.x86.avx2.packssdw"(a, b);
}
pub inline fn packs16(a: W16, b: W16) B32 {
    return @"llvm.x86.avx2.packsswb"(a, b);
}
pub inline fn cvtps2dq(a: F8) I8 {
    return @"llvm.x86.avx.cvt.ps2dq.256"(a);
}
pub inline fn maxps(a: F8, b: F8) F8 {
    return @"llvm.x86.avx.max.ps.256"(a, b);
}
pub inline fn maxps128(a: F4, b: F4) F4 {
    return @"llvm.x86.sse.max.ps"(a, b);
}
pub inline fn shuffle8_128(a: B16, b: B16) B16 {
    return @"llvm.x86.ssse3.pshuf.b.128"(a, b);
}
pub inline fn hadd16(a: W8, b: W8) W8 {
    return @"llvm.x86.ssse3.phadd.w.128"(a, b);
}
pub inline fn hadd32(a: I4, b: I4) I4 {
    return @"llvm.x86.ssse3.phadd.d.128"(a, b);
}
pub inline fn madd128(a: W8, b: W8) I4 {
    return @"llvm.x86.sse2.pmadd.wd"(a, b);
}

pub inline fn splat(v: f32) F8 {
    return @splat(v);
}
pub inline fn loadF(p: [*]const f32) F8 {
    return @as(*align(1) const [8]f32, @ptrCast(p)).*;
}
pub inline fn storeF(p: [*]f32, v: F8) void {
    @as(*align(1) [8]f32, @ptrCast(p)).* = v;
}
/// _mm256_cvtph_ps(_mm_loadu_si128(p))
pub inline fn loadPh(p: [*]const u16) F8 {
    const h: @Vector(8, f16) = @as(*align(1) const [8]f16, @ptrCast(p)).*;
    return @floatCast(h);
}
/// _mm_storeu_si128(p, _mm256_cvtps_ph(v, 0))
pub inline fn storePh(p: [*]u16, v: F8) void {
    const h: @Vector(8, f16) = @floatCast(v);
    @as(*align(1) [8]f16, @ptrCast(p)).* = h;
}
pub inline fn loadB(p: *const anyopaque) B32 {
    return @as(*align(1) const [32]i8, @ptrCast(p)).*;
}
pub inline fn loadW(p: *const anyopaque) W16 {
    return @as(*align(1) const [16]i16, @ptrCast(p)).*;
}
pub inline fn fp16(v: u16) f32 {
    return @floatCast(@as(f16, @bitCast(v)));
}
pub inline fn fp32to16(v: f32) u16 {
    return @bitCast(@as(f16, @floatCast(v)));
}
/// hsum_float_8 (quants.c) ve sgemm.cpp hsum: toplama sırası sonuca yansır, @reduce kullanma
pub inline fn hsum(x: F8) f32 {
    var res = high128(x) + low128(x);
    res += @shuffle(f32, res, undefined, [4]i32{ 2, 3, 2, 3 });
    return res[0] + res[1];
}
/// GGML_F32x8_REDUCE: _mm_hadd_ps iki kez
pub inline fn hsumHadd(x: F8) f32 {
    const t0 = low128(x) + high128(x);
    const t1 = @"llvm.x86.sse3.hadd.ps"(t0, t0);
    return @"llvm.x86.sse3.hadd.ps"(t1, t1)[0];
}
pub inline fn low128(x: anytype) @Vector(@typeInfo(@TypeOf(x)).vector.len / 2, @typeInfo(@TypeOf(x)).vector.child) {
    const n = @typeInfo(@TypeOf(x)).vector.len / 2;
    const m = comptime blk: {
        var a: [n]i32 = undefined;
        for (0..n) |i| a[i] = @intCast(i);
        break :blk a;
    };
    return @shuffle(@typeInfo(@TypeOf(x)).vector.child, x, undefined, m);
}
pub inline fn high128(x: anytype) @Vector(@typeInfo(@TypeOf(x)).vector.len / 2, @typeInfo(@TypeOf(x)).vector.child) {
    const n = @typeInfo(@TypeOf(x)).vector.len / 2;
    const m = comptime blk: {
        var a: [n]i32 = undefined;
        for (0..n) |i| a[i] = @intCast(i + n);
        break :blk a;
    };
    return @shuffle(@typeInfo(@TypeOf(x)).vector.child, x, undefined, m);
}
