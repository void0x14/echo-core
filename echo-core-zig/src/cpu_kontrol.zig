//! main'den önce çalışır (.init_array), temel x86-64 ile ayrı nesne olarak derlenir: derleme CPU'sundaki
//! özellik bu işlemcide yoksa "illegal instruction" yerine anlaşılır hata verir. Gerekli liste build.zig'den.
const gerekli = @import("cpu_gerekli");
// std kullanılmaz: std kodu ana programla paylaşılıp yerli CPU komutlarıyla derlenmiş olabilir
extern "c" fn write(fd: c_int, buf: [*]const u8, n: usize) isize;
extern "c" fn _exit(code: c_int) noreturn;
fn yaz(t: []const u8) void {
    _ = write(2, t.ptr, t.len);
}

fn cpuid(leaf: u32, sub: u32) [4]u32 {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (sub),
    );
    return .{ eax, ebx, ecx, edx };
}

fn kontrol() callconv(.c) void {
    const l1 = cpuid(1, 0);
    const l7 = if (cpuid(0, 0)[0] >= 7) cpuid(7, 0) else [4]u32{ 0, 0, 0, 0 };
    const liste = .{
        .{ "avx", l1[2], 28 },      .{ "fma", l1[2], 12 },      .{ "f16c", l1[2], 29 },
        .{ "avx2", l7[1], 5 },      .{ "bmi", l7[1], 3 },       .{ "bmi2", l7[1], 8 },
        .{ "avx512f", l7[1], 16 },  .{ "avx512bw", l7[1], 30 }, .{ "avx512vl", l7[1], 31 },
        .{ "avx512vnni", l7[2], 11 },
    };
    var eksik = false;
    inline for (liste) |o| {
        if (@field(gerekli, o[0]) and (o[1] >> o[2]) & 1 == 0) {
            if (!eksik) yaz("hata: bu program bu işlemcide çalışamaz; eksik CPU özelliği:");
            eksik = true;
            yaz(" " ++ o[0]);
        }
    }
    if (eksik) {
        yaz("\nBu makinede yeniden derleyin: zig build -Doptimize=ReleaseFast\n");
        _exit(1);
    }
}

export const echo_cpu_kontrol: *const fn () callconv(.c) void linksection(".init_array") = &kontrol;
