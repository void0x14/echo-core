const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create modules
    const core_config_mod = b.createModule(.{
        .root_source_file = b.path("src/core/config.zig"),
        .target = target,
        .optimize = optimize,
    });

    const gguf_mod = b.createModule(.{
        .root_source_file = b.path("src/gguf/reader.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_mod.addImport("core_config", core_config_mod);

    // Create the main module
    const echo_core_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true, // src/llama: glibc expf/logf/sinf/cosf (llama.cpp ile aynı)
    });
    echo_core_module.addImport("core_config", core_config_mod);
    echo_core_module.addImport("gguf", gguf_mod);
    echo_core_module.addImport("config", core_config_mod);

    // Executable
    const exe = b.addExecutable(.{
        .name = "echo-core-zig",
        .root_module = echo_core_module,
        .use_llvm = true, // src/llama: llvm.x86.* intrinsic extern'leri
    });
    b.installArtifact(exe);
    // kullanıcıya dönük ad (eski ad betikler için kalır); `echo` kabuk komutuyla çakıştığı için kullanılmadı
    b.getInstallStep().dependOn(&b.addInstallArtifact(exe, .{ .dest_sub_path = "echo-core" }).step);

    // Deney/araç programları yalnız -Ddeneyler=true ile derlenir (varsayılan derleme hızlı kalsın)
    const deneyler = b.option(bool, "deneyler", "deney ve araç programlarını da derle") orelse false;

    // Birim testleri: zig build test
    const test_mod = b.createModule(.{ .root_source_file = b.path("src/testler.zig"), .target = target, .optimize = optimize, .link_libc = true });
    test_mod.addImport("core_config", core_config_mod);
    const tests = b.addTest(.{ .root_module = test_mod, .use_llvm = true });
    b.step("test", "birim testleri").dependOn(&b.addRunArtifact(tests).step);

    // Dump model tool
    const dump_model_mod = b.createModule(.{
        .root_source_file = b.path("src/tools/dump_model_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    dump_model_mod.addImport("core_config", core_config_mod);
    dump_model_mod.addImport("gguf", gguf_mod);
    if (deneyler) b.installArtifact(b.addExecutable(.{
        .name = "dump-model",
        .root_module = dump_model_mod,
    }));

    // Deney A (blok düzeyi): tek çekirdekli blok ring'i, gerçek GGUF tensör baytları üzerinde
    const deney_a_mod = b.createModule(.{
        .root_source_file = b.path("src/deney_a_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    deney_a_mod.addImport("gguf", gguf_mod);
    if (deneyler) b.installArtifact(b.addExecutable(.{
        .name = "deney-a",
        .root_module = deney_a_mod,
    }));

    // Deney A (uçtan uca): decode döngüsüne ring geçişi, token/s ölçümü
    const deney_e2e_mod = b.createModule(.{
        .root_source_file = b.path("src/deney_a_e2e_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    deney_e2e_mod.addImport("core_config", core_config_mod);
    deney_e2e_mod.addImport("gguf", gguf_mod);
    deney_e2e_mod.addImport("config", core_config_mod);
    if (deneyler) b.installArtifact(b.addExecutable(.{
        .name = "deney-a-e2e",
        .root_module = deney_e2e_mod,
    }));

    // Deney K: kolektif dolaşım (B tarifi). Deney B (SPSC yerleşimi) ile karıştırılmamalı.
    const kolektif_mod = b.createModule(.{
        .root_source_file = b.path("src/kolektif/kolektif_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (deneyler) b.installArtifact(b.addExecutable(.{
        .name = "kolektif",
        .root_module = kolektif_mod,
    }));

    // Deney S: Satürn. Kolektif dolaşımın donanım gözlemleri (PMU/MSR/resctrl).
    const saturn_mod = b.createModule(.{
        .root_source_file = b.path("src/saturn/saturn_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    if (deneyler) b.installArtifact(b.addExecutable(.{
        .name = "saturn",
        .root_module = saturn_mod,
    }));

    // Analyze GGUF tool - standalone, needs Zig 0.17 API refactor
    // Temporarily disabled until old std.fs API calls are updated
    // const analyze_mod = b.createModule(.{
    //     .root_source_file = b.path("src/tools/analyze_gguf.zig"),
    //     .target = target,
    //     .optimize = optimize,
    // });
    // b.installArtifact(b.addExecutable(.{
    //     .name = "analyze-gguf",
    //     .root_module = analyze_mod,
    // }));
}
