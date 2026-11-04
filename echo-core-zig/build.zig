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
    });
    echo_core_module.addImport("core_config", core_config_mod);
    echo_core_module.addImport("gguf", gguf_mod);
    echo_core_module.addImport("config", core_config_mod);

    // Executable
    const exe = b.addExecutable(.{
        .name = "echo-core-zig",
        .root_module = echo_core_module,
    });
    b.installArtifact(exe);

    // Dump model tool
    const dump_model_mod = b.createModule(.{
        .root_source_file = b.path("src/tools/dump_model_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    dump_model_mod.addImport("core_config", core_config_mod);
    dump_model_mod.addImport("gguf", gguf_mod);
    b.installArtifact(b.addExecutable(.{
        .name = "dump-model",
        .root_module = dump_model_mod,
    }));

    // Deney C (blok düzeyi): istasyonsuz tek geçiş vs istasyonlu geçiş
    const deney_c_mod = b.createModule(.{
        .root_source_file = b.path("src/deney_c_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    deney_c_mod.addImport("gguf", gguf_mod);
    b.installArtifact(b.addExecutable(.{
        .name = "deney-c",
        .root_module = deney_c_mod,
    }));

    // Deney C (uçtan uca): decode döngüsüne istasyon boru hattı
    const deney_c_e2e_mod = b.createModule(.{
        .root_source_file = b.path("src/deney_c_e2e_main.zig"),
        .target = target,
        .optimize = optimize,
    });
    deney_c_e2e_mod.addImport("core_config", core_config_mod);
    deney_c_e2e_mod.addImport("gguf", gguf_mod);
    deney_c_e2e_mod.addImport("config", core_config_mod);
    b.installArtifact(b.addExecutable(.{
        .name = "deney-c-e2e",
        .root_module = deney_c_e2e_mod,
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
