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

    // Test files to run
    const test_files = [_][]const u8{
        "src/kernel_tests.zig",
        "src/inference_tests.zig",
        "src/gguf_tests.zig",
        "src/kv_cache_tests.zig",
        "src/tokenizer_tests.zig",
    };

    const test_step = b.step("test", "Run all tests");
    for (test_files) |test_file| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(test_file),
            .target = target,
            .optimize = optimize,
        });
        test_mod.addImport("core_config", core_config_mod);
        const test_exe = b.addTest(.{ .root_module = test_mod });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }

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
