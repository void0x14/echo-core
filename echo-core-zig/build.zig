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
        const test_exe = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(test_file),
                .target = target,
                .optimize = optimize,
            }),
        });
        const run_test = b.addRunArtifact(test_exe);
        test_step.dependOn(&run_test.step);
    }

    // Executable
    const exe = b.addExecutable(.{
        .name = "echo-core-zig",
        .root_module = echo_core_module,
    });
    b.installArtifact(exe);

    // NOTE: Tool executables temporarily disabled due to module/import path issues
    // These need separate refactoring to fix relative imports
    //
    // // Dump model tool
    // const dump_model_exe = b.addExecutable(.{
    //     .name = "dump-model",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("src/tools/dump_model.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //     }),
    // });
    // dump_model_exe.root_module.addImport("gguf", gguf_mod);
    // b.installArtifact(dump_model_exe);
    //
    // // Analyze GGUF tool
    // const analyze_exe = b.addExecutable(.{
    //     .name = "analyze-gguf",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("src/tools/analyze_gguf.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //     }),
    // });
    // analyze_exe.root_module.addImport("gguf", gguf_mod);
    // analyze_exe.root_module.addImport("core_config", core_config_mod);
    // b.installArtifact(analyze_exe);

    // Analyze GGUF tool - standalone, no module dependencies
    // Temporarily disabled due to Zig 0.16 API changes
    // const analyze_exe = b.addExecutable(.{
    //     .name = "analyze-gguf",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("src/tools/analyze_gguf.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //     }),
    // });
    // b.installArtifact(analyze_exe);
}
