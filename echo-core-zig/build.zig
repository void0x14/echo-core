const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);
}
