const std = @import("std");
const config = @import("../core/config.zig");
const engine = @import("../inference/engine.zig");
const tokenizer = @import("../tokenizer/tokenizer.zig");

pub const ModelLoader = struct {
    config: config.ModelConfig,
    engine_: engine.Engine,

    pub fn load(model_path: []const u8, allocator: std.mem.Allocator) !ModelLoader {
        _ = model_path;
        const cfg = config.ModelConfig{
            .vocab_size = 32000,
            .hidden_dim = 4096,
            .num_heads = 32,
            .num_kv_heads = 32,
            .head_dim = 128,
            .num_layers = 32,
            .ffn_hidden_dim = 11008,
            .max_seq_len = 2048,
            .ffn_type = .gated_swi_glu,
            .norm_type = .rms_norm,
            .pos_encoding = .rope,
            .use_kv_quantization = true,
        };
        var eng = try engine.Engine.init(cfg, allocator);
        errdefer eng.deinit(allocator);

        return .{
            .config = cfg,
            .engine_ = eng,
        };
    }

    pub fn deinit(self: *ModelLoader, allocator: std.mem.Allocator) void {
        self.engine_.deinit(allocator);
    }
};

pub const InferencePort = struct {
    loader: ModelLoader,
    tokenizer_: tokenizer.SimpleTokenizer,

    pub fn init(model_path: []const u8, allocator: std.mem.Allocator) !InferencePort {
        var loader = try ModelLoader.load(model_path, allocator);
        var tok = tokenizer.SimpleTokenizer.init(allocator);

        return .{
            .loader = loader,
            .tokenizer_ = tok,
        };
    }

    pub fn deinit(self: *InferencePort, allocator: std.mem.Allocator) void {
        self.loader.deinit(allocator);
        self.tokenizer_.deinit();
    }

    pub fn forward(self: *InferencePort, input_ids: []const u32) ![]f32 {
        return self.loader.engine_.forward(input_ids);
    }

    pub fn generate(self: *InferencePort, prompt: []const u8, max_tokens: u32) ![]u8 {
        return self.loader.engine_.generate(&self.tokenizer_, prompt, max_tokens);
    }

    pub fn reset(self: *InferencePort) void {
        if (self.loader.engine_.kv_cache) |*cache| {
            cache.reset();
        }
    }

    pub fn getConfig(self: *const InferencePort) config.ModelConfig {
        return self.loader.config;
    }

    pub fn vocabSize(self: *const InferencePort) u32 {
        return self.loader.config.vocab_size;
    }
};

test "InferencePort init" {
    var port = try InferencePort.init("/tmp/model.gguf", std.testing.allocator);
    defer port.deinit(std.testing.allocator);
    try std.testing.expectEqual(port.vocabSize(), 32000);
}
