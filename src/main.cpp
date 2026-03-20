#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>

#include "config.h"
#include "types.h"
#include "memory.h"
#include "inference.h"

int main() {
    printf("========================================\n");
    printf("  echo-core — Bare-Metal LLM Inference\n");
    printf("  AVX2 | Model-Agnostic | INT8 KV-Cache\n");
    printf("========================================\n\n");

    // --- Hardware preset info ---
    printf("[Hardware Presets]\n");
    printf("  Intel i5-13500H:  TILE_K=%u  TILE_M=%u  L3_budget=%llu MB\n",
           Intel13500H_Tiles::TILE_K,
           Intel13500H_Tiles::TILE_M,
           (unsigned long long)(Intel13500H_Tiles::L3_BUDGET / (1024 * 1024)));
    printf("  AMD Ryzen 5 3600: TILE_K=%u  TILE_M=%u  L3_budget=%llu MB\n\n",
           AMD_Ryzen3600_Tiles::TILE_K,
           AMD_Ryzen3600_Tiles::TILE_M,
           (unsigned long long)(AMD_Ryzen3600_Tiles::L3_BUDGET / (1024 * 1024)));

    // --- Demo model configuration ---
    // A small generic transformer — no model names, pure config
    ModelConfig cfg{};
    cfg.vocab_size         = 256;
    cfg.hidden_dim         = 64;
    cfg.num_heads          = 4;
    cfg.num_kv_heads       = 4;
    cfg.head_dim           = 16;
    cfg.num_layers         = 4;
    cfg.ffn_hidden_dim     = 128;
    cfg.max_seq_len        = 128;
    cfg.ffn_type           = ModelConfig::FFNType::GatedSwiGLU;
    cfg.norm_type          = ModelConfig::NormType::RMSNorm;
    cfg.pos_encoding       = ModelConfig::PosEncoding::RoPE;
    cfg.use_kv_quantization = true;

    printf("[Model Config]\n");
    printf("  vocab=%u  hidden=%u  heads=%u(%u kv)  head_dim=%u\n",
           cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("  layers=%u  ffn=%u  max_seq=%u\n", cfg.num_layers, cfg.ffn_hidden_dim, cfg.max_seq_len);
    printf("  FFN=GatedSwiGLU  Norm=RMSNorm  Pos=RoPE  KV_quant=INT8\n\n");

    // --- Weight layout ---
    WeightLayout layout = WeightLayout::compute(cfg);
    printf("[Weight Layout]\n");
    printf("  Total: %zu bytes (%.2f MB)\n", layout.total_size, layout.total_size / (1024.0 * 1024.0));
    printf("  Per-layer: %zu bytes\n\n", layout.per_layer_size);

    // --- Allocate and initialize synthetic weights ---
    size_t total_elements = layout.total_size / sizeof(fp16_t);
    fp16_t* weights = new fp16_t[total_elements];

    // Simple pattern: small random-ish values
    for (size_t i = 0; i < total_elements; ++i) {
        float val = 0.02f * std::sin(static_cast<float>(i) * 0.1f);
        weights[i] = fp32_to_fp16(val);
    }

    // --- Create engine and load weights ---
    InferenceEngine engine(cfg);
    engine.load_weights(weights);

    // --- Autoregressive generation demo ---
    printf("[Inference Demo — Autoregressive Generation]\n");
    const uint32_t generate_tokens = 8;
    uint32_t tokens[generate_tokens];
    tokens[0] = 42; // seed token

    float logits[256];
    auto t_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step = 0; step < generate_tokens; ++step) {
        // Forward pass with all tokens up to current step
        engine.forward(tokens, step + 1, logits);

        // Greedy decode: pick argmax
        uint32_t best_token = 0;
        float best_score = logits[0];
        for (uint32_t v = 1; v < cfg.vocab_size; ++v) {
            if (logits[v] > best_score) {
                best_score = logits[v];
                best_token = v;
            }
        }

        if (step + 1 < generate_tokens) {
            tokens[step + 1] = best_token;
        }

        printf("  step %u: token=%u  logit=%.4f\n", step, best_token, best_score);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    printf("\n[Performance]\n");
    printf("  %u tokens in %.2f ms (%.2f ms/token)\n",
           generate_tokens, ms, ms / generate_tokens);
    printf("\nDone.\n");

    delete[] weights;
    return 0;
}
