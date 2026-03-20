#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>

#include "config.h"
#include "types.h"
#include "memory.h"
#include "inference.h"
#include "gguf_reader.h"

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("  echo-core — Bare-Metal LLM Inference\n");
    printf("  AVX2 | mmap Zero-Copy | INT8 KV-Cache\n");
    printf("========================================\n\n");

    // --- Model path from argv or default ---
    std::string model_path = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf";
    if (argc > 1) model_path = argv[1];

    // --- Parse GGUF ---
    printf("[GGUF Loader]\n");
    printf("  File: %s\n", model_path.c_str());

    GGUFReader reader(model_path);

    int align = reader.alignment();
    printf("  Alignment: %d bytes\n", align);
    printf("  Tensor count: %zu\n", reader.tensors().size());
    printf("  Data offset: %lld\n", (long long)reader.data_offset());

    // Validate alignment — hard fail if corrupt, no fallback
    reader.assert_alignment(align);

    const ModelConfig& cfg = reader.config();
    printf("\n[Model Config]\n");
    printf("  vocab=%u  hidden=%u  heads=%u(%u kv)  head_dim=%u\n",
           cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("  layers=%u  ffn=%u  max_seq=%u\n", cfg.num_layers, cfg.ffn_hidden_dim, cfg.max_seq_len);

    // --- Load via mmap (zero-copy) ---
    printf("\n[Zero-Copy mmap]\n");
    InferenceEngine engine(cfg);
    engine.load_weights_from_gguf(reader, model_path);
    printf("  Weights mapped read-only from GGUF file\n");
    printf("  No memcpy — tensor data accessed directly from page cache\n\n");

    // --- Autoregressive generation demo ---
    printf("[Inference Demo — Autoregressive Generation]\n");
    const uint32_t generate_tokens = 8;
    uint32_t tokens[generate_tokens];
    tokens[0] = 42; // seed token

    float logits[256];
    auto t_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step = 0; step < generate_tokens; ++step) {
        engine.forward(tokens, step + 1, logits);

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

    return 0;
}
