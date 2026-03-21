#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <iostream>

#include "config.h"
#include "types.h"
#include "memory.h"
#include "inference.h"
#include "gguf_reader.h"
#include "tokenizer.h"

int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("  echo-core — Bare-Metal LLM Inference\n");
    printf("  AVX2 | mmap Zero-Copy | INT8 KV-Cache\n");
    printf("========================================\n\n");

    std::string model_path = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf";
    if (argc > 1) model_path = argv[1];

    // --- Parse GGUF ---
    printf("[GGUF Loader]\n");
    printf("  File: %s\n", model_path.c_str());

    GGUFReader reader(model_path);
    reader.assert_alignment(reader.alignment());

    ModelConfig cfg = reader.config();
    cfg.max_seq_len = 2048;

    printf("\n[Model Config]\n");
    printf("  vocab=%u  hidden=%u  heads=%u(%u kv)  head_dim=%u\n",
           cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    printf("  layers=%u  ffn=%u  max_seq=%u\n",
           cfg.num_layers, cfg.ffn_hidden_dim, cfg.max_seq_len);

    // --- Load weights (Q8_0/Q2_K/Q4_K → FP16 dequantization) ---
    printf("\n[Weight Loading]\n");
    InferenceEngine engine(cfg);
    engine.load_weights_from_gguf(reader, model_path);
    printf("  Weights dequantized to FP16\n\n");

    // --- Tokenizer ---
    printf("[Tokenizer]\n");
    SimpleTokenizer tokenizer(reader);

    // --- Tokenizer roundtrip test ---
    printf("\n=== Tokenizer Roundtrip Test ===\n\n");

    std::vector<std::string> test_inputs = {
        "Hello",
        "Hello world",
        "What is 2+2?",
        "Merhaba dunya",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (const auto& input : test_inputs) {
        std::vector<int> ids = tokenizer.encode(input);
        std::vector<int> ids_no_bos(ids.begin() + 1, ids.end());
        std::string decoded = tokenizer.decode(ids_no_bos);
        bool ok = (decoded == input);
        printf("  \"%s\" -> %zu tokens -> \"%s\" %s\n",
               input.c_str(), ids.size(), decoded.c_str(), ok ? "OK" : "FAIL");
    }

    // --- Single forward pass test ---
    printf("\n=== Inference Test ===\n\n");
    {
        std::string prompt = "Hello";
        std::vector<int> encoded = tokenizer.encode(prompt);
        printf("Prompt: \"%s\" -> %zu tokens\n", prompt.c_str(), encoded.size());

        std::vector<uint32_t> tokens(encoded.begin(), encoded.end());
        std::vector<float> logits(cfg.vocab_size);

        printf("Running forward pass...\n");
        fflush(stdout);

        auto t0 = std::chrono::high_resolution_clock::now();
        engine.forward(tokens.data(), static_cast<uint32_t>(tokens.size()), logits.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        printf("Forward pass OK (%.2f ms)\n", ms);

        uint32_t best = 0;
        for (uint32_t v = 1; v < cfg.vocab_size; ++v)
            if (logits[v] > logits[best]) best = v;

        printf("Next token: %d\n", best);
    }

    // --- Interactive REPL ---
    printf("\n=== Interactive ===\n");
    printf("Type text and press Enter (Ctrl+D to exit)\n\n");

    const uint32_t max_gen = 16; // CPU'da her token ~18sn suruyor
    std::vector<float> logits(cfg.vocab_size);
    std::string input;

    while (true) {
        printf("> ");
        fflush(stdout);
        if (!std::getline(std::cin, input)) break;
        if (input.empty()) continue;

        std::vector<int> encoded = tokenizer.encode(input);
        printf("  Tokens (%zu):", encoded.size());
        for (size_t i = 0; i < std::min(encoded.size(), (size_t)15); ++i)
            printf(" %d", encoded[i]);
        if (encoded.size() > 15) printf(" ...");
        printf("\n");

        std::vector<uint32_t> tokens(encoded.begin(), encoded.end());
        uint32_t n_prompt = static_cast<uint32_t>(tokens.size());
        engine.reset();

        auto t_start = std::chrono::high_resolution_clock::now();
        uint32_t generated = 0;

        try {
            for (uint32_t step = 0; step < n_prompt; ++step)
                engine.forward(tokens.data(), step + 1, logits.data());

            while (generated < max_gen) {
                uint32_t best = 0;
                for (uint32_t v = 1; v < cfg.vocab_size; ++v)
                    if (logits[v] > logits[best]) best = v;
                if (best == tokenizer.eos()) break;
                tokens.push_back(best);
                ++generated;
                if (tokens.size() >= cfg.max_seq_len) break;
                engine.forward(tokens.data(), static_cast<uint32_t>(tokens.size()), logits.data());
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "  Error: %s\n", e.what());
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::vector<int> gen_ids(tokens.begin() + n_prompt, tokens.end());
        std::string output = tokenizer.decode(gen_ids);

        printf("\n%s\n", output.c_str());
        if (generated > 0)
            printf("  [%u tokens, %.2f ms, %.2f ms/tok]\n\n",
                   generated, ms, ms / generated);
    }

    printf("\nDone.\n");
    return 0;
}
