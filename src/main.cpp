#include <cstdio>
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <execinfo.h>

#include "config.h"
#include "types.h"
#include "memory.h"
#include "inference.h"
#include "gguf_reader.h"
#include "tokenizer.h"

namespace {

bool g_debug_enabled = false;
const char* g_last_stage = "startup";

const char* ggml_type_name(GGMLType dtype) {
    switch (dtype) {
        case GGMLType::F32: return "F32";
        case GGMLType::F16: return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0";
        case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::IQ2_XXS: return "IQ2_XXS";
        case GGMLType::IQ2_XS: return "IQ2_XS";
        case GGMLType::I16: return "I16";
        case GGMLType::F64: return "F64";
        case GGMLType::IQ1_S: return "IQ1_S";
        case GGMLType::IQ4_NL: return "IQ4_NL";
        case GGMLType::IQ4_XS: return "IQ4_XS";
        case GGMLType::I8: return "I8";
        case GGMLType::I32: return "I32";
        case GGMLType::IQ2_S: return "IQ2_S";
        case GGMLType::IQ3_XXS: return "IQ3_XXS";
        case GGMLType::BF16: return "BF16";
        case GGMLType::Q4_0_4_4: return "Q4_0_4_4";
        case GGMLType::Q4_0_4_8: return "Q4_0_4_8";
        case GGMLType::Q4_0_8_8: return "Q4_0_8_8";
        case GGMLType::COUNT: return "COUNT";
    }
    return "UNKNOWN";
}

void set_stage(const char* stage) {
    g_last_stage = stage;
    if (g_debug_enabled) {
        std::fprintf(stderr, "[debug] stage=%s\n", g_last_stage);
        std::fflush(stderr);
    }
}

void fatal_signal_handler(int sig) {
    void* frames[64];
    int frame_count = ::backtrace(frames, 64);

    std::fprintf(stderr, "\n[Fatal Signal]\n");
    std::fprintf(stderr, "  signal=%d (%s)\n", sig, ::strsignal(sig));
    std::fprintf(stderr, "  last_stage=%s\n", g_last_stage);
    std::fprintf(stderr, "  backtrace:\n");
    ::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    std::fflush(stderr);
    std::_Exit(128 + sig);
}

void install_signal_handlers() {
    struct sigaction sa {};
    sa.sa_handler = fatal_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND;
    sigaction(SIGBUS, &sa, nullptr);
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
}

std::string metadata_string(const GGUFReader& reader, const std::string& key) {
    auto it = reader.metadata().find(key);
    if (it == reader.metadata().end()) return {};
    if (const auto* value = std::get_if<std::string>(&it->second))
        return *value;
    return {};
}

bool has_exact_tensor(const GGUFReader& reader, const std::string& name) {
    return reader.tensors().find(name) != reader.tensors().end();
}

void append_issue(std::vector<std::string>& issues, const std::string& issue) {
    issues.push_back(issue);
}

std::string build_compatibility_report(const GGUFReader& reader) {
    std::vector<std::string> issues;
    const auto architecture = metadata_string(reader, "general.architecture");

    if (!architecture.empty() && architecture != "llama" && architecture != "qwen2") {
        append_issue(issues,
                     "general.architecture=\"" + architecture +
                     "\". Engine mevcut haliyle yalnizca standart transformer benzeri "
                     "katman dizilimini bekliyor.");
    }

    for (const auto& [name, info] : reader.tensors()) {
        (void)info;
        if (name.find(".ssm_") != std::string::npos) {
            append_issue(issues,
                         "SSM tensorlari bulundu (ornek: \"" + name +
                         "\"). Bu inference engine SSM/Mamba benzeri dali hic uygulamiyor.");
            break;
        }
    }

    if (reader.metadata().count(architecture + ".full_attention_interval") > 0) {
        append_issue(issues,
                     "Model attention ile SSM'i karma kullaniyor "
                     "(metadata: \"" + architecture + ".full_attention_interval\").");
    }

    const TensorInfo* token_embd = reader.find_tensor_by_suffix("token_embd.weight");
    if (token_embd && token_embd->dtype != GGMLType::F16) {
        append_issue(issues,
                     "token_embd.weight dtype=" + std::string(ggml_type_name(token_embd->dtype)) +
                     ". Kod embedding lookup'ta quantized row stride'i hesaba katmiyor ve FP16 satir "
                     "varmis gibi dogrudan pointer arithmetic yapiyor.");
    }

    const TensorInfo* output_norm = reader.find_tensor_by_suffix("output_norm.weight");
    if (output_norm && output_norm->dtype != GGMLType::F16) {
        append_issue(issues,
                     "output_norm.weight dtype=" + std::string(ggml_type_name(output_norm->dtype)) +
                     ". Kod norm agirligini FP16 varsayiyor.");
    }

    const bool has_q = reader.find_tensor_by_suffix("attn_q.weight") != nullptr;
    const bool has_k = reader.find_tensor_by_suffix("attn_k.weight") != nullptr;
    const bool has_v = reader.find_tensor_by_suffix("attn_v.weight") != nullptr;
    const TensorInfo* fused_qkv = reader.find_tensor_by_suffix("attn_qkv.weight");
    if (fused_qkv && !(has_q && has_k && has_v)) {
        std::ostringstream oss;
        oss << "attn_qkv.weight bulundu (dtype=" << ggml_type_name(fused_qkv->dtype)
            << "), fakat ayri Q/K/V tensorlari yok. Kod fused QKV'yi ayirmadan ayni pointer'i "
            << "Q, K ve V icin kullaniyor.";
        append_issue(issues, oss.str());
    }

    const bool has_attn_output = reader.find_tensor_by_suffix("attn_output.weight") != nullptr;
    const bool has_attn_gate = reader.find_tensor_by_suffix("attn_gate.weight") != nullptr;
    if (!has_attn_output && has_attn_gate) {
        append_issue(issues,
                     "attn_output.weight yok, attn_gate.weight var. Kod bunu output projection yerine "
                     "gecirmeye calisiyor; semantik olarak ayni sey degil.");
    }

    if (reader.find_tensor_by_suffix("post_attention_norm.weight")) {
        append_issue(issues,
                     "post_attention_norm.weight bulundu. Kod ikinci norm olarak bunu kullanmiyor; "
                     "ilk attn_norm agirligini tekrar kullaniyor.");
    }

    if (!reader.find_tensor_by_suffix("output.weight")) {
        append_issue(issues,
                     "output.weight bulunamadi. Logits projection olmadan generation tamamlanamaz.");
    }

    if (token_embd && token_embd->dtype == GGMLType::Q2_K) {
        append_issue(issues,
                     "Q2_K kernel tarafinda somut bug var: block stride 86 yazilmis, GGUF hesaplamasi 84 "
                     "olmasi gerektigini soyluyor. Bu, mimari uyumsuzluk olmasa bile SIGBUS/SIGSEGV "
                     "uretebilir.");
    }

    if (issues.empty()) return {};

    std::ostringstream report;
    report << "Model compatibility check FAILED.\n";
    report << "Bu crash cozulmedigi icin tekrar tekrar ayni hata aliyorduk; cunku once modelin bu "
           << "engine tarafindan gercekten desteklenip desteklenmedigi dogrulanmadi.\n";
    for (size_t i = 0; i < issues.size(); ++i) {
        report << "  [" << (i + 1) << "] " << issues[i] << '\n';
    }
    report << "Engine calismayi durduruyor; aksi halde rastgele bellek okumalari ve SIGBUS ile devam etme riski var.\n";
    return report.str();
}

void print_debug_summary(const GGUFReader& reader) {
    std::fprintf(stderr, "\n[Debug Summary]\n");
    std::fprintf(stderr, "  architecture=%s\n",
                 metadata_string(reader, "general.architecture").c_str());
    std::fprintf(stderr, "  alignment=%d  data_offset=%lld  tensors=%zu  metadata=%zu\n",
                 reader.alignment(),
                 static_cast<long long>(reader.data_offset()),
                 reader.tensors().size(),
                 reader.metadata().size());

    std::unordered_map<std::string, size_t> dtype_counts;
    for (const auto& [name, info] : reader.tensors()) {
        (void)name;
        dtype_counts[ggml_type_name(info.dtype)]++;
    }

    std::vector<std::pair<std::string, size_t>> dtype_rows(dtype_counts.begin(), dtype_counts.end());
    std::sort(dtype_rows.begin(), dtype_rows.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::fprintf(stderr, "  dtype histogram:\n");
    for (const auto& [dtype, count] : dtype_rows) {
        std::fprintf(stderr, "    %s=%zu\n", dtype.c_str(), count);
    }

    const char* interesting[] = {
        "token_embd.weight",
        "output_norm.weight",
        "output.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_gate.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ssm_out.weight",
    };

    std::fprintf(stderr, "  selected tensors:\n");
    for (const char* tensor_name : interesting) {
        auto it = reader.tensors().find(tensor_name);
        if (it == reader.tensors().end()) {
            std::fprintf(stderr, "    %s -> missing\n", tensor_name);
            continue;
        }
        const auto& info = it->second;
        std::fprintf(stderr, "    %s -> dtype=%s offset=%llu shape=[",
                     tensor_name,
                     ggml_type_name(info.dtype),
                     static_cast<unsigned long long>(info.offset));
        for (size_t i = 0; i < info.shape.size(); ++i) {
            if (i != 0) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%llu", static_cast<unsigned long long>(info.shape[i]));
        }
        std::fprintf(stderr, "]\n");
    }
    std::fflush(stderr);
}

} // namespace

int main(int argc, char* argv[]) {
    install_signal_handlers();

    try {
        printf("========================================\n");
        printf("  echo-core — Bare-Metal LLM Inference\n");
        printf("  AVX2 | mmap Zero-Copy | INT8 KV-Cache\n");
        printf("========================================\n\n");

        std::string model_path = "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf";
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--debug") == 0) {
                g_debug_enabled = true;
            } else if ((std::strcmp(argv[i], "--model") == 0 || std::strcmp(argv[i], "-m") == 0)
                       && i + 1 < argc) {
                model_path = argv[++i];
            } else if (argv[i][0] != '-') {
                model_path = argv[i]; // positional fallback
            }
        }
        if (const char* env = std::getenv("ECHO_DEBUG")) {
            g_debug_enabled = std::strcmp(env, "0") != 0;
        }

        set_stage("parse_gguf");
        printf("[GGUF Loader]\n");
        printf("  File: %s\n", model_path.c_str());

        GGUFReader reader(model_path);
        reader.assert_alignment(reader.alignment());
        if (g_debug_enabled)
            print_debug_summary(reader);

        const std::string compatibility_report = build_compatibility_report(reader);
        if (!compatibility_report.empty()) {
            throw std::runtime_error(compatibility_report);
        }

        ModelConfig cfg = reader.config();
        cfg.max_seq_len = 2048;

        printf("\n[Model Config]\n");
        printf("  vocab=%u  hidden=%u  heads=%u(%u kv)  head_dim=%u\n",
               cfg.vocab_size, cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
        printf("  layers=%u  ffn=%u  max_seq=%u\n",
               cfg.num_layers, cfg.ffn_hidden_dim, cfg.max_seq_len);

        set_stage("load_weights");
        printf("\n[Weight Loading]\n");
        InferenceEngine engine(cfg);
        engine.load_weights_from_gguf(reader, model_path);
        printf("  Weights loaded (quantized, zero-copy mmap)\n\n");

        set_stage("tokenizer_init");
        printf("[Tokenizer]\n");
        SimpleTokenizer tokenizer(reader);

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

        set_stage("single_forward_test");
        printf("\n=== Inference Test ===\n\n");
        {
            std::string prompt = "Hello";
            std::vector<int> encoded = tokenizer.encode(prompt);
            printf("Prompt: \"%s\" -> %zu tokens\n", prompt.c_str(), encoded.size());

            std::vector<uint32_t> tokens(encoded.begin(), encoded.end());
            std::vector<float> logits(cfg.vocab_size);

            if (g_debug_enabled) {
                std::fprintf(stderr, "[debug] prompt token ids:");
                for (int token : encoded) std::fprintf(stderr, " %d", token);
                std::fprintf(stderr, "\n");
                std::fflush(stderr);
            }

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

        set_stage("interactive_repl");
        printf("\n=== Interactive ===\n");
        printf("Type text and press Enter (Ctrl+D to exit)\n\n");

        const uint32_t max_gen = 16;
        std::vector<float> logits(cfg.vocab_size);
        std::string input;

        while (true) {
            printf("> ");
            fflush(stdout);
            if (!std::getline(std::cin, input)) break;
            if (input.empty()) continue;

            std::vector<int> encoded = tokenizer.encode(input);
            printf("  Tokens (%zu):", encoded.size());
            for (size_t i = 0; i < std::min(encoded.size(), static_cast<size_t>(15)); ++i)
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[Error]\n%s\n", e.what());
        return 1;
    }
}
