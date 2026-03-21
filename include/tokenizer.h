#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "gguf_reader.h"

// --- Token metadata ---
struct TokenData {
    std::string text;
    float       score;   // SPM log-probability; 0.0f for BPE
    int32_t     type;    // 1=normal, 2=unknown, 3=control, 4=user_defined, 6=byte
};

// --- Simple greedy tokenizer from GGUF vocabulary ---
class SimpleTokenizer {
    std::vector<TokenData>               id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<std::pair<std::string, int>> sorted_tokens_; // longest-first
    uint32_t    bos_id;
    uint32_t    eos_id;
    std::string tokenizer_type; // "spm" or "bpe"

public:
    explicit SimpleTokenizer(const GGUFReader& reader);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    uint32_t bos() const { return bos_id; }
    uint32_t eos() const { return eos_id; }
    size_t   vocab_size() const { return id_to_token.size(); }
    const std::string& type() const { return tokenizer_type; }
};
