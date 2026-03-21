#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

// GPT-2 byte-to-unicode mapping: raw byte -> UTF-8 string
static const std::string& gpt2_byte_to_unicode(uint8_t b) {
    static std::string table[256];
    static bool initialized = false;
    if (!initialized) {
        // Printable ASCII mapped directly
        for (int i = 33; i <= 126; ++i) table[i] = std::string(1, (char)i);
        for (int i = 161; i <= 172; ++i) table[i] = std::string(1, (char)i);
        for (int i = 174; i <= 255; ++i) table[i] = std::string(1, (char)i);
        // Remaining bytes -> code points starting at 256
        int n = 0;
        for (int i = 0; i < 256; ++i) {
            if (table[i].empty()) {
                int cp = 256 + n++;
                if (cp < 0x80) {
                    table[i] = std::string(1, (char)cp);
                } else if (cp < 0x800) {
                    table[i].push_back((char)(0xC0 | (cp >> 6)));
                    table[i].push_back((char)(0x80 | (cp & 0x3F)));
                } else {
                    table[i].push_back((char)(0xE0 | (cp >> 12)));
                    table[i].push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
                    table[i].push_back((char)(0x80 | (cp & 0x3F)));
                }
            }
        }
        initialized = true;
    }
    return table[b];
}

// Reverse mapping: UTF-8 string -> raw byte
static int gpt2_unicode_to_byte(const std::string& s) {
    static int reverse[256]; // indexed by unicode code point offset
    static int table[768];   // full reverse table
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < 768; ++i) table[i] = -1;
        for (int b = 0; b < 256; ++b) {
            const std::string& u = gpt2_byte_to_unicode((uint8_t)b);
            // Decode UTF-8 to code point
            int cp = 0;
            if ((u[0] & 0x80) == 0) {
                cp = u[0];
            } else if ((u[0] & 0xE0) == 0xC0) {
                cp = ((u[0] & 0x1F) << 6) | (u[1] & 0x3F);
            } else if ((u[0] & 0xF0) == 0xE0) {
                cp = ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
            }
            if (cp >= 0 && cp < 768)
                table[cp] = b;
        }
        initialized = true;
    }
    // Decode input string to code point
    int cp = -1;
    if (!s.empty()) {
        unsigned char c0 = (unsigned char)s[0];
        if ((c0 & 0x80) == 0) {
            cp = c0;
        } else if ((c0 & 0xE0) == 0xC0 && s.size() >= 2) {
            cp = ((c0 & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
        } else if ((c0 & 0xF0) == 0xE0 && s.size() >= 3) {
            cp = ((c0 & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
        }
    }
    if (cp >= 0 && cp < 768 && table[cp] >= 0)
        return table[cp];
    return -1;
}

SimpleTokenizer::SimpleTokenizer(const GGUFReader& reader) {
    const auto& meta = reader.metadata();

    // --- tokenizer type ---
    auto model_it = meta.find("tokenizer.ggml.model");
    if (model_it != meta.end()) {
        if (auto* p = std::get_if<std::string>(&model_it->second)) {
            std::string m = *p;
            if (m == "llama" || m == "sp")
                tokenizer_type = "spm";
            else if (m == "gpt2" || m == "qwen2" || m == "qwen")
                tokenizer_type = "bpe";
            else {
                fprintf(stderr, "Warning: unknown tokenizer model '%s', defaulting to spm\n",
                        m.c_str());
                tokenizer_type = "spm";
            }
        }
    } else {
        fprintf(stderr, "Warning: tokenizer.ggml.model not found, defaulting to spm\n");
        tokenizer_type = "spm";
    }

    // --- tokens (moved by GGUFReader into reader.tokens()) ---
    const auto& tokens_vec = reader.tokens();
    if (tokens_vec.empty())
        throw std::runtime_error("tokenizer.ggml.tokens not found or empty");

    size_t vocab = tokens_vec.size();
    id_to_token.resize(vocab);
    for (size_t i = 0; i < vocab; ++i) {
        id_to_token[i].text  = tokens_vec[i];
        id_to_token[i].score = 0.0f;
        id_to_token[i].type  = 1; // default: normal
        token_to_id[tokens_vec[i]] = static_cast<int>(i);
    }

    // --- scores (optional, SPM models) ---
    auto sc_it = meta.find("tokenizer.ggml.scores");
    if (sc_it != meta.end()) {
        if (auto* p = std::get_if<std::vector<double>>(&sc_it->second)) {
            if (p->size() != vocab)
                throw std::runtime_error(
                    "tokenizer.ggml.scores size (" + std::to_string(p->size()) +
                    ") != vocab size (" + std::to_string(vocab) + ")");
            for (size_t i = 0; i < vocab; ++i)
                id_to_token[i].score = static_cast<float>((*p)[i]);
        }
    }

    // --- token types ---
    auto tt_it = meta.find("tokenizer.ggml.token_type");
    if (tt_it != meta.end()) {
        if (auto* p = std::get_if<std::vector<int64_t>>(&tt_it->second)) {
            if (p->size() != vocab)
                throw std::runtime_error(
                    "tokenizer.ggml.token_type size (" + std::to_string(p->size()) +
                    ") != vocab size (" + std::to_string(vocab) + ")");
            for (size_t i = 0; i < vocab; ++i)
                id_to_token[i].type = static_cast<int32_t>((*p)[i]);
        }
    }

    // --- BOS / EOS ---
    bos_id = 0;
    eos_id = 0;
    auto bos_it = meta.find("tokenizer.ggml.bos_token_id");
    if (bos_it != meta.end()) {
        if (auto* p = std::get_if<uint64_t>(&bos_it->second))
            bos_id = static_cast<uint32_t>(*p);
    }
    auto eos_it = meta.find("tokenizer.ggml.eos_token_id");
    if (eos_it != meta.end()) {
        if (auto* p = std::get_if<uint64_t>(&eos_it->second))
            eos_id = static_cast<uint32_t>(*p);
    }

    // --- sorted tokens (longest first) for greedy matching ---
    sorted_tokens_.reserve(token_to_id.size());
    for (const auto& kv : token_to_id)
        sorted_tokens_.emplace_back(kv.first, kv.second);
    std::sort(sorted_tokens_.begin(), sorted_tokens_.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    printf("  Tokenizer: %s  vocab=%zu  bos=%u  eos=%u\n",
           tokenizer_type.c_str(), vocab, bos_id, eos_id);
}

std::vector<int> SimpleTokenizer::encode(const std::string& text) const {
    std::vector<int> result;
    result.push_back(static_cast<int>(bos_id));

    if (text.empty()) return result;

    std::string normalized;

    if (tokenizer_type == "spm") {
        normalized.reserve(text.size() + 4);
        normalized += "\xe2\x96\x81"; // U+2581 ▁
        for (char c : text) {
            if (c == ' ')
                normalized += "\xe2\x96\x81";
            else
                normalized += c;
        }
    } else {
        normalized = text;
    }

    size_t pos = 0;
    while (pos < normalized.size()) {
        bool matched = false;

        for (const auto& [tok_text, tok_id] : sorted_tokens_) {
            if (tok_text.empty()) continue;
            size_t len = tok_text.size();
            if (pos + len > normalized.size()) continue;

            if (normalized.compare(pos, len, tok_text) == 0) {
                result.push_back(tok_id);
                pos += len;
                matched = true;
                break;
            }
        }

        if (!matched) {
            uint8_t byte_val = static_cast<uint8_t>(normalized[pos]);
            std::string byte_tok;

            if (tokenizer_type == "bpe") {
                byte_tok = gpt2_byte_to_unicode(byte_val);
            } else {
                static const char hex[] = "0123456789ABCDEF";
                char buf[7] = {'<', '0', 'x', hex[byte_val >> 4], hex[byte_val & 0x0F], '>', '\0'};
                byte_tok = buf;
            }

            auto it = token_to_id.find(byte_tok);
            if (it != token_to_id.end()) {
                result.push_back(it->second);
            } else {
                throw std::runtime_error(
                    "No token for byte at position " + std::to_string(pos));
            }
            ++pos;
        }
    }

    return result;
}

std::string SimpleTokenizer::decode(const std::vector<int>& ids) const {
    std::string result;

    for (int id : ids) {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token.size())
            continue;

        // Skip BOS/EOS tokens
        if (static_cast<uint32_t>(id) == bos_id || static_cast<uint32_t>(id) == eos_id)
            continue;

        const TokenData& td = id_to_token[id];

        // Skip control tokens
        if (td.type == 3)
            continue;

        if (tokenizer_type == "bpe") {
            // Check if this is a single-byte-mapped token (1-3 UTF-8 bytes)
            // Only try reverse mapping for tokens that look like single characters
            bool is_byte_token = false;
            if (!td.text.empty()) {
                unsigned char c0 = (unsigned char)td.text[0];
                int expected_len = 0;
                if ((c0 & 0x80) == 0) expected_len = 1;
                else if ((c0 & 0xE0) == 0xC0) expected_len = 2;
                else if ((c0 & 0xF0) == 0xE0) expected_len = 3;

                if (expected_len > 0 && (int)td.text.size() == expected_len) {
                    int byte_val = gpt2_unicode_to_byte(td.text);
                    if (byte_val >= 0) {
                        result += static_cast<char>(byte_val);
                        is_byte_token = true;
                    }
                }
            }
            if (!is_byte_token) {
                result += td.text;
            }
        } else {
            result += td.text;
        }
    }

    // SPM post-process: replace ▁ (U+2581) with space, trim leading space
    if (tokenizer_type == "spm") {
        std::string cleaned;
        cleaned.reserve(result.size());
        for (size_t i = 0; i < result.size(); ) {
            if (i + 2 < result.size() &&
                static_cast<unsigned char>(result[i]) == 0xE2 &&
                static_cast<unsigned char>(result[i+1]) == 0x96 &&
                static_cast<unsigned char>(result[i+2]) == 0x81) {
                cleaned += ' ';
                i += 3;
            } else {
                cleaned += result[i];
                ++i;
            }
        }
        if (!cleaned.empty() && cleaned[0] == ' ')
            cleaned.erase(0, 1);
        return cleaned;
    }

    return result;
}
