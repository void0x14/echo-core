#include "gguf_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring>
#include <algorithm>
#include <cassert>

// ===========================================================================
// Constants
// ===========================================================================

static constexpr uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF" little-endian
static constexpr uint32_t GGUF_VERSION = 3;

// GGUF metadata value type IDs
static constexpr uint32_t GGUF_VAL_UINT8   = 0;
static constexpr uint32_t GGUF_VAL_INT8    = 1;
static constexpr uint32_t GGUF_VAL_UINT16  = 2;
static constexpr uint32_t GGUF_VAL_INT16   = 3;
static constexpr uint32_t GGUF_VAL_UINT32  = 4;
static constexpr uint32_t GGUF_VAL_INT32   = 5;
static constexpr uint32_t GGUF_VAL_FLOAT32 = 6;
static constexpr uint32_t GGUF_VAL_BOOL    = 7;
static constexpr uint32_t GGUF_VAL_STRING  = 8;
static constexpr uint32_t GGUF_VAL_ARRAY   = 9;
static constexpr uint32_t GGUF_VAL_UINT64  = 10;
static constexpr uint32_t GGUF_VAL_INT64   = 11;
static constexpr uint32_t GGUF_VAL_FLOAT64 = 12;

// ===========================================================================
// GGML type conversion
// ===========================================================================

GGMLType ggml_type_from_uint32(uint32_t v) {
    if (v < static_cast<uint32_t>(GGMLType::COUNT))
        return static_cast<GGMLType>(v);
    throw std::runtime_error("GGUF: unknown GGML type id " + std::to_string(v));
}

// ===========================================================================
// Block size helper for quantized types
// ===========================================================================

static uint64_t block_size_bytes(GGMLType dtype) {
    switch (dtype) {
        case GGMLType::F32:       return 4;
        case GGMLType::F16:       return 2;
        case GGMLType::BF16:      return 2;
        case GGMLType::F64:       return 8;
        case GGMLType::I8:        return 1;
        case GGMLType::I16:       return 2;
        case GGMLType::I32:       return 4;
        case GGMLType::Q4_0:      return 18;   // 2 + 16
        case GGMLType::Q4_1:      return 20;   // 2*2 + 16
        case GGMLType::Q5_0:      return 22;   // 2 + 4 + 16
        case GGMLType::Q5_1:      return 24;   // 2*2 + 4 + 16
        case GGMLType::Q8_0:      return 34;   // 2 + 32
        case GGMLType::Q8_1:      return 36;   // 4 + 4 + 32
        case GGMLType::Q2_K:      return 256 / 16 + 256 / 4 + 2 * 2; // 84
        case GGMLType::Q3_K:      return 256 / 8 + 256 / 4 + 12 + 2; // 110
        case GGMLType::Q4_K:      return 2 * 2 + 256 / 2 + 128 / 16 * 2; // 144
        case GGMLType::Q5_K:      return 2 * 2 + 256 / 8 * 5 + 128 / 16 * 2; // 176
        case GGMLType::Q6_K:      return 256 / 2 + 256 / 4 + 256 / 16 + 2; // 210
        case GGMLType::IQ2_XXS:   return 256 / 8 * 2 + 2; // 66
        case GGMLType::IQ2_XS:    return 256 / 8 * 2 + 256 / 64 + 2; // 70
        case GGMLType::IQ2_S:     return 256 / 8 * 2 + 256 / 32 + 2; // 74
        case GGMLType::IQ3_XXS:   return 256 / 8 * 3 + 256 / 32 + 2; // 102
        case GGMLType::IQ1_S:     return 256 / 8 + 256 / 32 + 2; // 42
        case GGMLType::IQ4_NL:    return 256 / 2 + 256 / 32 * 2; // 144
        case GGMLType::IQ4_XS:    return 256 / 2 + 256 / 64 * 2; // 136
        case GGMLType::Q4_0_4_4:  return 18;
        case GGMLType::Q4_0_4_8:  return 18;
        case GGMLType::Q4_0_8_8:  return 18;
        default: return 0;
    }
}

static constexpr uint64_t GGML_QUANT_BLOCK_SIZE = 32;

static uint64_t compute_tensor_byte_size(GGMLType dtype,
                                          const std::vector<uint64_t>& shape) {
    // Total elements across all dimensions
    uint64_t n_elements = 1;
    for (auto d : shape) n_elements *= d;

    switch (dtype) {
        case GGMLType::F32:  return n_elements * 4;
        case GGMLType::F16:  return n_elements * 2;
        case GGMLType::BF16: return n_elements * 2;
        case GGMLType::F64:  return n_elements * 8;
        case GGMLType::I8:   return n_elements;
        case GGMLType::I16:  return n_elements * 2;
        case GGMLType::I32:  return n_elements * 4;
        default: {
            // Quantized: block-based layout
            uint64_t bs = block_size_bytes(dtype);
            if (bs == 0)
                throw std::runtime_error("GGUF: unsupported dtype for size calc");
            // For quantized types, the "effective" elements are along the last dim
            // but the standard formula is: ceil(n_elements / block_size) * block_bytes
            // where block_size = GGML_QUANT_BLOCK_SIZE (32) for most types
            uint64_t n_blocks = (n_elements + GGML_QUANT_BLOCK_SIZE - 1) / GGML_QUANT_BLOCK_SIZE;
            return n_blocks * bs;
        }
    }
}

// ===========================================================================
// Low-level I/O
// ===========================================================================

void GGUFReader::read_exact(void* buf, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t remaining = n;
    while (remaining > 0) {
        ssize_t r = ::read(fd_, p, remaining);
        if (r <= 0)
            throw std::runtime_error("GGUF: read failed (EOF or I/O error)");
        p += r;
        remaining -= static_cast<size_t>(r);
    }
}

uint8_t GGUFReader::read_u8() {
    uint8_t v;
    read_exact(&v, 1);
    return v;
}

uint32_t GGUFReader::read_u32() {
    uint32_t v;
    read_exact(&v, 4);
    return v;
}

int32_t GGUFReader::read_i32() {
    int32_t v;
    read_exact(&v, 4);
    return v;
}

uint64_t GGUFReader::read_u64() {
    uint64_t v;
    read_exact(&v, 8);
    return v;
}

int64_t GGUFReader::read_i64() {
    int64_t v;
    read_exact(&v, 8);
    return v;
}

float GGUFReader::read_f32() {
    float v;
    read_exact(&v, 4);
    return v;
}

double GGUFReader::read_f64() {
    double v;
    read_exact(&v, 8);
    return v;
}

std::string GGUFReader::read_string() {
    uint64_t len = read_u64();
    std::string s(len, '\0');
    if (len > 0)
        read_exact(s.data(), len);
    return s;
}

GGUFValue GGUFReader::read_value(uint32_t type_id) {
    switch (type_id) {
        case GGUF_VAL_UINT8:   return static_cast<uint64_t>(read_u8());
        case GGUF_VAL_INT8:    return static_cast<int64_t>(static_cast<int8_t>(read_u8()));
        case GGUF_VAL_UINT16:  return static_cast<uint64_t>([this]() { uint16_t v; read_exact(&v, 2); return v; }());
        case GGUF_VAL_INT16:   return static_cast<int64_t>([this]() { int16_t v; read_exact(&v, 2); return v; }());
        case GGUF_VAL_UINT32:  return static_cast<uint64_t>(read_u32());
        case GGUF_VAL_INT32:   return static_cast<int64_t>(read_i32());
        case GGUF_VAL_FLOAT32: return static_cast<double>(read_f32());
        case GGUF_VAL_BOOL:    return static_cast<bool>(read_u8());
        case GGUF_VAL_STRING:  return read_string();
        case GGUF_VAL_UINT64:  return read_u64();
        case GGUF_VAL_INT64:   return read_i64();
        case GGUF_VAL_FLOAT64: return read_f64();
        case GGUF_VAL_ARRAY: {
            uint32_t elem_type = read_u32();
            uint64_t n_elems   = read_u64();

            switch (elem_type) {
                case GGUF_VAL_UINT8: {
                    std::vector<uint64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_u8();
                    return v;
                }
                case GGUF_VAL_INT8: {
                    std::vector<int64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i)
                        v[i] = static_cast<int64_t>(static_cast<int8_t>(read_u8()));
                    return v;
                }
                case GGUF_VAL_UINT16: {
                    std::vector<uint64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) {
                        uint16_t val; read_exact(&val, 2); v[i] = val;
                    }
                    return v;
                }
                case GGUF_VAL_INT16: {
                    std::vector<int64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) {
                        int16_t val; read_exact(&val, 2);
                        v[i] = static_cast<int64_t>(val);
                    }
                    return v;
                }
                case GGUF_VAL_UINT32: {
                    std::vector<uint64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_u32();
                    return v;
                }
                case GGUF_VAL_INT32: {
                    std::vector<int64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_i32();
                    return v;
                }
                case GGUF_VAL_FLOAT32: {
                    std::vector<double> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_f32();
                    return v;
                }
                case GGUF_VAL_BOOL: {
                    std::vector<bool> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = static_cast<bool>(read_u8());
                    return v;
                }
                case GGUF_VAL_STRING: {
                    std::vector<std::string> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_string();
                    return v;
                }
                case GGUF_VAL_UINT64: {
                    std::vector<uint64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_u64();
                    return v;
                }
                case GGUF_VAL_INT64: {
                    std::vector<int64_t> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_i64();
                    return v;
                }
                case GGUF_VAL_FLOAT64: {
                    std::vector<double> v(n_elems);
                    for (uint64_t i = 0; i < n_elems; ++i) v[i] = read_f64();
                    return v;
                }
                default:
                    throw std::runtime_error(
                        "GGUF: unsupported array element type " + std::to_string(elem_type));
            }
        }
        default:
            throw std::runtime_error("GGUF: unknown metadata value type " + std::to_string(type_id));
    }
}

// ===========================================================================
// Constructor — does all parsing
// ===========================================================================

GGUFReader::GGUFReader(const std::string& path)
    : fd_(-1), data_offset_(0)
{
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw std::runtime_error("GGUF: cannot open '" + path + "'");

    try {
        // --- Header ---
        uint32_t magic = read_u32();
        if (magic != GGUF_MAGIC) {
            char m[5];
            std::memcpy(m, &magic, 4);
            m[4] = '\0';
            throw std::runtime_error(
                std::string("GGUF: bad magic 0x") + std::to_string(magic) +
                " (\"" + m + "\") — expected \"GGUF\"");
        }

        uint32_t version = read_u32();
        if (version != GGUF_VERSION) {
            throw std::runtime_error(
                "GGUF: unsupported version " + std::to_string(version) + " — expected 3");
        }

        uint64_t tensor_count    = read_u64();
        uint64_t metadata_kv_count = read_u64();

        // --- Metadata ---
        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = read_string();
            uint32_t type_id = read_u32();
            GGUFValue val = read_value(type_id);
            metadata_.emplace(std::move(key), std::move(val));
        }

        // --- Detect model prefix ---
        std::string detected_prefix;
        for (const auto& [k, v] : metadata_) {
            static const char* target_suffixes[] = {
                ".context_length",
                ".embedding_length",
                ".block_count",
                ".feed_forward_length",
                ".attention.head_count",
                ".attention.head_count_kv",
            };
            for (const char* suf : target_suffixes) {
                std::string suffix(suf);
                if (k.size() > suffix.size() &&
                    k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    detected_prefix = k.substr(0, k.size() - suffix.size() + 1);
                    break;
                }
            }
            if (!detected_prefix.empty()) break;
        }
        model_prefix_ = detected_prefix;

        // --- Populate ModelConfig from metadata ---
        config_ = {};

        auto get_uint = [&](const std::string& suffix) -> uint64_t {
            auto it = metadata_.find(model_prefix_ + suffix);
            if (it == metadata_.end()) return 0;
            if (auto* p = std::get_if<uint64_t>(&it->second)) return *p;
            if (auto* p = std::get_if<int64_t>(&it->second)) return static_cast<uint64_t>(*p);
            return 0;
        };

        config_.max_seq_len   = static_cast<uint32_t>(get_uint("context_length"));
        config_.hidden_dim    = static_cast<uint32_t>(get_uint("embedding_length"));
        config_.num_layers    = static_cast<uint32_t>(get_uint("block_count"));
        config_.ffn_hidden_dim = static_cast<uint32_t>(get_uint("feed_forward_length"));
        config_.num_heads     = static_cast<uint32_t>(get_uint("attention.head_count"));

        uint64_t kv_heads = get_uint("attention.head_count_kv");
        config_.num_kv_heads = kv_heads ? static_cast<uint32_t>(kv_heads)
                                        : config_.num_heads;

        // head_dim from embedding_length / num_heads
        if (config_.num_heads > 0)
            config_.head_dim = config_.hidden_dim / config_.num_heads;

        // vocab_size from tokens array (set later after tokenizer parse)
        // or from metadata key if available
        {
            auto it = metadata_.find(model_prefix_ + "vocab_size");
            if (it == metadata_.end())
                it = metadata_.find("vocab_size");  // un-prefixed fallback
            if (it != metadata_.end()) {
                if (auto* p = std::get_if<uint64_t>(&it->second))
                    config_.vocab_size = static_cast<uint32_t>(*p);
                else if (auto* p = std::get_if<int64_t>(&it->second))
                    config_.vocab_size = static_cast<uint32_t>(*p);
            }
        }

        // FFN type detection
        {
            config_.ffn_type = ModelConfig::FFNType::Dense;
            std::string gate_key = model_prefix_ + "feed_forward_gate_proj.tensor_name";
            // Simpler: check if ffn_hidden_dim != hidden_dim and 3 FFN weight tensors exist
            // Default to GatedSwiGLU for modern models
            auto it = metadata_.find(model_prefix_ + "expert_feed_forward_length");
            // Also check for .feed_forward_gate_proj presence via tensor names
            // For now: if we have ffn info, assume GatedSwiGLU (most common)
            if (config_.ffn_hidden_dim > 0)
                config_.ffn_type = ModelConfig::FFNType::GatedSwiGLU;
        }

        // Norm: RMSNorm for GGUF models (virtually all modern models use it)
        config_.norm_type = ModelConfig::NormType::RMSNorm;

        // Position encoding: RoPE for GGUF models
        config_.pos_encoding = ModelConfig::PosEncoding::RoPE;

        // KV quantization: default off, can be set externally
        config_.use_kv_quantization = false;

        // --- Tokenizer ---
        auto tok_it = metadata_.find("tokenizer.ggml.tokens");
        if (tok_it != metadata_.end()) {
            if (auto* p = std::get_if<std::vector<std::string>>(&tok_it->second)) {
                tokens_ = std::move(*p);
                if (config_.vocab_size == 0)
                    config_.vocab_size = static_cast<uint32_t>(tokens_.size());
            }
        }

        // --- Tensors ---
        for (uint64_t i = 0; i < tensor_count; ++i) {
            std::string name = read_string();
            uint32_t n_dims  = read_u32();

            std::vector<uint64_t> shape(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                shape[d] = read_u64();
            }

            GGMLType dtype = ggml_type_from_uint32(read_u32());
            uint64_t offset = read_u64();

            uint64_t size = compute_tensor_byte_size(dtype, shape);

            tensors_.emplace(std::move(name), TensorInfo{offset, size, std::move(shape), dtype});
        }

        // Compute data_offset: current file position is end of tensor infos.
        // GGUF v3 spec: tensor data is aligned to 32 bytes from start of file.
        off_t cur = ::lseek(fd_, 0, SEEK_CUR);
        data_offset_ = (cur + 31) & ~static_cast<off_t>(31);

    } catch (...) {
        ::close(fd_);
        fd_ = -1;
        throw;
    }
}

GGUFReader::~GGUFReader() {
    if (fd_ >= 0)
        ::close(fd_);
}

// ===========================================================================
// Alignment access and validation
// ===========================================================================

int GGUFReader::alignment() const {
    auto it = metadata_.find("general.alignment");
    if (it != metadata_.end()) {
        if (auto* p = std::get_if<uint64_t>(&it->second))
            return static_cast<int>(*p);
        if (auto* p = std::get_if<int64_t>(&it->second))
            return static_cast<int>(*p);
    }
    return 32;
}

const TensorInfo* GGUFReader::find_tensor_by_suffix(const std::string& suffix) const {
    for (const auto& [name, info] : tensors_) {
        if (name.size() >= suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return &info;
        }
    }
    return nullptr;
}

void GGUFReader::assert_alignment(int align) const {
    for (const auto& [name, info] : tensors_) {
        if (info.offset % static_cast<uint64_t>(align) != 0) {
            throw std::runtime_error(
                "GGUF: tensor '" + name + "' offset " + std::to_string(info.offset) +
                " is not aligned to " + std::to_string(align) + " bytes — file is corrupt");
        }
    }
}

// ===========================================================================
// Lazy tensor load
// ===========================================================================

std::vector<uint8_t> GGUFReader::load_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("GGUF: tensor '" + name + "' not found");

    const TensorInfo& info = it->second;
    std::vector<uint8_t> data(info.size);

    // pread — thread-safe, doesn't change file offset
    size_t total = 0;
    while (total < info.size) {
        ssize_t r = ::pread64(fd_, data.data() + total,
                               info.size - total,
                               static_cast<off_t>(info.offset) + total);
        if (r <= 0)
            throw std::runtime_error(
                "GGUF: failed to read tensor '" + name + "' at offset " +
                std::to_string(info.offset));
        total += static_cast<size_t>(r);
    }

    return data;
}
