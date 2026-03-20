#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <stdexcept>

#include "config.h"

// --- GGML tensor data types ---
enum class GGMLType : uint32_t {
    F32       = 0,
    F16       = 1,
    Q4_0      = 2,
    Q4_1      = 3,
    Q5_0      = 4,
    Q5_1      = 5,
    Q8_0      = 6,
    Q8_1      = 7,
    Q2_K      = 8,
    Q3_K      = 9,
    Q4_K      = 10,
    Q5_K      = 11,
    Q6_K      = 12,
    IQ2_XXS   = 13,
    IQ2_XS    = 14,
    I16       = 15,
    F64       = 16,
    IQ1_S     = 17,
    IQ4_NL    = 18,
    IQ4_XS    = 19,
    I8        = 20,
    I32       = 21,
    IQ2_S     = 22,
    IQ3_XXS   = 23,
    BF16      = 24,
    Q4_0_4_4  = 25,
    Q4_0_4_8  = 26,
    Q4_0_8_8  = 27,
    COUNT,
};

GGMLType ggml_type_from_uint32(uint32_t v);

// --- Tensor info (no data loaded) ---
struct TensorInfo {
    uint64_t              offset;   // byte offset in file
    uint64_t              size;     // byte size in file
    std::vector<uint64_t> shape;    // dimensions
    GGMLType              dtype;
};

// --- GGUF metadata value types ---
using GGUFValue = std::variant<
    uint64_t,
    int64_t,
    double,
    bool,
    std::string,
    std::vector<uint64_t>,
    std::vector<int64_t>,
    std::vector<double>,
    std::vector<bool>,
    std::vector<std::string>
>;

// --- GGUF v3 file reader ---
class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);
    ~GGUFReader();

    const ModelConfig& config() const { return config_; }
    const std::vector<std::string>& tokens() const { return tokens_; }
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }

    // Lazy load: reads raw bytes from file on demand
    std::vector<uint8_t> load_tensor(const std::string& name) const;

    // Raw metadata access
    const std::unordered_map<std::string, GGUFValue>& metadata() const { return metadata_; }

private:
    // Low-level reads
    void read_exact(void* buf, size_t n);
    uint8_t  read_u8();
    uint32_t read_u32();
    int32_t  read_i32();
    uint64_t read_u64();
    int64_t  read_i64();
    float    read_f32();
    double   read_f64();
    std::string read_string();
    GGUFValue read_value(uint32_t type_id);

    int fd_;
    off_t data_offset_;  // start of tensor data (32-byte aligned)
    std::string model_prefix_;

    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    ModelConfig config_;
};
