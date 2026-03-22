# Fused Dequant+Matvec Kernel Integration Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [`) syntax for tracking.

**Goal:** Enable quantized matvec kernels (Q8_0/Q4_K/Q2_K) by removing eager dequant, implementing matvec_d dispatch, and fixing weight pointer resolution.

**Architecture:** Keep quantized weights in mmap buffer. Add dtype + raw pointer maps keyed by layout_offset. Route all weight matvecs through matvec_d() which dispatches to fused kernels based on dtype.

**Tech Stack:** C++17, AVX2 intrinsics, mmap, GGUF format

---

## File Structure

- `include/inference.h` — Add `weight_ptr_map_` member, `resolve_weight_ptr()` declaration
- `src/inference.cpp` — Remove eager dequant, add populate loops, implement matvec_d + resolve_weight_ptr, update call sites, fix VLA

No new files created.

---

### Task 1: Add new members to inference.h

**Files:**
- Modify: `include/inference.h:37-38`

- [ ] **Step 1: Add weight_ptr_map_ member**

After line 38 (`weight_dtype_`), add:

```cpp
    std::unordered_map<size_t, const void*> weight_ptr_map_; // layout_offset → mmap raw ptr
```

- [ ] **Step 2: Add resolve_weight_ptr() declaration**

After the `has_weight()` method (line 69), add:

```cpp
    const void* resolve_weight_ptr(size_t layout_offset) const {
        auto it = weight_ptr_map_.find(layout_offset);
        if (it != weight_ptr_map_.end()) return it->second;
        throw std::runtime_error(
            "resolve_weight_ptr: no weight at offset " +
            std::to_string(layout_offset));
    }
```

---

### Task 2: Remove eager dequant from load_weights_from_gguf

**Files:**
- Modify: `src/inference.cpp:188-212`

- [ ] **Step 1: Delete lines 188-212**

Remove the entire eager dequant block:
```cpp
    // --- Dequantize all weights to FP16 ---    std::unordered_map<size_t, const TensorInfo*> gguf_to_info;
    for (const auto& [name, info] : reader.tensors()) {        gguf_to_info[static_cast<size_t>(info.offset)] = &info;    }
    AlignedMemoryPool fp16_pool(layout_.total_size);    for (const auto& [layout_off, gguf_off] : gguf_offset_map_) {        auto it = gguf_to_info.find(gguf_off);
        if (it == gguf_to_info.end()) continue;        const TensorInfo* ti = it->second;        size_t n_elements = 1;
        for (auto d : ti->shape) n_elements *= d;        const void* src = weight_pool_.at<void>(gguf_off);        fp16_t* dst = fp16_pool.at<fp16_t>(layout_off);        switch (ti->dtype) {
            case GGMLType::Q8_0: dequantize_q8_0_to_fp16(src, dst, n_elements); break;
            case GGMLType::Q4_K: dequantize_q4_K_to_fp16(src, dst, n_elements); break;
            case GGMLType::Q2_K: dequantize_q2_K_to_fp16(src, dst, n_elements); break;
            case GGMLType::F16: std::memcpy(dst, src, n_elements * sizeof(fp16_t)); break;
            case GGMLType::F32: { fp32_to_fp16_row(static_cast<const float*>(src), dst, n_elements); break; }
            default: break;        }    }
    weight_pool_ = std::move(fp16_pool);    gguf_offset_map_.clear();
```

---

### Task 3: Add dtype + pointer map population

**Files:**
- Modify: `src/inference.cpp` (after the existing tensor mapping loops, before where line 188 was)

- [ ] **Step 1: Add populate loop**

Insert after the output projection mapping block (after line 186, before where eager dequant was):

```cpp
    // Populate weight_dtype_ and weight_ptr_map_ (layout_offset keys)
    std::unordered_map<size_t, uint32_t> gguf_off_to_dtype;
    for (const auto& [name, info] : reader.tensors()) {
        gguf_off_to_dtype[static_cast<size_t>(info.offset)] =
            static_cast<uint32_t>(info.dtype);
    }
    for (const auto& [layout_off, gguf_off] : gguf_offset_map_) {
        auto it = gguf_off_to_dtype.find(gguf_off);
        if (it == gguf_off_to_dtype.end())
            throw std::runtime_error("load_weights: dtype not found for gguf_off="
                                     + std::to_string(gguf_off));
        weight_dtype_[layout_off]   = it->second;
        weight_ptr_map_[layout_off] = weight_pool_.at<void>(gguf_off);
    }
```

---

### Task 4: Implement matvec_d()

**Files:**
- Modify: `src/inference.cpp` (after the existing `matvec()` method, around line 527)

- [ ] **Step 1: Add matvec_d implementation**

```cpp
void InferenceEngine::matvec_d(const float* input, float* output,
                                size_t layout_offset, uint32_t rows, uint32_t cols) {
    auto it = weight_dtype_.find(layout_offset);
    if (it == weight_dtype_.end()) [[unlikely]]
        throw std::runtime_error("matvec_d: no dtype for offset " +
                                 std::to_string(layout_offset));
    const void* w = resolve_weight_ptr(layout_offset);
    switch (static_cast<GGMLType>(it->second)) {
        case GGML_TYPE_Q8_0: matvec_q8_0(w, input, output, rows, cols); break;
        case GGML_TYPE_Q4_K: matvec_q4_K(w, input, output, rows, cols); break;
        case GGML_TYPE_Q2_K: matvec_q2_K(w, input, output, rows, cols); break;
        case GGML_TYPE_F16:
            matvec_dispatch(static_cast<const fp16_t*>(w),
                            input, output, rows, cols, config_);
            break;
        default:
            throw std::runtime_error("matvec_d: unsupported dtype " +
                                     std::to_string(it->second));
    }
}
```

---

### Task 5: Update call sites — forward()

**Files:**
- Modify: `src/inference.cpp:238-241` (output projection in forward())

- [ ] **Step 1: Replace output projection matvec**

Change:
```cpp
    if (has_weight(layout_.output_proj_offset)) {
        const fp16_t* out_w = resolve_weight<fp16_t>(layout_.output_proj_offset);
        std::memset(logits, 0, vocab * sizeof(float));
        matvec(hidden_state_, logits, out_w, vocab, hidden);
    }
```
To:
```cpp
    if (has_weight(layout_.output_proj_offset)) {
        std::memset(logits, 0, vocab * sizeof(float));
        matvec_d(hidden_state_, logits, layout_.output_proj_offset, vocab, hidden);
    }
```

---

### Task 6: Update call sites — attention()

**Files:**
- Modify: `src/inference.cpp:300-308` (Q/K/V projections)
- Modify: `src/inference.cpp:398-401` (O projection)

- [ ] **Step 1: Replace Q/K/V projection matvecs**

Change:
```cpp
    const fp16_t* W_q = resolve_weight<fp16_t>(layer_base + layout_.q_proj_offset);
    const fp16_t* W_k = resolve_weight<fp16_t>(layer_base + layout_.k_proj_offset);
    const fp16_t* W_v = resolve_weight<fp16_t>(layer_base + layout_.v_proj_offset);

    if (W_q && W_k && W_v) {
        matvec(input, q_proj_, W_q, hidden, hidden);
        matvec(input, k_proj_, W_k, kv_dim,  hidden);
        matvec(input, v_proj_, W_v, kv_dim,  hidden);
    }
```
To:
```cpp
    if (has_weight(layer_base + layout_.q_proj_offset) &&
        has_weight(layer_base + layout_.k_proj_offset) &&
        has_weight(layer_base + layout_.v_proj_offset)) {
        matvec_d(input, q_proj_, layer_base + layout_.q_proj_offset, hidden, hidden);
        matvec_d(input, k_proj_, layer_base + layout_.k_proj_offset, kv_dim,  hidden);
        matvec_d(input, v_proj_, layer_base + layout_.v_proj_offset, kv_dim,  hidden);
    }
```

- [ ] **Step 2: Replace O projection matvec**

Change:
```cpp
    const fp16_t* W_o = resolve_weight<fp16_t>(layer_base + layout_.o_proj_offset);
    std::memset(output, 0, hidden * sizeof(float));
    if (W_o)
        matvec(attn_out_, output, W_o, hidden, hidden);
```
To:
```cpp
    std::memset(output, 0, hidden * sizeof(float));
    if (has_weight(layer_base + layout_.o_proj_offset))
        matvec_d(attn_out_, output, layer_base + layout_.o_proj_offset, hidden, hidden);
```

---

### Task 7: Update call sites — ffn()

**Files:**
- Modify: `src/inference.cpp:411-413` (Dense FFN weights)
- Modify: `src/inference.cpp:420, 425` (Dense FFN matvecs)
- Modify: `src/inference.cpp:431-433` (Gated FFN weights)
- Modify: `src/inference.cpp:438-439` (Gated FFN matvecs)
- Modify: `src/inference.cpp:449` (Gated FFN down matvec)
- Modify: `src/inference.cpp:463-464` (GatedGeLU matvecs)
- Modify: `src/inference.cpp:476` (GatedGeLU down matvec)

- [ ] **Step 1: Replace Dense FFN weight resolution + matvecs**

Change:
```cpp
    const fp16_t* W1 = resolve_weight<fp16_t>(current_layer_base_ + layout_.ffn_weight1_offset);
    const fp16_t* W2 = resolve_weight<fp16_t>(current_layer_base_ + layout_.ffn_weight2_offset);

    if (!W1 || !W2) return;

    switch (config_.ffn_type) {
        case ModelConfig::FFNType::Dense:
            // y = W2 * relu(W1 * x)
            std::memset(ffn_scratch_, 0, ffn_h * sizeof(float));
            matvec(input, ffn_scratch_, W1, ffn_h, hidden);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                ffn_scratch_[i] = (ffn_scratch_[i] > 0.0f) ? ffn_scratch_[i] : 0.0f; // ReLU
            }
            std::memset(output, 0, hidden * sizeof(float));
            matvec(ffn_scratch_, output, W2, hidden, ffn_h);
            break;
```
To:
```cpp
    if (!has_weight(current_layer_base_ + layout_.ffn_weight1_offset) ||
        !has_weight(current_layer_base_ + layout_.ffn_weight2_offset)) return;

    switch (config_.ffn_type) {
        case ModelConfig::FFNType::Dense:
            std::memset(ffn_scratch_, 0, ffn_h * sizeof(float));
            matvec_d(input, ffn_scratch_, current_layer_base_ + layout_.ffn_weight1_offset, ffn_h, hidden);
            for (uint32_t i = 0; i < ffn_h; ++i) {
                ffn_scratch_[i] = (ffn_scratch_[i] > 0.0f) ? ffn_scratch_[i] : 0.0f;
            }
            std::memset(output, 0, hidden * sizeof(float));
            matvec_d(ffn_scratch_, output, current_layer_base_ + layout_.ffn_weight2_offset, hidden, ffn_h);
            break;
```

- [ ] **Step 2: Replace GatedSwiGLU weight resolution + matvecs**

Change:
```cpp
        case ModelConfig::FFNType::GatedSwiGLU: {
            const fp16_t* W_gate = W1;
            const fp16_t* W_up   = W2;
            const fp16_t* W_down = resolve_weight<fp16_t>(
                current_layer_base_ + layout_.ffn_weight3_offset);
            if (!W_down) break;

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec(input, ffn_gate_buf_, W_gate, ffn_h, hidden);
            matvec(input, ffn_up_buf_, W_up, ffn_h, hidden);
```
To:
```cpp
        case ModelConfig::FFNType::GatedSwiGLU: {
            if (!has_weight(current_layer_base_ + layout_.ffn_weight3_offset)) break;

            std::memset(ffn_gate_buf_, 0, ffn_h * sizeof(float));
            std::memset(ffn_up_buf_, 0, ffn_h * sizeof(float));
            matvec_d(input, ffn_gate_buf_, current_layer_base_ + layout_.ffn_weight1_offset, ffn_h, hidden);
            matvec_d(input, ffn_up_buf_, current_layer_base_ + layout_.ffn_weight2_offset, ffn_h, hidden);
```

Also change the down projection:
```cpp
            std::memset(output, 0, hidden * sizeof(float));
            matvec(ffn_gate_buf_, output, W_down, hidden, ffn_h);
```
To:
```cpp
            std::memset(output, 0, hidden * sizeof(float));
            matvec_d(ffn_gate_buf_, output, current_layer_base_ + layout_.ffn_weight3_offset, hidden, ffn_h);
```

- [ ] **Step 3: Replace GatedGeLU weight resolution + matvecs**

Same pattern as SwiGLU — replace `W_gate/W_up/W_down` + `matvec()` with `matvec_d()` calls.

---

### Task 8: Fix VLA in norm()

**Files:**
- Modify: `src/inference.cpp:486`

- [ ] **Step 1: Replace VLA with std::vector**

Change:
```cpp
    float scale_buf[hidden];
```
To:
```cpp
    std::vector<float> scale_buf(hidden);
```

Add `#include <vector>` at the top of inference.cpp if not already present.

---

### Task 9: Build and verify

**Files:**
- None (verification only)

- [ ] **Step 1: Build the project**

```bash
cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core
cmake --build build/ 2>&1
```

Expected: Clean build with no errors or warnings.

- [ ] **Step 2: Run verification grep**

```bash
grep -rn "matvec_q8_0\|matvec_q4_K\|matvec_q2_K" src/ include/ \
  --include="*.cpp" --include="*.h" \
  | grep -v "^src/kernels/matvec.cpp" \
  | grep -v "^include/kernels/matvec.h"
```

Expected: Lines from `src/inference.cpp` showing `matvec_d()` switch cases calling the quantized kernels. No declarations or definitions outside matvec files.

- [ ] **Step 3: Run existing tests**

```bash
cd build && ctest --output-on-failure 2>&1
```

Expected: All tests pass.
