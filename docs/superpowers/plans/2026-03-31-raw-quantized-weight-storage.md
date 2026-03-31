# Raw Quantized Weight Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store quantized model weights in their original compressed format instead of dequantizing to FP16, reducing RAM usage from 87.8 GB to ~4.5 GB for Q8_0 models.

**Architecture:** Modify memory allocation to calculate per-tensor quantized byte sizes, store raw GGUF tensor data directly, and use existing quantized matvec kernels that dequantize on-the-fly during inference.

**Tech Stack:** Zig 0.16, GGUF format, GGML quantization types (Q8_0, Q4_K, Q6_K, Q2_K, etc.)

---

## Pre-Implementation Analysis

### Current Problem
- 4.16 GB Q8_0 model → load dequantizes to FP16 → 87.8 GB allocation → OOM
- `memory.zig:WeightLayout.compute()` assumes all weights are FP16 (`sizeof_fp16 = 2`)
- `inference.zig:loadTensorIfPresent()` dequantizes tensors before storage

### Existing Infrastructure (Already Present!)
- `inference.zig:computeTensorByteSize()` - correctly calculates quantized bytes
- `matvec.zig:matvecDispatchQuant()` - already dispatches to quantized kernels
- `engine.zig:weight_dtypes[]` - already tracks per-tensor dtypes
- `matvec.zig:matvecQ80/Q4K/Q2K` - already read quantized blocks directly

### Files to Modify

| File | Lines | Responsibility |
|------|-------|----------------|
| `memory.zig` | ~50 | Dtype-aware byte size calculation |
| `inference.zig` | ~30 | Raw byte storage, remove dequantize |
| `engine.zig` | ~10 | Verify offset calculations |
| `matvec.zig` | ~5 | Verify kernel dispatch |

---

## Task 1: Add Quantized Byte Calculation to memory.zig

**Files:**
- Modify: `echo-core-zig/src/core/memory.zig`

- [ ] **Step 1: Add dtype parameter to WeightLayout.compute()**

Add `reader: *const gguf.Reader` parameter to access tensor dtype information:

```zig
pub fn compute(config_: anytype, reader: *const gguf.Reader, allocator: std.mem.Allocator) !WeightLayout {
```

- [ ] **Step 2: Add computeTensorQuantizedBytes helper function**

Insert after imports (around line 10):

```zig
/// Compute byte size for a tensor given its shape and dtype
fn computeTensorQuantizedBytes(shape: []const u64, dtype: gguf.GGMLType) usize {
    var n_elements: u64 = 1;
    for (shape) |dim| n_elements *= dim;
    
    return switch (dtype) {
        .f32 => n_elements * 4,
        .f16, .bf16 => n_elements * 2,
        .f64 => n_elements * 8,
        .i8 => n_elements,
        .i16 => n_elements * 2,
        .i32 => n_elements * 4,
        else => blk: {
            // Quantized types: calculate blocks
            const block_bytes = blockSizeBytes(dtype);
            const block_elems = blockElements(dtype);
            if (block_bytes == 0 or block_elems == 0) unreachable;
            const n_blocks = (n_elements + block_elems - 1) / block_elems;
            break :blk n_blocks * block_bytes;
        },
    };
}

fn blockSizeBytes(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .q4_0 => 18,    // 32 elements, 16 bytes quants + 2 bytes scale
        .q4_1 => 20,    // 32 elements, 16 bytes quants + 2 bytes scale + 2 bytes min
        .q5_0 => 22,    // 32 elements, 20 bytes quants + 2 bytes scale
        .q5_1 => 24,    // 32 elements, 20 bytes quants + 2 bytes scale + 2 bytes min
        .q8_0 => 34,    // 32 elements, 32 bytes quants + 2 bytes scale
        .q8_1 => 36,    // 32 elements, 32 bytes quants + 2 bytes scale + 2 bytes min
        .q2_k => 84,    // 256 elements, super-block
        .q3_k => 110,   // 256 elements, super-block
        .q4_k => 144,   // 256 elements, super-block
        .q5_k => 176,   // 256 elements, super-block
        .q6_k => 210,   // 256 elements, super-block
        .q8_k => 34,    // 32 elements (note: different from Q8_0)
        .iq2_xxs => 66, .iq2_xs => 74, .iq2_s => 74,
        .iq3_xxs => 102, .iq3_s => 109,
        .iq1_s => 42, .iq1_m => 58,
        .iq4_nl => 144, .iq4_xs => 136,
        else => 0,
    };
}

fn blockElements(dtype: gguf.GGMLType) u64 {
    return switch (dtype) {
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1, .q8_k => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => 256,
        .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
        .iq1_s, .iq1_m, .iq4_nl, .iq4_xs => 256,
        else => 1,
    };
}
```

- [ ] **Step 3: Modify tensor size calculations to use actual dtype**

Replace the hardcoded `sizeof_fp16` multiplication with dynamic calculation. For token embedding:

```zig
// OLD: layout.token_embedding_size = vocab * hidden * sizeof_fp16;
// NEW: Look up actual dtype from GGUF
const token_embd_name = "token_embd.weight";
const token_embd_bytes = if (reader.tensors.get(token_embd_name)) |tensor_info|
    computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype)
else
    vocab * hidden * sizeof_fp16;  // fallback
layout.token_embedding_size = token_embd_bytes;
```

For per-layer attention weights, calculate based on actual tensor presence:

```zig
// For each layer, check what dtype the tensors actually use
var layer_offset: usize = layout.token_embedding_size;
for (0..config_.num_layers) |layer_idx| {
    // attn_q.weight dtype lookup
    var tensor_name_buf: [64]u8 = undefined;
    const q_name = std.fmt.bufPrint(&tensor_name_buf, "blk.{d}.attn_q.weight", .{layer_idx}) catch continue;
    
    const q_bytes = if (reader.tensors.get(q_name)) |tensor_info|
        computeTensorQuantizedBytes(tensor_info.shape, tensor_info.dtype)
    else
        hidden * q_dim * sizeof_fp16;
    
    // Similar for k_proj, v_proj, o_proj...
    // Accumulate offsets based on actual sizes
}
```

- [ ] **Step 4: Update total size calculation**

Replace `layout.total_size` calculation at end of function:

```zig
// OLD: layout.total_size = layout.ssm_region_offset + ssm_region_size;
// NEW: Sum all actual calculated sizes
layout.total_size = layout.ssm_region_offset + ssm_region_size;
layout.raw_pool_size = layout.total_size;
```

- [ ] **Step 5: Test compilation**

Run: `cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig && zig build`

Expected: SUCCESS (no compile errors)

---

## Task 2: Store Raw Bytes in inference.zig

**Files:**
- Modify: `echo-core-zig/src/ports/inference.zig`

- [ ] **Step 1: Modify loadTensorIfPresent to store raw bytes**

Change the function (around line 316) to store raw quantized data:

```zig
fn loadTensorIfPresent(
    reader: *const gguf.Reader,
    suffix: []const u8,
    dst_bytes: []u8,
    dtype_slot_index: usize,
    weight_dtypes: []gguf.GGMLType,
    allocator: std.mem.Allocator,
) !bool {
    if (findTensorNameBySuffix(reader, suffix)) |tensor_name| {
        const info = reader.findTensorBySuffix(suffix).?;

        // Store the dtype for this tensor slot
        weight_dtypes[dtype_slot_index] = info.dtype;

        // Compute actual byte size from tensor shape and dtype
        const actual_bytes = computeTensorByteSize(info.dtype, info.shape);

        // Verify destination can hold the data
        if (actual_bytes > dst_bytes.len) {
            std.debug.print("WARN: tensor '{s}' needs {d} bytes but only {d} allocated\n", .{ tensor_name, actual_bytes, dst_bytes.len });
            // Still load what fits
        }

        // Load raw tensor data directly - NO DEQUANTIZE
        const bytes_to_load = @min(actual_bytes, dst_bytes.len);
        try reader.loadTensorInto(tensor_name, dst_bytes[0..bytes_to_load]);

        return true;
    }
    return false;
}
```

- [ ] **Step 2: Remove copyTensorToFp16 usage**

Delete or comment out calls to `copyTensorToFp16()` in `loadTensorIfPresent()`. The raw bytes are now stored directly.

- [ ] **Step 3: Update loadWeightsFromReader slot calculations**

Ensure slot indices align with engine.zig expectations. The slot layout must be:
- Slot 0: token_embd
- Slots 1-11 per layer: attn_norm, q_proj, k_proj, v_proj, o_proj, ffn_norm, ffn_w1, ffn_w2, ffn_w3, attn_q_norm, attn_k_norm
- Slots 12-19 per layer (if SSM): ssm_out, ssm_x, ssm_dt, ssm_A, ssm_B, ssm_C, ssm_D, ssm_conv1d
- Final slots: final_norm, output_proj

- [ ] **Step 4: Test compilation**

Run: `cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig && zig build`

Expected: SUCCESS

---

## Task 3: Verify Engine Offset Calculations

**Files:**
- Modify: `echo-core-zig/src/inference/engine.zig`

- [ ] **Step 1: Verify weight_pool allocation**

Check line 81-83 in Engine.init():

```zig
// Should already be correct - allocates layout.raw_pool_size bytes
const weight_pool = try allocator.alloc(u8, layout.raw_pool_size);
```

- [ ] **Step 2: Update dtypeForTensor/dtypeForGlobal if needed**

Verify the slot calculation in `dtypeForTensor()` (lines 19-33):

```zig
fn dtypeForTensor(weight_dtypes: []const gguf.GGMLType, layer_idx: u32, tensor_idx: u32) gguf.GGMLType {
    // Slot layout must match inference.zig exactly
    // Per layer: 0=attn_norm, 1=q_proj, 2=k_proj, 3=v_proj, 4=o_proj, 
    //            5=ffn_norm, 6=ffn_w1, 7=ffn_w2, 8=ffn_w3, 9=attn_q_norm, 10=attn_k_norm
    //            11=ssm_out, 12=ssm_x, 13=ssm_dt, 14=ssm_A, 15=ssm_B, 16=ssm_C, 17=ssm_D, 18=ssm_conv1d
    const slot = 1 + layer_idx * 19 + tensor_idx;
    return if (slot < weight_dtypes.len) weight_dtypes[slot] else .f16;
}
```

- [ ] **Step 3: Verify matvecDispatchQuant calls**

Check that all matvec calls use `matvecDispatchQuant()` with correct dtype (already done per memory):

```zig
// Line ~431 in attention()
matvec.matvecDispatchQuant(TILE_K, TILE_M, weight_ptr, input.ptr, output.ptr, M, K, 
    dtypeForTensor(self.weight_dtypes, layer_idx, tensor_idx));
```

- [ ] **Step 4: Test compilation**

Run: `cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig && zig build`

Expected: SUCCESS

---

## Task 4: Verify Kernel Dispatch in matvec.zig

**Files:**
- Modify: `echo-core-zig/src/kernels/matvec.zig`

- [ ] **Step 1: Verify matvecDispatchQuant handles all dtypes**

Check line 28-45:

```zig
pub fn matvecDispatchQuant(
    comptime TILE_K: u32,
    comptime TILE_M: u32,
    W: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
    dtype: gguf.GGMLType,
) void {
    switch (dtype) {
        .f16, .f32 => matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K),
        .q8_0 => matvecQ80(W, x, y, M, K),
        .q4_k => matvecQ4K(W, x, y, M, K),
        .q2_k => matvecQ2K(W, x, y, M, K),
        .q6_k => matvecQ6K(W, x, y, M, K),  // Add if missing
        else => {
            std.debug.print("WARN: unsupported dtype {s}, falling back to f16\n", .{@tagName(dtype)});
            matvecFp16Fp32(TILE_K, TILE_M, W, x, y, M, K);
        },
    }
}
```

- [ ] **Step 2: Add Q6_K kernel if missing**

If Q6_K kernel doesn't exist, add basic implementation:

```zig
pub fn matvecQ6K(
    blocks: [*]const u8,
    x: [*]const f32,
    y: [*]f32,
    M: u32,
    K: u32,
) void {
    const blocks_per_row = K / 256;
    const block_stride = 210;  // Q6_K block size

    var m: u32 = 0;
    while (m < M) : (m += 1) {
        var sum: f32 = 0;
        const row_ptr = blocks + @as(usize, m) * blocks_per_row * block_stride;

        var b: u32 = 0;
        while (b < blocks_per_row) : (b += 1) {
            const bp = row_ptr + b * block_stride;
            // Q6_K layout: 210 bytes per 256 elements
            // d (fp16) + scales (64 bytes) + ql (128 bytes) + qh (64 bytes)
            const d = types.fp16_to_fp32(std.mem.readInt(u16, bp[0..2], .little));
            const scales = bp[2..66];
            const ql = bp[66..194];
            const qh = bp[194..258];
            const x_blk = x + b * 256;

            // Dequantize and multiply
            var j: u32 = 0;
            while (j < 256) : (j += 1) {
                const scale_idx = j / 16;
                const scale = d * @as(f32, @floatFromInt(scales[scale_idx]));
                
                const ql_idx = j;
                const qh_idx = j / 2;
                const qh_shift: u3 = @intCast((j % 2) * 4);
                
                const low_bits = ql[ql_idx] & 0x0F;
                const high_bits = (qh[qh_idx] >> qh_shift) & 0x03;
                const q = low_bits | (high_bits << 4);
                
                sum += (scale * @as(f32, @floatFromInt(q)) - 32.0) * x_blk[j];
            }
        }
        y[m] += sum;
    }
}
```

- [ ] **Step 3: Test compilation**

Run: `cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig && zig build`

Expected: SUCCESS

---

## Task 5: Integration Testing

- [ ] **Step 1: Build the project**

```bash
cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig
zig build
```

Expected output: `success`

- [ ] **Step 2: Run unit tests**

```bash
cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig
zig build test
```

Expected: All tests pass

- [ ] **Step 3: Test with actual Q8_0 model**

```bash
cd /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/echo-core-zig
./zig-out/bin/echo-core-zig "../Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" \
    --bench --json --prompt "Merhaba" --max-tokens 10
```

Expected:
- No OOM error
- RAM usage < 10 GB (check with `htop` or `free -h`)
- Model loads successfully
- Inference produces output (may be slow)

- [ ] **Step 4: Record results in In-Memoria**

Use In-Memoria MCP to save implementation details:

```json
{
  "type": "implementation",
  "feature": "Raw Quantized Weight Storage",
  "files_changed": [
    "src/core/memory.zig",
    "src/ports/inference.zig",
    "src/inference/engine.zig",
    "src/kernels/matvec.zig"
  ],
  "memory_savings": "~93% reduction (87.8 GB → ~4.5 GB)",
  "approach": "Store raw quantized bytes, dequantize on-the-fly in kernels",
  "test_results": "PASS / FAIL (with details)"
}
```

---

## Success Criteria Checklist

- [ ] `zig build` completes without errors
- [ ] `zig build test` passes all tests
- [ ] Q8_0 model loads without OOM
- [ ] RAM usage stays below 10 GB during loading
- [ ] Inference produces coherent output
- [ ] Results recorded in In-Memoria

---

## Debugging Guide

### If compilation fails:
1. Check import statements - add `const gguf = @import("../gguf/reader.zig");` if missing
2. Verify function signatures match between files
3. Check for missing switch cases in dtype handling

### If OOM still occurs:
1. Add debug print in `memory.zig` to log calculated sizes:
   ```zig
   std.debug.print("token_embd: {d} bytes\n", .{layout.token_embedding_size});
   ```
2. Verify `layout.total_size` matches file size expectations
3. Check that quantized byte calculation is correct (Q8_0 = 34 bytes per 32 elements)

### If inference produces garbage:
1. Verify weight_dtypes array is populated correctly
2. Check that matvecDispatchQuant receives correct dtype
3. Add debug output in kernels to verify quantized data format
4. Test with small FP16 model first to verify engine logic

---

## Notes for Implementer

**CRITICAL**: This plan assumes the existing `matvecDispatchQuant()` and quantized kernels (Q8_0, Q4_K, Q2_K) work correctly with raw quantized data. The main changes are:

1. **memory.zig**: Calculate byte sizes using actual quantized formats instead of assuming FP16
2. **inference.zig**: Store raw bytes instead of dequantizing (remove `copyTensorToFp16` calls)
3. **engine.zig**: Already uses quantized kernels, just verify slot mapping

The kernel infrastructure is already present - you're just feeding it raw quantized data instead of pre-dequantized FP16 data.

**Quantization Format Reference:**
- Q8_0: 34 bytes per 32 elements (scale: 2 bytes, quants: 32 bytes)
- Q4_0: 18 bytes per 32 elements (scale: 2 bytes, quants: 16 bytes)
- Q4_K: 144 bytes per 256 elements (super-block with multiple scales)
- Q6_K: 210 bytes per 256 elements (super-block)
