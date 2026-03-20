# echo-core

## Current Development: Low-Level Inference Sprint
Optimizing the execution engine to break memory bandwidth bottlenecks on local hardware.

### Implementation Details:
- **L3-Aware Cache Tiling:** `constexpr` optimized weight fetching for target CPUs (18MB/32MB L3).
- **INT8 KV-Cache:** Per-token symmetric quantization to halve RAM usage while maintaining logic.
- **Bare-Metal Kernels:** Direct AVX2/FMA/F16C implementation for maximum hardware saturation.
- **Unified Memory Architecture:** Aligned memory pooling for zero-overhead linear data flow.

## Goal
Achieving 50-100 t/s on local CPU/RAM configurations (i5-13500H / Ryzen 3600).
