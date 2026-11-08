# echo-core

## Current Development: Low-Level Inference Sprint
Optimizing the execution engine to break memory bandwidth bottlenecks on local hardware.

### Implementation Details:
- **L3-Aware Cache Tiling:** Optimized weight fetching for target CPUs (18MB/32MB L3).
- **INT8 KV-Cache:** Per-token symmetric quantization to halve RAM usage while maintaining logic.
- **Bare-Metal Kernels:** Direct AVX2/FMA/F16C implementation for maximum hardware saturation.
- **Unified Memory Architecture:** Aligned memory pooling for zero-overhead linear data flow.

## Goal
Achieving 50-100 t/s on local CPU/RAM configurations (i5-13500H / Ryzen 3600).

## Aşılcak barajlar;

### Prefill
- [x] 750 barajını aşmak
- [x] 1000 barajını aşmak (kayıplı: ECHO_SKIP 8 katman)
- [x] 1500 barajını aşmak (kayıplı: ECHO_FFN_KEEP=2176 + ECHO_FFN_ACT + ECHO_SKIP 9 katman, top-1 −%48)
- [ ] 2000 barajını aşmak
- [ ] 5000 barajını aşmak
- [ ] 7500 barajını aşmak
- [ ] 10000 barajını aşmak

### TG
- [x] 70 barajını aşmak
- [x] 100 barajını aşmak
- [x] 150 barajını aşmak
- [x] 200 barajını aşmak
- [x] 300 barajını aşmak
- [x] 500 barajını aşmak
- [x] 750 barajını aşmak
- [x] 1000 barajını aşmak (kayıplı: ECHO_SKIP 8 katman + --lm-ivf 4096,128; ayrıntı ISLEYIS-ARASTIRMA.md)
