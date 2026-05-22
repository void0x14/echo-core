# Optimization Plan — Zig-native AVX2 Kernels

## Hedef

0.02 tok/s → 50-100 tok/s. Zero C, zero dependency. Pure Zig.

## Strateji

Zig `@Vector(T, N)` → LLVM AVX2 kod uretir. Kritik operasyonlar icin inline asm.

### Zig SIMD araclari

| Zig kodu | Urettigi ASM | Kullanim |
|----------|-------------|----------|
| `@Vector(8, f32)` | 256-bit AVX/AVX2 | f32 matmul, softmax, norm |
| `@Vector(32, i8)` | 256-bit AVX2 | quantized data unpack |
| `@mulAdd(a, b, c)` | `VFMADD132PS` (FMA) | fused multiply-add |
| `@reduce(.Add, v)` | `VHADDPS` + shuffle | horizontal sum |
| `@shuffle(v, idx)` | `VPSHUFB` / `VPERM2F128` | byte/word permute |
| `@select(mask, a, b)` | `VBLENDVPS` | conditional select |
| `asm volatile` | raw ASM | When LLVM won't cooperate |

## Mikro G�revler

### Faz 0: Altyapi (1 gorev)

- [ ] **0.1** `src/kernels/simd.zig` — Zig SIMD yardimci fonksiyonlari
  - `hsumF32_8(v: @Vector(8, f32)) f32` — horizontal sum
  - `hsumI32_8(v: @Vector(8, i32)) i32`
  - `dotProductI8(v1: @Vector(32, i8), v2: @Vector(32, i8)) i32` — maddubs equivalent

### Faz 1: Q6_K AVX2 Kernel (en kritik)

Q6_K output proj suan 40sn. Hedef: <10ms.

- [ ] **1.1** `src/kernels/avx2_q6k.zig` — Q6_K blok yapisini coz
  - block_q6_K: ql[128], qh[64], scales[16], d(u16), dmin(u16)
  - ELLE coz: 6-bit quants, 4-bit low + 2-bit high packing
  
- [ ] **1.2** `vecDotQ6K_1block()` — tek Q6_K blok × f32 aktivasyon
  - Load ql (128 byte) → `@Vector(32, i8)` x 4
  - Load qh (64 byte) + unpack 2-bit high bits
  - Birles: 6-bit signed values
  - Dot product with activation f32 via FMA

- [ ] **1.3** `matvecQ6K_avx2()` — full M×K matvec
  - Loop over M rows, K/256 blok her row icin
  - `_mm256_fmadd_ps` accumulate

- [ ] **1.4** `matvecDispatchQuant`'a bagla
  - `.q6_k =>` yeni AVX2 kernel

### Faz 2: Q4_K AVX2 Kernel

Q4_K modeldeki tensorlerin cogu. Her matvec'te kullanilir.

- [ ] **2.1** `vecDotQ4K_1block()` — Q4_K blok × f32
  - block_q4_K: d[2], dmin[2], scales[12], qs[128]
  - 4-bit quants, scales 6-bit packing
  
- [ ] **2.2** `matvecQ4K_avx2()` — simdiki matvecQ4K'nin AVX2 versiyonu

### Faz 3: Q5_K AVX2 Kernel

- [ ] **3.1** `matvecQ5K_avx2()` — qh[32] high-bit unpack ile

### Faz 4: Softmax AVX2

- [ ] **4.1** `softmax_avx2()` — `@Vector(8, f32)` ile paralel exp
  - expf polinom yaklasimi (llama.cpp `ggml_v_expf`'in Zig portu)
  - max bul, exp hesapla, sum reduction

### Faz 5: RMS Norm AVX2

- [ ] **5.1** `rmsNorm_avx2()` — `@Vector(8, f32)` ile paralel normalize
  - sum-of-squares reduction, sqrt, scale

### Faz 6: Multi-threading

- [ ] **6.1** `std.Thread.Pool` ile layer paralel
  - 32 layer'i 8 thread'de isle
  - Her thread 4 layer alir
  
- [ ] **6.2** matvec paralel
  - Output proj (M=151665) M/thread chunk'lara bol

### Faz 7: Tiny Vector Edge Cases

- [ ] **7.1** K < 256 durumlari (alpha/beta: K=32)
  - scalar fallback veya `@Vector(32, f32)` ile handle

## Tahmini Kazanim (kumulatif)

| Faz | Eklenti | Tahmini hiz | Aciklama |
|-----|---------|-------------|----------|
| Su an | — | 0.02 tok/s | generic dequant |
| Faz 1 | Q6_K AVX2 | ~2 tok/s | output proj 40sn→~100ms |
| Faz 2-3 | Q4_K+Q5_K AVX2 | ~5 tok/s | tum matvec'ler hizlanir |
| Faz 4-5 | softmax+norm AVX2 | ~7 tok/s | softmax 8x hiz |
| Faz 6 | 8 thread | ~40 tok/s | thread scaling ~6x |
| Faz 7 | edge cases | ~40 tok/s | stabilized |

Not: 50-100 tok/s icin threading OLMAZSA OLMAZ. 8 thread hedef.

## Test Plani

Her kernel:
1. `test` blokunda scalar versiyonla karsilastir
2. Rastgele input + weight ile dogruluk kontrolu
3. Benchmark ile hiz olcumu

## Dosya Yapisi

```
src/kernels/
  simd.zig          — Zig SIMD helpers
  avx2_q6k.zig      — Q6_K AVX2 matvec
  avx2_q4k.zig      — Q4_K AVX2 matvec
  avx2_q5k.zig      — Q5_K AVX2 matvec
  avx2_softmax.zig  — Softmax AVX2
  avx2_norm.zig     — RMS Norm AVX2
  matvec.zig        — mevcut, dispatch guncelle
```
