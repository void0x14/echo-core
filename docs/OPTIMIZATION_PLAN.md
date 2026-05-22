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

- [x] **0.1** `src/kernels/simd.zig` — Zig SIMD yardimci fonksiyonlari
  - `hsumF32_8(v: @Vector(8, f32)) f32` — horizontal sum
  - `hsumI32_8(v: @Vector(8, i32)) i32`
  - `dotProductI8(v1: @Vector(32, i8), v2: @Vector(32, i8)) i32` — maddubs equivalent

### Faz 1: Q6_K AVX2 Kernel (en kritik)

Q6_K output proj suan 40sn. Hedef: <10ms.

- [x] **1.1** `src/kernels/avx2_q6k.zig` — Q6_K blok yapisi + 6-bit unpack
- [x] **1.2** `dotQ6Block()` — tek Q6_K blok × f32 via `@Vector(16, u8/f32)`
- [x] **1.3** `matvecQ6K_avx2()` — full M×K loop
- [x] **1.4** `matvecDispatchQuant`'a bagla + test

### Faz 2: Q4_K AVX2 Kernel

Q4_K modeldeki tensorlerin cogu. Her matvec'te kullanilir.

- [x] **2.1** `dotQ4Block()` — Q4_K blok × f32 via `@Vector`
- [x] **2.2** `matvecQ4K_avx2()` — M×K loop + dispatch baglama + test

### Faz 3: Q5_K AVX2 Kernel

- [x] **3.1** `matvecQ5K_avx2()` — qh[32] high-bit unpack + dispatch + test

### Faz 4: Softmax AVX2

- [x] **4.1** `softmaxAvx2()` — `@Vector(8, f32)` ile `@exp` builtin + engine baglama

### Faz 5: RMS Norm AVX2

- [x] **5.1** `rmsNormAvx2()` — `@Vector(8, f32)` + fp16 weight load + engine baglama

### Faz 6: Multi-threading

- [x] **6.1** `parallelMatvec()` — output proj (M=151665) 4 thread chunk parallel
- [x] **6.2** engine entegrasyonu — `forwardToken`'da output proj cagrisi parallel

### Faz 7: Tiny Vector Edge Cases

- [x] **7.1** K < 256 guard — Q6_K dispatch'te K >= 256 kontrolu eklendi

## Tahmini Kazanim (kumulatif)

| Faz | Eklenti | Gercek hiz | Kazanc |
|-----|---------|-----------|-------|
| Baseline | generic dequant | 0.024 tok/s | 1x |
| Faz 1 | Q6_K AVX2 | 0.031 tok/s | 1.27x |
| Faz 2-3 | Q4_K+Q5_K AVX2 | 0.044 tok/s | 1.83x |
| Faz 4-5 | softmax+norm AVX2 | 0.044 tok/s | ≈ |
| Faz 6 | 4 thread output proj | 0.054 tok/s | 2.25x |
| Faz 7 | K<256 guard | 0.054 tok/s | stabilized |

Not: 50-100 tok/s icin daha fazla threading + integer dot product gerekli.

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
