# llama.cpp CPU parity — ilerleme kaydı

Hedef: `BENCHMARK-CPU-GPU-2026-11-04.md`'deki CPU throughput'unu Zig motorunda 1:1 yakalamak, tercihen geçmek.
Yöntem: llama.cpp b11146 (`7fe450e`) sıcak yolunu Zig'e satır satır kopyalamak.

## Ölçüm ve doğrulama

| Araç | Ne yapar |
|---|---|
| `zig-out/bin/echo-core-zig --llama-bench -m M -t 6 -r 5 -p 128,512 -n 32,128` | `llama-bench` ile aynı döngü: taze context, warmup, tekrar başına memory clear, rastgele token |
| `tools/llama_parity.py` | iki motoru aynı testte sırayla koşar, oran tablosu üretir |
| `tools/ref_logits.cpp` + `tools/compare_logits.py` | aynı token dizisinde llama.cpp ve echo-core logit'lerini karşılaştırır |
| `zig test -OReleaseFast -fllvm -lc -lm src/llama/ops.zig` | tiled/skaler flash, `simdGemm`, `vExpf` birim testleri |

## Kopyalanan parçalar (`echo-core-zig/src/llama/`)

| Dosya | llama.cpp kaynağı |
|---|---|
| `pool.zig` | `ggml_barrier`, `ggml_threadpool_chunk_*`, `ggml_graph_compute_thread` |
| `vec.zig` | `vec.h`, `vec.cpp`, `simd-gemm.h` |
| `quants.zig` | `arch/x86/quants.c`, `ggml-quants.c` |
| `sgemm.zig` | `llamafile/sgemm.cpp`: `tinyBLAS`, `tinyBLAS_Q0_AVX` |
| `ops.zig` | `ggml_compute_forward_mul_mat`, `rms_norm`, `flash_attn_ext`, `gated_delta_net`, `ssm_conv`, `rope` |
| `qwen35.zig` | `models/qwen35.cpp`, `delta-net-base.cpp`, KV/recurrent memory |
| `bench.zig` | `tools/llama-bench/llama-bench.cpp` |

## Kayıt

### 2026-11-05: ilk port

Kanıtlanan:
- `perf record` llama.cpp sıcak yolunu gösterdi: `tinyBLAS_Q0_AVX`, `tinyBLAS<F16>`, `ggml_vec_dot_q8_0_q8_0`, `ggml_vec_dot_f16`, `ggml_vec_dot_q5_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`.
- Eski motorda pp'nin 10 kat geride olmasının nedeni: prefill her token için LM head çalıştırıyordu, batch yoktu.
- Q8_0, 128 token batch: 24 katmanın `l_out` tensörleri llama.cpp ile bit düzeyinde aynı. Kalan fark flash attention çıktısında yaklaşık 1e-6; son katmanda tek kuantalama çevrilmesi logit'te yaklaşık 0,2 fark bırakıyor.

Çürütülen:
- "Fark gerçek bir hata": Q5_K ve UD modellerindeki 0,2–0,6 logit farkı doğrudan bug değil. 1e-6'lık fark `quantize_row_q8_K` eşiğinde tek bir değeri çevirip katmanlarla büyüyor.

Bulunan tuzaklar:
- `@reduce(.Add)` toplama sırası ggml'den farklı: `hsum_float_8` ve `GGML_F32x8_REDUCE` aynen kopyalandı.
- `@min(Q_TILE, x)` sonucu `u7`'ye daraltıp `tile_rows * KV_TILE`'ı taşırıyor; yalnız Debug yakaladı.
- Zig `compiler_rt`, `expf/logf/sinf/cosf` adlarını binary içinde tanımlayıp glibc'yi gölgeliyor: `dlsym` ile alınıyor.

### 2026-11-05: kernel kod üretimi ve ilk tam matris

Kanıtlanan:
- Q8_0 modelinde `compare_logits.py --steps 5,1,1,128,1`: tüm adımlarda max|fark| = 0,00000, argmax ve top5 aynı.
- Tek thread `pp128`: Q8_0 %99,4, UD %100,4, Q5_K %109,4 (llama.cpp'ye oran).
- 6 thread tam matris, aynı anda sırayla ölçüm (oran = echo-core / llama.cpp):

| Model | pp128 | pp512 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 | 258,44 / 237,10 = %109,0 | 248,55 / 232,99 = %106,7 | 35,05 / 29,16 = %120,2 | 36,46 / 30,49 = %119,6 |
| Qwen3.5-0.8B-UD-Q8_K_XL | 251,84 / 226,35 = %111,3 | 211,59 / 228,39 = %92,6 | 26,55 / 23,55 = %112,8 | 27,98 / 23,31 = %120,0 |
| unc-0.8b-q5km | 188,45 / 165,41 = %113,9 | 187,90 / 156,91 = %119,7 | 67,10 / 50,12 = %133,9 | 69,53 / 54,06 = %128,6 |

Bulunan tuzaklar:
- Kuantalı kernel'da GCC'ye göre %18 yavaşlık: LLVM `vpinsrw` ile xmm'e sahte bağımlılık yaratıyor, `|a|` değerlerini CSE edip akümülatörleri spill ediyordu. `gemm4xN<2>` iç döngüsü inline asm'e alındı.
- KV buffer sıfırlanmazsa tile içindeki boş hücrelerden NaN geliyor (ggml `ggml_backend_buffer_clear` yapar).

Sıradaki tek adım: UD pp512 hücresini %100'e çıkarmak.
