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
