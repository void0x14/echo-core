# llama.cpp CPU parity — ilerleme kaydı

Hedef: `BENCHMARK-CPU-GPU-2026-11-04.md`'deki CPU throughput'unu Zig motorunda 1:1 yakalamak, tercihen geçmek.
Yöntem: llama.cpp b11146 (`7fe450e`) sıcak yolunu Zig'e satır satır kopyalamak.

**Durum: hedef tuttu.** Belgedeki 12 CPU hücresinin tamamı son kodla geçildi, en kötü oran %104,2; logit'ler 4 modelde llama.cpp ile sıfır farkla aynı. Tablolar en alttaki "Son durum" bölümünde.

## Ölçüm ve doğrulama

| Araç | Ne yapar |
|---|---|
| `zig-out/bin/echo-core-zig --llama-bench -m M -t 6 -r 5 -p 128,512 -n 32,128` | `llama-bench` ile aynı döngü: taze context, warmup, tekrar başına memory clear, rastgele token |
| `tools/llama_parity.py` | iki motoru aynı testte sırayla koşar, oran tablosu üretir |
| `tools/doc_parity.py` | belgedeki CPU hücrelerini okur, echo-core'u ölçüp belge değerine oranlar |
| `tools/ref_logits.cpp` + `tools/compare_logits.py` | aynı token dizisinde llama.cpp ve echo-core logit'lerini karşılaştırır |
| `zig test -OReleaseSafe -fllvm -lc src/llama/ops.zig` | tiled/skaler flash, `simdGemm`, `vExpf` birim testleri |

## Kopyalanan parçalar (`echo-core-zig/src/llama/`)

| Dosya | llama.cpp kaynağı |
|---|---|
| `pool.zig` | `ggml_barrier`, `ggml_threadpool_chunk_*`, `ggml_graph_compute_thread` |
| `vec.zig` | `vec.h`, `vec.cpp`, `simd-gemm.h` |
| `quants.zig` | `arch/x86/quants.c`, `ggml-quants.c` |
| `sgemm.zig` | `llamafile/sgemm.cpp`: `tinyBLAS`, `tinyBLAS_Q0_AVX` |
| `repack.zig` | `repack.cpp`, `arch/x86/repack.cpp`: Q4_K 8x8 `gemm`, `gemv`, `quantize_mat`, `forward_mul_mat` |
| `ops.zig` | `ggml_compute_forward_mul_mat`, `rms_norm`, `flash_attn_ext`, `gated_delta_net`, `ssm_conv`, `rope` |
| `qwen35.zig` | `models/qwen35.cpp`, `delta-net-base.cpp`, KV/recurrent memory |
| `bench.zig` | `tools/llama-bench/llama-bench.cpp` |
| `port.zig` | `--bench`, `--prompt` ve REPL'in qwen35 modellerindeki arayüzü |

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

Bu matristeki UD pp512 %92,6 sonradan gürültü çıktı (aşağıda).

### 2026-11-05: Q4_K repack ve doğruluk

Kanıtlanan:
- llama.cpp Q4_K tensörlerini (`ne[1] % 8 == 0`) `block_q4_Kx8`'e çevirip `ggml_gemm_q4_K_8x8_q8_K` çalıştırıyor. Port sonrası Q4_K_M modelde pp %89–95'ten %107–122'ye çıktı.
- `compare_logits.py`, 4 modelde şu dizilerin tümünde `SONUC: ESIT` (max|fark| = 0,00000, argmax ve top5 aynı): `5,1,1,128,1`, `512,1,1`, `300,300,1,1`, `7,300,300,1,1`, `5,1,128,1,300,1`. `300,300` dizisi `n_kv >= 512` olan split-KV yolunu da sınar.
- ReleaseSafe ile aynı diziler ve birim testleri temiz; `-Dcpu=x86_64` ile derleme eski motora düşer, kırılmaz.

Çürütülen:
- "Ağırlıkları THP'ye kopyalamak LM head'i hızlandırır": 2 MiB hizalı anonim kopya tg'de %1–3 kötü çıktı, kaldırıldı.
- "Thread pinning varyansı düşürür": pinned 208–266, pinned'siz 260–267 tok/s. Eklenmedi.
- "UD pp128/pp512 llama.cpp'nin gerisinde": %92,6 ve %97,6 tek turluk değerler arka plan yükündendi; aynı hücrelerin diğer turları %111–120.

### 2026-11-05: belgeye karşı doğrulama ve Q8_0 kernel'i

Kanıtlanan:
- `gemm4x2_asm` yeniden planlandı (8 akümülatör ymm8–15; `|a|` ve `sign(b,a)` her satır/sütunda yeniden üretiliyor, spill yok). Tek thread Q8_0 pp128 yaklaşık 53'ten 56,5 tok/s'ye çıktı.
- Son binary ile belgedeki 12 hücrenin hepsi geçildi (aşağıdaki birinci tablo).
- Son binary ile `compare_logits.py`, 4 modelde `5,1,128,1,300,1`, `512,1,1`, `7,300,300,1,1` dizilerinde `SONUC: ESIT`, max|fark| = 0,00000. Birim testleri geçiyor.

Çürütülen:
- "Yük altında belge değerine ulaşılamaz": load 8,2 iken bile en kötü hücre %104,2. Önceki tek turda Q8_0 pp128 %92 kalmıştı, şimdi %108,4. Bu 16 puanın yaklaşık 6'sı kernel'den, kalanı turlar arasındaki yük farkından geliyor.

## Son durum

### Belgeye karşı (hedef ölçütü)

Ortam: Ryzen 5 3600, 6 thread, load average 8,2. 3 tur, tur başına 5 tekrar, medyan. Ham veri: `echo-core-zig/benchmarks/doc-parity-2026-11-05.json`. Belge değerleri boşta ölçülmüş llama.cpp sayılarıdır.

| Model | Test | Belge (llama.cpp) | echo-core | Oran |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 | pp128 | 263,39 | 285,56 | %108,4 |
| | pp512 | 253,33 | 281,94 | %111,3 |
| | tg32 | 33,28 | 39,09 | %117,5 |
| | tg128 | 30,01 | 36,77 | %122,5 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp128 | 237,12 | 253,81 | %107,0 |
| | pp512 | 190,66 | 258,90 | %135,8 |
| | tg32 | 24,59 | 25,61 | %104,2 |
| | tg128 | 24,89 | 27,22 | %109,4 |
| unc-0.8b-q5km | pp128 | 169,59 | 191,26 | %112,8 |
| | pp512 | 161,86 | 193,04 | %119,3 |
| | tg32 | 53,60 | 68,72 | %128,2 |
| | tg128 | 55,42 | 66,90 | %120,7 |

### Aynı anda sırayla (önceki kernel)

Bu tablo `gemm4x2_asm` değişikliğinden önce alındı. Ortam: tarayıcı açık (load average 7–11). Tek tur, 5 tekrar, medyan. Ham veri: `echo-core-zig/benchmarks/llama-parity-2026-11-05-son.json`.

| Model | Test | llama.cpp | echo-core | Oran |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 | pp128 | 234,76 | 249,08 | %106,1 |
| | pp512 | 220,65 | 248,77 | %112,7 |
| | tg32 | 30,74 | 36,40 | %118,4 |
| | tg128 | 31,31 | 35,87 | %114,6 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp128 | 203,20 | 198,40 | %97,6 (bkz. not) |
| | pp512 | 207,03 | 213,01 | %102,9 |
| | tg32 | 20,15 | 24,25 | %120,3 |
| | tg128 | 20,17 | 24,13 | %119,7 |
| unc-0.8b-q5km | pp128 | 163,36 | 173,94 | %106,5 |
| | pp512 | 158,61 | 197,36 | %124,4 |
| | tg32 | 55,44 | 71,67 | %129,3 |
| | tg128 | 55,50 | 71,00 | %127,9 |
| Qwen3.5-0.8B-Base-Q4_K_M | pp128 | 239,97 | 278,68 | %116,1 |
| | pp512 | 222,11 | 270,64 | %121,9 |
| | tg32 | 43,57 | 55,34 | %127,0 |
| | tg128 | 45,12 | 53,09 | %117,7 |

Not: UD pp128 aynı koşullarda 4 tur tekrarlandı: llama.cpp 174,84, echo-core 209,75, %120,0 (`llama-parity-2026-11-05-ud-pp128.json`). 2 turluk önceki matris `llama-parity-2026-11-05.json`: en düşük oran %101,9.

Belgedeki mutlak değerler: `BENCHMARK-CPU-GPU-2026-11-04.md` boşta ölçülmüş. Yük altında llama.cpp de o değerlerin altında kalıyor (Q8_0 pp128: 263 yerine 235). Makine sakinken echo-core Q8_0 pp128 269, tg32 39, UD tg32 28 tok/s verdi; belgede sırasıyla 263, 33 ve 24,6.

Kapsam dışı kalanlar:
- Ölçümler yalnız bu CPU'da: AVX-512 veya başka mimari dalları port edilmedi; AVX2 yoksa eski motor çalışır.
- Tokenizer eski (`src/tokenizer`): BPE yerine en uzun eşleşme yapıyor, çıktıda `Ġ` kalıyor, başa BOS ekliyor. Logit ve hız ölçümleri token kimlikleriyle yapıldığı için bunlara etkisi yok, ama `--prompt` metni llama.cpp'den farklı token'lara çevriliyor.
- `src/llama` yalnız `qwen35` mimarisini kapsar; öteki mimariler eski motordadır.

Sıradaki tek adım: llama.cpp BPE tokenizer'ını port etmek (`--prompt` çıktısının da llama.cpp ile aynı olması için).
