# Model Benchmark — llama.cpp vs echo-core (CPU / GPU)

Tarih: 2026-11-04 · echo-core: `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core`

## Ortam

| Öğe | Değer |
|---|---|
| CPU | AMD Ryzen 5 3600, 6 çekirdek / 12 thread, AVX2+FMA+F16C |
| GPU | AMD Radeon RX 460 (RADV POLARIS11), 2048 MiB VRAM, fp16 yok |
| RAM | 15 GiB |
| OS | CachyOS Linux, kernel 6.12.69-2-cachyos-lts-lto |
| llama.cpp | b11146 (commit 7fe450e193), CPU + Vulkan backend, kaynak derleme (Release) |
| echo-core | Zig 0.17.0-dev.332+4a65cc4ac, `zig build -Doptimize=ReleaseFast` |
| Thread sayısı | llama.cpp `-t 6`; echo-core thread pool = 6 (mantıksal CPU/2) |
| Tekrar | llama.cpp `-r 5` (5 ölçüm, medyan); echo-core 3 ölçüm, medyan |

## Tek bakışta (CPU)

| Model | Motor | pp128 | pp512 | tg32 | tg128 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | llama.cpp | 263.39 | 253.33 | 33.28 | 30.01 |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | echo-core (eski motor, 11-04) | 25.50 | 26.25 | 24.48 | 26.80 |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | **echo-core (güncel, 11-05)** | **285.56** | **281.94** | **39.09** | **36.77** |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | llama.cpp | 237.12 | 190.66 | 24.59 | 24.89 |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | echo-core (eski motor, 11-04) | 19.39 | 20.58 | 20.29 | 19.01 |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | **echo-core (güncel, 11-05)** | **253.81** | **258.90** | **25.61** | **27.22** |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | llama.cpp | 169.59 | 161.86 | 53.60 | 55.42 |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | echo-core (eski motor, 11-04) | 20.34 | 24.21 | 24.13 | 25.41 |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | **echo-core (güncel, 11-05)** | **191.26** | **193.04** | **68.72** | **66.90** |

## Tek bakışta (GPU)

| Model | Motor | pp128 | pp512 | tg32 | tg128 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | llama.cpp (Vulkan, ngl=99) | 793.93 | 828.58 | 76.08 | 77.07 |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | echo-core | Yok | Yok | Yok | Yok |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | llama.cpp (Vulkan, ngl=99) | 736.04 | 483.91 | 57.73 | 58.37 |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | echo-core | Yok | Yok | Yok | Yok |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | llama.cpp (Vulkan, ngl=99) | 758.53 | 782.10 | 95.11 | 95.69 |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | echo-core | Yok | Yok | Yok | Yok |

## Ölçüm tanımları

| Kısaltma | Anlamı |
|---|---|
| pp128 | 128 token prefill (boş KV cache) |
| pp512 | 512 token prefill (boş KV cache) |
| tg32 | 32 token decode, boş context (llama.cpp: `n_prompt=0`; echo-core: BOS only) |
| tg128 | 128 token decode, boş context |

llama.cpp prefill için rastgele token, echo-core için aynı sayıda token üreten metin kullanıldı
(echo-core `prompt_tokens`: pp128 = 128, pp512 = 512 — her iki motor için de birebir eşit token sayısı).
## Metodoloji farkları (ölçüm koşulları)

| Öğe | llama.cpp | echo-core |
|---|---|---|
| Tekrar sayısı | 5 (aynı süreç içinde; ilk ölçüm ısınma) | 3 (her tekrar ayrı süreç; ısınma yok) |
| Raporlanan değer | 5 örneğin medyanı | 3 örneğin medyanı |
| Prefill girdisi | rastgele token (n_prompt) | metin (echo-core tokenizer ile birebir 128/512 token) |
| Decode context | boş (n_prompt=0) | BOS only (1 token) |
| Ölçüm kapsamı | prefill veya decode | prefill (`prefill_ms`) veya decode (`decode_ms`) |
| GPU offload | `ngl=99` → 26/26 katman Vulkan0'a | yok |
| Aynı metindeki token sayısı (Qwen3.5-0.8B-Q8_0, 687 karakter) | 128 (BPE) | 283 (byte-level, uzun eşleşme yok) |

## CPU

llama.cpp: `GGML_DISABLE_VULKAN=1 llama-bench -t 6 -r 5 -p 128,512 -n 0,32,128 -ngl 0 -o json`
(Vulkan backend devre dışı; `ngl=0` ile katmanların tamamı CPU'da)

echo-core: `echo-core-zig <model> --prompt <p> --max-tokens <0|32|128> --bench --json`

| Model | Test | llama.cpp (tok/s) | echo-core (tok/s) | echo-core / llama.cpp |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | pp128 | 263.39 | 25.50 | 9.7% |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | pp512 | 253.33 | 26.25 | 10.4% |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | tg32 | 33.28 | 24.48 | 73.6% |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | tg128 | 30.01 | 26.80 | 89.3% |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | pp128 | 237.12 | 19.39 | 8.2% |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | pp512 | 190.66 | 20.58 | 10.8% |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | tg32 | 24.59 | 20.29 | 82.5% |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | tg128 | 24.89 | 19.01 | 76.4% |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | pp128 | 169.59 | 20.34 | 12.0% |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | pp512 | 161.86 | 24.21 | 15.0% |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | tg32 | 53.60 | 24.13 | 45.0% |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | tg128 | 55.42 | 25.41 | 45.8% |

### CPU — güncel echo-core (2026-11-05)

Yukarıdaki echo-core sütunu 11-04 tarihli eski motora aittir. Güncel motor: `echo-core-zig/src/llama`, commit `2f281fb`.

echo-core: `tools/doc_parity.py --reps 5 --rounds 3` (`--llama-bench -t 6 -r 5`; `llama-bench` döngüsü, rastgele token). Değer: 3 turun medyanlarının medyanı. Ham veri: `echo-core-zig/benchmarks/doc-parity-2026-11-05.json`.

llama.cpp sütunu 11-04 ölçümüdür, aynı anda alınmadı. echo-core koşusu sonunda load average 8,2 idi.

| Model | Test | llama.cpp 11-04 (tok/s) | echo-core 11-05 (tok/s) | echo-core / llama.cpp |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 | pp128 | 263.39 | 285.56 | 108.4% |
| Qwen3.5-0.8B-Q8_0 | pp512 | 253.33 | 281.94 | 111.3% |
| Qwen3.5-0.8B-Q8_0 | tg32 | 33.28 | 39.09 | 117.5% |
| Qwen3.5-0.8B-Q8_0 | tg128 | 30.01 | 36.77 | 122.5% |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp128 | 237.12 | 253.81 | 107.0% |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp512 | 190.66 | 258.90 | 135.8% |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg32 | 24.59 | 25.61 | 104.2% |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg128 | 24.89 | 27.22 | 109.4% |
| unc-0.8b-q5km | pp128 | 169.59 | 191.26 | 112.8% |
| unc-0.8b-q5km | pp512 | 161.86 | 193.04 | 119.3% |
| unc-0.8b-q5km | tg32 | 53.60 | 68.72 | 128.2% |
| unc-0.8b-q5km | tg128 | 55.42 | 66.90 | 120.7% |

### CPU — ham ölçümler

| Model | Test | llama.cpp örnekleri (tok/s) | echo-core örnekleri (tok/s) |
|---|---|---|---|
| Qwen3.5-0.8B-Q8_0 | pp128 | 263.78, 264.88, 260.76, 262.76, 263.39 | 24.36, 25.63, 25.50 |
| Qwen3.5-0.8B-Q8_0 | pp512 | 255.59, 220.26, 253.33, 258.67, 250.05 | 26.25, 26.47, 24.10 |
| Qwen3.5-0.8B-Q8_0 | tg32 | 33.32, 33.51, 30.37, 33.28, 32.07 | 24.48, 22.61, 25.22 |
| Qwen3.5-0.8B-Q8_0 | tg128 | 31.32, 29.61, 29.64, 30.22, 30.01 | 25.93, 26.80, 26.92 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp128 | 241.28, 237.96, 233.19, 232.65, 237.12 | 19.39, 19.70, 19.25 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp512 | 227.75, 228.29, 180.08, 184.91, 190.66 | 20.66, 20.43, 20.58 |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg32 | 24.59, 24.75, 24.17, 24.31, 24.73 | 20.85, 20.29, 19.28 |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg128 | 24.89, 24.58, 23.77, 24.96, 25.06 | 16.70, 20.27, 19.01 |
| unc-0.8b-q5km | pp128 | 170.46, 169.59, 170.67, 151.56, 162.83 | 20.12, 20.34, 20.71 |
| unc-0.8b-q5km | pp512 | 171.37, 170.24, 161.86, 146.31, 161.46 | 24.21, 23.57, 25.24 |
| unc-0.8b-q5km | tg32 | 53.67, 53.60, 49.61, 53.72, 51.83 | 23.97, 24.13, 26.42 |
| unc-0.8b-q5km | tg128 | 55.91, 55.42, 54.71, 55.68, 53.54 | 24.97, 25.41, 25.88 |

## GPU

llama.cpp: `llama-bench -t 6 -r 5 -p 128,512 -n 0,32,128 -ngl 99 -o json` (Vulkan0 = AMD Radeon RX 460)

echo-core: **GPU yolu yok — ölçüm yapılmadı** (aşağıdaki kanıta bak) 

| Model | Test | llama.cpp Vulkan (tok/s) | echo-core GPU (tok/s) | echo-core / llama.cpp |
|---|---|---:|---:|---:|
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | pp128 | 793.93 | Yok | — |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | pp512 | 828.58 | Yok | — |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | tg32 | 76.08 | Yok | — |
| Qwen3.5-0.8B-Q8_0 (Q8_0, 794.6 MiB) | tg128 | 77.07 | Yok | — |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | pp128 | 736.04 | Yok | — |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | pp512 | 483.91 | Yok | — |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | tg32 | 57.73 | Yok | — |
| Qwen3.5-0.8B-UD-Q8_K_XL (UD-Q8_K_XL, 1149.4 MiB) | tg128 | 58.37 | Yok | — |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | pp128 | 758.53 | Yok | — |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | pp512 | 782.10 | Yok | — |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | tg32 | 95.11 | Yok | — |
| unc-0.8b-q5km (Q5_K_M, 382.6 MiB) | tg128 | 95.69 | Yok | — |

### GPU — ham ölçümler (llama.cpp)

| Model | Test | llama.cpp örnekleri (tok/s) |
|---|---|---|
| Qwen3.5-0.8B-Q8_0 | pp128 | 770.81, 793.93, 782.66, 796.84, 796.17 |
| Qwen3.5-0.8B-Q8_0 | pp512 | 832.13, 828.58, 823.95, 826.21, 828.65 |
| Qwen3.5-0.8B-Q8_0 | tg32 | 76.70, 76.01, 76.21, 76.08, 75.89 |
| Qwen3.5-0.8B-Q8_0 | tg128 | 77.20, 77.07, 77.02, 77.14, 77.04 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp128 | 729.13, 736.69, 736.04, 736.89, 732.72 |
| Qwen3.5-0.8B-UD-Q8_K_XL | pp512 | 483.83, 483.60, 484.83, 485.03, 483.91 |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg32 | 57.80, 57.82, 57.59, 57.73, 57.53 |
| Qwen3.5-0.8B-UD-Q8_K_XL | tg128 | 58.33, 58.37, 58.35, 58.38, 58.39 |
| unc-0.8b-q5km | pp128 | 746.63, 759.65, 758.53, 754.11, 761.24 |
| unc-0.8b-q5km | pp512 | 779.57, 787.45, 780.07, 782.10, 782.87 |
| unc-0.8b-q5km | tg32 | 95.48, 95.11, 95.07, 94.68, 95.26 |
| unc-0.8b-q5km | tg128 | 96.02, 95.67, 95.82, 95.69, 95.69 |

## echo-core GPU destek kanıtı

| Kontrol | Sonuç |
|---|---|
| `grep -rniE 'cuda\|vulkan\|gpu\|rocm\|metal\|hip\|spirv\|opencl\|sycl' src/ build.zig` | 0 eşleşme |
| `ldd zig-out/bin/echo-core-zig` | `not a dynamic executable` (GPU kütüphanesi bağlı değil) |
| CLI bayrakları | `--model --prompt --max-tokens --bench --json` (GPU/ngl bayrağı yok) |
| Vulkan backend yükleme testi | `llama-bench --list-devices` → `Vulkan0: AMD Radeon RX 460` |

## Doğrulama tekrarı (re-run)

Rapordaki değerlerin tekrarlanabilirliği için aynı hücreler yeniden koşuldu.

| Model | Test | Motor | İlk koşu (raporlanan medyan) | Tekrar koşu |
|---|---|---|---:|---|
| unc-0.8b-q5km | tg128 | echo-core | 25.41 | 27.84, 23.37 |
| unc-0.8b-q5km | tg128 | llama.cpp CPU | 55.42 | 48.77 ± 0.45 (3 tekrar) |

## Ham sonuç dosyaları

| Dosya | İçerik |
|---|---|
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/results/llamacpu_<model>.json` | llama.cpp CPU, 5 tekrar |
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/results/llamagpu_<model>.json` | llama.cpp GPU (Vulkan, ngl=99), 5 tekrar |
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/results/echocpu.jsonl` | echo-core CPU, 3 tekrar |
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/prompts/echo_<model>_{128,512}.txt` | echo-core için 128/512 token'a ayarlanmış promptlar |
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/tokcount/` | echo-core tokenizer sayacı (orijinal kodla 36/36 doğrulandı) |
| `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/bench-build/llama.cpp/` | llama.cpp b11146 kaynak ağacı + Vulkan build (`build-vk/`) |

