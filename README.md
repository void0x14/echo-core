# echo-core

## 5 dakikada başla

```bash
cd echo-core-zig
zig build -Doptimize=ReleaseFast                    # → zig-out/bin/echo-core
# sisteme kur (isteğe bağlı): zig build -Doptimize=ReleaseFast --prefix ~/.local

M=../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf
./zig-out/bin/echo-core -m $M                       # sohbet
./zig-out/bin/echo-core -m $M -p "Türkiye'nin başkenti?"   # tek soru
echo "2+2?" | ./zig-out/bin/echo-core -m $M          # stdin'den
./zig-out/bin/echo-core -m $M --server --port 8080   # OpenAI uyumlu sunucu
./zig-out/bin/echo-core --help                       # tüm seçenekler
```

Sunucu: `POST /v1/chat/completions` (stream dahil), `POST /v1/completions`, `GET /v1/models`, `GET /health`.
`openai` Python istemcisiyle `base_url="http://127.0.0.1:8080/v1"` vererek kullanılır.

Sohbette: `/reset` geçmişi siler, `/sys METİN` sistem mesajı, `/quit` çıkış, satır sonunda `\` çok satırlı giriş, Ctrl+C üretimi keser.

### Modlar ve hız (Ryzen 5 3600, 433 token prompt + 128 token üretim, 8 Kas 2026)

| Mod | Prompt (tok/s) | Üretim (tok/s) | Prompt kat artışı | Üretim kat artışı | Kalite |
|---|---|---|---|---|---|
| llama.cpp (pp512 / tg128) | 171 | 30.5 | 1.00x | 1.00x | referans |
| echo-core varsayılan | 757 | 34.8 | 4.43x | 1.14x | llama.cpp ile ilk tahmin uyumu %96.5, ppl +%1.1 |
| echo-core `--exact` | ~250 | 36.5 | 1.46x | 1.20x | llama.cpp ile token token aynı |
| echo-core `--fast` | 1145 | 117 | 6.70x | 3.84x | 8 katman atlanır: sohbette anlamsız metin |

- Varsayılan: Qwen önerisi örnekleme (temp 0.7, top-k 20, top-p 0.8), tam model, chat şablonu, düşünme kapalı (`--think` açar).
- `--exact`: logit'leri llama.cpp ile bit bit aynı olan hesap yolu (`src/llama/exact/`). Greedy çıktı 5 prompt × 128 tokende llama.cpp ile karakter karakter aynı (`benchmarks/greedy-llama.sh`). pp ölçümü 601 tokenlik prompt ile.
- `--fast`: barajlarda ölçülen hız ayarı; sohbet kalitesi bozuk.
- Tokenizer: Qwen resmi tokenizer ile 6903/6903 paragraf aynı (`--tok-check`, `benchmarks/tokenizer/`).


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
- [x] 2000 barajını aşmak (kayıplı: ECHO_FFN_KEEP=1920 + ECHO_FFN_ACT + ECHO_SKIP 12 katman, top-1 −%59)
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
- [x] 1000 barajını aşmak (varsayılan: 8 katman atlama + nprobe 128, tg128 ~1270; tam model için ECHO_SKIP= boş; ayrıntı TAVAN-VE-VARSAYIMLAR.md)
