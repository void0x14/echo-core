# Tokenizer ölçümü — 2026-11-08

Otomatik üretildi (`calistir.sh`). CPU: AMD Ryzen 5 3600 6-Core Processor, tek çekirdek (`taskset`). Model: Qwen3.5-0.8B-Q8_0 (vocab 248 320, merge 247 587, ön-bölme `qwen35`).
Karşılaştırma ölçüsü: modelin kendi tokenizer'ı (Qwen resmi `tokenizer.json`, huggingface.co/Qwen/Qwen3.5-0.8B). Doğruluk = paragraf bazında token id dizisi resmi tokenizer ile birebir aynı mı.

## 1. Hız (MB/s, tek çekirdek, medyan)

| Yöntem | TR | EN | Kod | 16 KB tek kelime (en kötü) | Metindeki iddia | TR hızı, resmi tokenizer = 1.00x |
|---|---|---|---|---|---|---|
| llama.cpp (libllama) | 1.19 | 2.13 | 2.57 | 1.63 | 15–40 | 0.92x |
| Qwen BPE: heap (oncelik kuyrugu) | 17 | 17 | 23 | 7.49 | — | 13.42x |
| Qwen BPE: dizi tarama | 30 | 31 | 37 | 0.0205 | > heap | 22.84x |
| Qwen BPE: dizi tarama + tam kelime kisayolu | 25 | 31 | 35 | 0.0204 | — | 19.08x |
| fastBPE cekirdegi (process_bpe) | 2.66 | 2.50 | 3.43 | 0.0099 | 40–80 | 2.05x |
| LinMaxMatch (trie, en uzun eslesme) | 46 | 51 | 62 | 45 | 200+ | 35.45x |
| Viterbi (en az token, trie) | 21 | 18 | 31 | 32 | — | 16.29x |
| Yalniz on-bolme (regex elle) | 150 | 159 | 154 | 455 | — | 116.10x |
| echo bugunku (tokenizer.zig kopyasi) | 0.0017 | — | — | — | — | 0.00x |
| Qwen resmi tokenizer.json (HF, referans) | 1.29 | 1.55 | 1.41 | 2.35 | — | 1.00x |
| HF tokenizers (Rust) | 1.58 | 1.60 | 1.65 | 2.72 | 10–25 | 1.22x |
| tiktoken (Rust) | 9.26 | 11 | 9.78 | 6.33 | 30–60 | 7.16x |
| kitoken (Rust) | 7.26 | 8.88 | 7.29 | 6.90 | 60–120 | 5.61x |
| SentencePiece (uyarlanmis, on-bolme Python) | 1.20 | 1.60 | 1.43 | 3.43 | 20–50 | 0.93x |
| Aho-Corasick (pyahocorasick C, on-bolme Python) | 2.84 | 3.22 | 2.84 | 11 | ~500 | 2.20x |

## 2. Resmi tokenizer ile birebir aynı bölünen paragraf

| Yöntem | TR | EN | Kod | Toplam | Token sayısı (TR) |
|---|---|---|---|---|---|
| llama.cpp (libllama) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| Qwen BPE: heap (oncelik kuyrugu) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| Qwen BPE: dizi tarama | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| Qwen BPE: dizi tarama + tam kelime kisayolu | 1737/1740 | 2734/2734 | 2428/2428 | %99.96 | 376993 (ref 377001) |
| fastBPE cekirdegi (process_bpe) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| LinMaxMatch (trie, en uzun eslesme) | 579/1740 | 1038/2734 | 787/2428 | %34.83 | 373918 (ref 377001) |
| Viterbi (en az token, trie) | 623/1740 | 1332/2734 | 955/2428 | %42.16 | 372047 (ref 377001) |
| echo bugunku (tokenizer.zig kopyasi) | 0/8 | — | — | %0.00 | 2167 (ref 377001) |
| Qwen resmi tokenizer.json (HF, referans) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| HF tokenizers (Rust) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| tiktoken (Rust) | 1737/1740 | 2734/2734 | 2428/2428 | %99.96 | 376993 (ref 377001) |
| kitoken (Rust) | 1737/1740 | 2734/2734 | 2428/2428 | %99.96 | 376993 (ref 377001) |
| SentencePiece (uyarlanmis, on-bolme Python) | 1740/1740 | 2734/2734 | 2428/2428 | %100.00 | 377001 (ref 377001) |
| Aho-Corasick (pyahocorasick C, on-bolme Python) | 579/1740 | 1038/2734 | 787/2428 | %34.83 | 373918 (ref 377001) |

## 3. 10 000 token prompt'ta tokenizer payı (TR metin)

TR'de ölçülen bayt/token: **2.61** (EN 3.40, kod 2.92). 10 000 token = 25 KB.

| Yöntem | Tokenize (ms) | Pay @ kayıpsız pp512 823 tok/s | Pay @ kayıplı pp512 2109 tok/s | Pay @ metindeki varsayım 3000 tok/s |
|---|---|---|---|---|
| llama.cpp (libllama) | 21.86 | %0.180 | %0.459 | %0.652 |
| Qwen BPE: heap (oncelik kuyrugu) | 1.50 | %0.012 | %0.032 | %0.045 |
| Qwen BPE: dizi tarama | 0.88 | %0.007 | %0.019 | %0.026 |
| Qwen BPE: dizi tarama + tam kelime kisayolu | 1.06 | %0.009 | %0.022 | %0.032 |
| fastBPE cekirdegi (process_bpe) | 9.81 | %0.081 | %0.206 | %0.293 |
| LinMaxMatch (trie, en uzun eslesme) | 0.57 | %0.005 | %0.012 | %0.017 |
| Viterbi (en az token, trie) | 1.24 | %0.010 | %0.026 | %0.037 |
| Yalniz on-bolme (regex elle) | 0.17 | %0.001 | %0.004 | %0.005 |
| echo bugunku (tokenizer.zig kopyasi) | 15211.29 | %55.593 | %76.236 | %82.025 |
| Qwen resmi tokenizer.json (HF, referans) | 20.15 | %0.166 | %0.423 | %0.601 |
| HF tokenizers (Rust) | 16.52 | %0.136 | %0.347 | %0.493 |
| tiktoken (Rust) | 2.81 | %0.023 | %0.059 | %0.084 |
| kitoken (Rust) | 3.59 | %0.030 | %0.076 | %0.108 |
| SentencePiece (uyarlanmis, on-bolme Python) | 21.75 | %0.179 | %0.457 | %0.648 |
| Aho-Corasick (pyahocorasick C, on-bolme Python) | 9.17 | %0.075 | %0.193 | %0.274 |

## 4. Ham gerçekler

- Trie düğüm sayısı: 591,554. Yoğun Aho-Corasick tablosu (düğüm × 256 × 4 B) ≈ **606 MB** → L2/L3'e sığmaz (Ryzen 5 3600 L3 = 32 MB).
- Bayt/token: TR 2.61, EN 3.40, kod 2.92 (metindeki varsayım: 4).
- `echo bugünkü` satırı: `tokenizer.zig:137` algoritmasının C kopyası, yalnız ilk 4 KB TR üzerinde (çok yavaş olduğu için).
- Python ile ölçülenlerde (HF, tiktoken, kitoken, SPM) Python çağrı + id listesi oluşturma maliyeti dahil; tek çağrı / tüm dosya olduğu için pay küçük.
- SentencePiece byte-level GPT-2 sözlüğünü doğrudan yükleyemiyor: puan = −merge sırası verilerek uyarlandı, ön-bölme Python `regex` ile yapıldı (süreye dahil).
- Aho-Corasick: pyahocorasick (C) `iter_long` = soldan en uzun eşleşme; kelimeler `\x00` ile ayrılır, ön-bölme Python `regex` (süreye dahil).
- fastBPE: yalnız `process_bpe` birleştirme çekirdeği kullanıldı (kendi ön-bölmesi boşlukla böler, Qwen'e uymaz); ön-bölme bizimki.
