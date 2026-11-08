# Tokenizer seçenekleri — objektif değerlendirme (8 Kas 2026)

Ölçüm: `calistir.sh` → `SONUC.md`. Karar kullanıcıda.

Karşılaştırma ölçüsü: modelin kendi tokenizer'ı (Qwen resmi `tokenizer.json`, huggingface.co/Qwen/Qwen3.5-0.8B; NFC normalizer, `ignore_merges: false`). llama.cpp bu dosyayla 6902/6902 paragraf aynı (özel token ayırma açıkken). Kapalıyken metin içindeki `<|im_start|>` yazılarında 4 paragraf farklı.

## Metindeki iddialar vs ölçüm

| İddia | Ölçülen | Durum |
|---|---|---|
| tiktoken 30–60 MB/s | 9.5–12 | abartılı (3–5x) |
| HF tokenizers 10–25 MB/s | 1.8–1.9 | abartılı (5–13x) |
| llama.cpp 15–40 MB/s | 2.5–2.8 | abartılı (6–14x) |
| kitoken 60–120 MB/s | 7.6–9.8 | abartılı (8–12x) |
| fastBPE 40–80 MB/s | 2.8–3.7 | abartılı (14–20x) |
| LinMaxMatch 200+ MB/s, BPE ile %99+ aynı | 58–74 MB/s, paragrafların **%34.8**'i aynı | ikisi de yanlış |
| Aho-Corasick ~500 MB/s, L2/L3'e sığar | pyahocorasick (C): ön-bölme dahil 2.8 MB/s, yalnız otomat taraması 10.8 MB/s; 591 554 düğüm → yoğun tablo ~606 MB | ikisi de yanlış |
| Dizi tarama, heap'ten hızlı | normal metinde doğru (31 vs 21 MB/s); 16 KB tek kelimede 360x yavaş | koşullu doğru |
| 1 token ≈ 4 bayt | TR 2.61, EN 3.40, kod 2.92 | TR'de yanlış |
| Prefill 3000 tok/s | bizde 823 (kayıpsız) / 2109 (kayıplı) | bizim için geçersiz |
| "Kayıp binde birin altında" | gerçek BPE'lerde %0.02–0.5 | büyük ölçüde doğru |

## Seçenekler

### A. Kendi Qwen BPE'miz (Zig'e taşınır)
- **+** Resmi tokenizer ile 6902/6902 paragraf birebir aynı (heap ve dizi tarama).
- **+** En hızlı birebir doğru yol: 31 MB/s (TR), 10k token prompt'ta 0.85 ms (prefill'in %0.007–0.025'i).
- **+** Dış bağımlılık yok, tek statik binary kalır.
- **−** Yazılacak kod: ön-bölme (qwen35 regex elle), merge tablosu, kullanıcı tanımlı tokenlar (`<think>`, `<tool_call>`), unicode sınıf tablosu (~44 KB).
- **−** Dizi taramanın en kötü durumu O(n²): boşluksuz 16 KB metinde 0.78 s. Heap ile birlikte (uzun kelimede heap'e geç) kapanır.
- **nötr** Unicode tablosu Python `unicodedata` sürümüne bağlı. Bu korpusta fark çıkmadı, egzotik karakterde çıkabilir.

### B. llama.cpp tokenizer'ını bağlamak (libllama)
- **+** Resmi tokenizer ile 6902/6902 aynı (özel token ayırma açık), sıfır kod.
- **−** 2.7 MB/s, kendi BPE'mizden 11x yavaş (10k token'da 9.6 ms, %0.08–0.29).
- **−** Yerini aldığımız motora bağımlılık: libllama + ggml paketlenecek.

### C. HF tokenizers (Rust, C binding)
- **+** %100 doğru (kullanıcı tanımlı tokenlar vocab id'leriyle eklendikten sonra). `tokenizer.json` standardı.
- **−** Bu ölçümdeki en yavaşlardan biri: 1.9 MB/s (10k token'da 13.8 ms, %0.11–0.41).
- **−** Rust araç zinciri + C sarmalayıcı (`tokenizers-cpp`) gerekiyor, ayrıca GGUF'ta `tokenizer.json` yok, üretmek gerekiyor.

### D. tiktoken / kitoken (Rust)
- **+** Hazır kütüphaneler arasında en hızlılar: 9.5 / 7.8 MB/s.
- **−** Birebir değil: TR'de 3 paragraf farklı (%99.96). Sebep, "kelime sözlükte varsa tek token" kısayolu. Örnek: `" dikkat"` → llama.cpp `[ d][ikk][at]`, tiktoken `[ dikkat]`. Resmi `tokenizer.json`'da `ignore_merges: false`, yani resmi tokenizer bu kısayolu kullanmıyor.
- **−** Rust bağımlılığı. tiktoken'ın resmi C ABI'si yok.

### E. SentencePiece
- **−** Qwen'in byte-level GPT-2 sözlüğünü doğrudan yükleyemiyor. Uyarlanmış haliyle %100 doğru ama 1.5 MB/s, ön-bölme dışarıda.
- **−** Protobuf bağımlılığı.

### F. fastBPE
- **+** Birleştirme çekirdeği %100 doğru (ön-bölme bizden).
- **−** 2.8 MB/s. String kopyalama ağırlıklı, uzun kelimede 0.011 MB/s. Qwen ön-bölmesi ve özel token desteği yok.

### G. LinMaxMatch / Aho-Corasick / Viterbi (merge'süz)
- **+** Hızlı (Viterbi 30–45, LinMaxMatch 58–74 MB/s), en kötü durumda da doğrusal.
- **−** Paragrafların yalnız %35–42'si birebir aynı, token sayısı %0.8–1.3 az. Model eğitimde görmediği bölünmeleri alır. Kalite etkisi ölçülmedi (`kalite.sh` ile ölçülebilir).
- **ölçüldü** Aho-Corasick (pyahocorasick C, `iter_long`): çıktı LinMaxMatch ile birebir aynı (%34.8). Hız ön-bölme dahil 2.8 MB/s, yalnız otomat taraması 10.8 MB/s. Bizim trie LinMaxMatch'imizden (58 MB/s) yavaş. Yoğun tablo ~606 MB, cache'e sığmaz.

### H. Bugünkü echo tokenizer'ı
- **−** Hiçbir paragraf doğru değil. Hız 0.002 MB/s: 10k token'da 13.5 s, prefill süresinin %53–80'i. Asıl Zig kodu Türkçede `UnknownToken` veriyor.

## Ortak gerçek
Birebir doğru seçeneklerin hepsi 10k token'da prefill'in **%0.5'inden** az yer tutuyor. Hız açısından seçim fark etmiyor. Fark doğruluk, bağımlılık ve en kötü durum davranışında.

## Benim görüşüm (karar değil)
A: Zig'de kendi Qwen BPE'miz. Kısa kelimede dizi tarama, uzun kelimede heap. Kullanıcı tanımlı tokenlar önceden ayrılır. Gerekçe: tek birebir doğru, bağımlılıksız ve en hızlı seçenek bu; en kötü durumu da heap kapatıyor. Doğrulama hazır: `tokbench` referansı ve 6902 paragraf.
