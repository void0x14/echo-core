# Echo — Kullanılabilirlik Planı (8 Kas 2026)

Hedef: Hızı koruyarak,çıkarım motorumuzu gerçekten kullanılabilir hale getirmek ve gerçekten agentic iş akışlarında kullanmaya hazır anahtar teslim hale getirmek
Kullanılabilirlik problemi çözüldükten sonra uğraşılcak,çözülcek problem: context window (bu planın dışında).

## Bugünkü durum (ölçüldü, 8 Kas)

| Deneme | Sonuç |
|---|---|
| `-p "The capital of France is"` (varsayılan) | JSON basıyor: `{"name": "France", "age": 1 ...}` — grammar açık + 8 katman atlama |
| Aynı, `ECHO_SKIP=` (tam model) | Yine JSON, ama `"Paris"` doğru |
| `--no-grammar`, tam model | Metin doğru ama `Ġcity ĠofĠParis` — boşluk karakteri çözülmüyor |
| `-p "Türkiye'nin başkenti"` | `error.UnknownToken` — Türkçe karakter tokenize edilemiyor |
| REPL (`Merhaba`) | JSON + `ĠtheĠloveĠof...` döngüsü |

Kök sebepler:
- Tokenizer gerçek BPE değil, açgözlü en uzun eşleşme (`src/tokenizer/tokenizer.zig:137`).
- Decode, GPT-2 bayt→unicode eşlemesini geri çevirmiyor (`Ġ` = boşluk).
- Grammar (H06) ve 8 katman atlama varsayılan açık; bunlar bench ayarı, sohbet ayarı değil.
- Yalnız greedy (`src/llama/port.zig` `greedy()`); sampling yok.
- Chat şablonu yok; model talimat formatında konuşmuyor.

## Adımlar

İşaret: `[x]` yapıldı ve denendi.

### 1. Tokenizer doğru olsun
- [x] Qwen BPE: GGUF'tan merges okunuyor, sıralı birleştirme (`src/tokenizer/bpe.zig`)
- [x] Qwen3.5 ön-bölme kuralı elle yazıldı (llama.cpp `qwen35` kuralıyla aynı)
- [x] Bayt eşleme: Türkçe `ş ğ ı` hata vermiyor
- [x] Özel tokenlar tanınıyor (`<|im_start|>`, `<think>`, `<tool_call>` …)
- [x] BOS eklenmiyor (GGUF bayrağına uyuluyor)
- [x] Doğrulama: ölçü modelin resmi `tokenizer.json`'u. TR 1740/1740, EN 2734/2734, kod 2428/2428, uzun kelime 1/1 paragraf aynı (`--tok-check`)
- [x] NFC (resmi normalizer): NFD'ye çevrilmiş tr/en/kod korpusları + 2000 rastgele aksan yığını resmi tokenizer ile aynı (`src/tokenizer/nfc.zig`, `benchmarks/tokenizer/nfc_test.py`)

### 2. Decode doğru olsun
- [x] `Ġ` → boşluk, `Ċ` → satır sonu
- [x] Yarım UTF-8 karakter tamponlanıyor (`Utf8Stream`)
- [x] Özel tokenlar çıktıya basılmıyor
- [x] Doğrulama: decode(encode(metin)) = metin (4 korpusta tüm paragraflar) ve `--exact` greedy metni llama.cpp ile karakter karakter aynı (5 prompt × 128 token)

### 3. Varsayılanları sohbete göre ayarla
- [x] Grammar varsayılan kapalı (yeni CLI yolunda yok; eski `--grammar` bayrağı eski yolu açar)
- [x] Varsayılan katman atlama yok
- [x] `--fast` (TG ayarı), `--prefill-fast` (prefill ayarı: 613 tok prompt 730 → 1954 tok/s)
- [x] `--skip`, `--ffn-keep`, `--ffn-act` bayrakları (env'ler de çalışıyor)
- [x] Doğrulama: README hız tablosu (varsayılan tg 34.8, llama.cpp 30.5 tok/s)

### 4. Chat şablonu
- [x] ChatML elle uygulandı; GGUF'taki şablon okunup Qwen3.5 ChatML değilse uyarı veriliyor
- [x] `system` / `user` / `assistant` rolleri (CLI `--sys`, sunucu `messages`)
- [x] Düşünme modu varsayılan kapalı, `--think` ile açık
- [x] Doğrulama: ölçü modelin resmi `chat_template.jinja`'sı — 3 örnek (sade, system'li, think açık) token token aynı (`--print-tokens`); sunucuda düşünceli geçmiş mesaj da aynı

### 5. Durma koşulları
- [x] EOS / `<|im_end|>` / `<|endoftext|>` → dur
- [x] `--stop "metin"` (birden fazla)
- [x] `-n -1` = durma tokenına kadar

### 6. Sampling
- [x] `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--presence-penalty`, `--seed`
- [x] `--temp 0` = greedy
- [x] n-gram yalnız greedy'de açılıyor; temp>0'da kapalı (dağılım bozulmaz)
- [x] IVF (`--fast`) yalnız greedy ile çalışıyor (temp 0'a zorlanıyor)
- [x] Doğrulama: aynı `--seed` → aynı çıktı (birim testi + elle)

### 7. Akışlı çıktı
- [x] Token token stdout'a, flush
- [x] İstatistik stderr'de; stdout yalnız metin
- [x] Sonda tek satır özet

### 8. CLI
- [x] `-m`, `-p`, `-f`, stdin
- [x] `-n`, `-c`, `-t`, `--sys`
- [x] `-cnv` / `-i`; prompt yoksa sohbet
- [x] `--help`, bilinmeyen bayrakta hata (çıkış kodu 2)
- [x] Model yok / ctx doldu / desteklenmeyen mimari mesajları

### 9. Sohbet modu
- [x] Çok tur: geçmiş KV'de kalıyor, yalnız yeni mesaj işleniyor
- [x] Ctrl+C üretimi kesiyor; beklerken Ctrl+C çıkıyor
- [x] `/reset`, `/sys`, `/quit`, `\` ile çok satır
- [x] Ctx dolunca çökmüyor, mesaj + `/reset` önerisi

### 10. Sunucu
- [x] `echo-core model.gguf --server --port 8080` (veya `serve`)
- [x] `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/models`, `/health`
- [x] Tek istek sırası
- [x] Doğrulama: `curl`, `openai` Python istemcisi, Open WebUI 0.13.2 (model listesi, akışlı ve çok turlu sohbet, Open WebUI sunucusu üzerinden). Arayüz tarayıcıda açılmadı (bu makinede Chromium engelli).
- [x] Önek yeniden kullanımı: son `<|im_start|>` öncesi durum kaydediliyor; 4 turlu testte 673 tokenın 639'u hazır, cevap sıfırdan işlemeyle aynı

### 11. Kurulum
- [x] Binary adı `echo-core` (`echo` kabuk komutuyla çakıştığı için)
- [x] `zig build -Doptimize=ReleaseFast install --prefix ~/.local` (denendi)
- [x] Deney programları yalnız `-Ddeneyler=true` ile
- [x] CPU özelliği yoksa anlaşılır hata (`src/cpu_kontrol.zig`, main'den önce, temel x86-64 ile derlenir). Denendi: Zen4 için derlenen sürüm bu Zen2'de "eksik CPU özelliği: avx512f …" deyip çıkıyor

### 12. Regresyon kapısı
- [x] `--llama-bench` eski yol çalışıyor (tg 1133 tok/s, `--fast` ayarı)
- [x] Kalite: llama.cpp'ye karşı 256 token — varsayılan uyum %96.5 ppl 13.83 (llama.cpp 13.68); `--exact` bit eşit
- [x] Greedy = llama.cpp greedy: `--exact` ile 5 prompt × 128 token karakter karakter aynı (`benchmarks/greedy-llama.sh`); `--logit-check --exact` 9/9 bit eşit

### 13. README
- [x] "5 dakikada başla"
- [x] Mod / hız / kalite tablosu

## Bitti tanımı
- [x] `echo-core Qwen3.5-0.8B-Q8_0.gguf` → Türkçe soru → akarak Türkçe cevap
- [x] Greedy çıktı llama.cpp ile token token aynı (`--exact`)
- [x] Varsayılan tg llama.cpp'nin altında değil (34.8 vs 30.5 tok/s)
- [x] `curl` ile `/v1/chat/completions` çalışıyor

## Açık kalanlar
- Varsayılan mod llama.cpp ile token token aynı değil (hız hileleri toplama sırasını değiştiriyor; eşitliği ilk bozan H38, `c927a19`, git bisect). Eşitlik gerekirse `--exact`.
- Araç çağırma (tool_calls) sunucuda yok — planda yoktu, agentic kullanım için sıradaki aday
- Sonraki faz: context window
