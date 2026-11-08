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

İşaret: `[x]` yapıldı ve denendi · `[~]` kısmen / farklı yapıldı (not var) · `[ ]` yapılmadı.

### 1. Tokenizer doğru olsun
- [x] Qwen BPE: GGUF'tan merges okunuyor, sıralı birleştirme (`src/tokenizer/bpe.zig`)
- [x] Qwen3.5 ön-bölme kuralı elle yazıldı (llama.cpp `qwen35` kuralıyla aynı)
- [x] Bayt eşleme: Türkçe `ş ğ ı` hata vermiyor
- [x] Özel tokenlar tanınıyor (`<|im_start|>`, `<think>`, `<tool_call>` …)
- [x] BOS eklenmiyor (GGUF bayrağına uyuluyor)
- [~] Doğrulama: `llama-tokenize` yerine modelin resmi `tokenizer.json`'u ölçü alındı. TR 1740/1740, EN 2734/2734, kod 2428/2428, uzun kelime 1/1 paragraf aynı (`--tok-check`). Not: NFC düzeltmesi yok (`bpe.zig` shortcut yorumu).

### 2. Decode doğru olsun
- [x] `Ġ` → boşluk, `Ċ` → satır sonu
- [x] Yarım UTF-8 karakter tamponlanıyor (`Utf8Stream`)
- [x] Özel tokenlar çıktıya basılmıyor
- [~] Doğrulama: llama.cpp yerine geri çevirme ile yapıldı — decode(encode(metin)) = metin, 4 korpusta tüm paragraflar

### 3. Varsayılanları sohbete göre ayarla
- [x] Grammar varsayılan kapalı (yeni CLI yolunda yok; eski `--grammar` bayrağı eski yolu açar)
- [x] Varsayılan katman atlama yok
- [x] `--fast` (TG ayarı), `--prefill-fast` (prefill ayarı: 613 tok prompt 730 → 1954 tok/s)
- [~] Env'ler çalışmaya devam ediyor; yalnız profil bayrakları eklendi, her env için ayrı bayrak yok
- [x] Doğrulama: README hız tablosu (varsayılan tg 34.8, llama.cpp 30.5 tok/s)

### 4. Chat şablonu
- [~] ChatML elle uygulandı; GGUF'taki şablon okunmuyor (Qwen3.5'e sabit, başka model reddediliyor)
- [x] `system` / `user` / `assistant` rolleri (CLI `--sys`, sunucu `messages`)
- [x] Düşünme modu varsayılan kapalı, `--think` ile açık
- [~] Doğrulama: llama.cpp yerine resmi `chat_template.jinja` ile — 3 örnek (sade, system'li, think açık) token token aynı (`--print-tokens`)

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
- [~] Doğrulama: `curl` + `openai` Python istemcisi. Open WebUI denenmedi. Not: her istek prompt'u baştan işliyor (önek KV yeniden kullanımı yok).

### 11. Kurulum
- [x] Binary adı `echo-core` (`echo` kabuk komutuyla çakıştığı için)
- [x] `zig build -Doptimize=ReleaseFast install --prefix ~/.local` (denendi)
- [x] Deney programları yalnız `-Ddeneyler=true` ile
- [ ] CPU özelliği yoksa anlaşılır hata — yapılmadı; derleme bu makinenin CPU'suna göre, başka CPU'da çalıştırmak için `-Dcpu=` ile derlemek gerekir

### 12. Regresyon kapısı
- [x] `--llama-bench` eski yol çalışıyor (tg 1133 tok/s, `--fast` ayarı)
- [x] Kalite: llama.cpp'ye karşı 256 token — varsayılan uyum %96.5 ppl 13.83 (llama.cpp 13.68); `--exact` uyum %99.2 ppl 13.68
- [~] Greedy = llama.cpp greedy: `--exact` ile ilk ~25-30 token aynı, sonra ayrışıyor (3 prompttan 1'i tamamen aynı)

### 13. README
- [x] "5 dakikada başla"
- [x] Mod / hız / kalite tablosu

## Bitti tanımı
- [x] `echo-core Qwen3.5-0.8B-Q8_0.gguf` → Türkçe soru → akarak Türkçe cevap
- [~] Greedy çıktı llama.cpp ile token token aynı — `--exact` ile kısmen (yukarıda)
- [x] Varsayılan tg llama.cpp'nin altında değil (34.8 vs 30.5 tok/s)
- [x] `curl` ile `/v1/chat/completions` çalışıyor

## Açık kalanlar
- Greedy'de llama.cpp ile tam eşitlik (ayrışma kaynağı: matris çarpımı toplama sırası; R8 kapalıyken de kalıyor)
- Sunucuda önek KV yeniden kullanımı (uzun sohbette her istek baştan işleniyor)
- CPU özelliği kontrolü, NFC
- Sonraki faz: context window
