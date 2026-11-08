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

### 1. Tokenizer doğru olsun (en kritik)
- [ ] Qwen BPE: GGUF'tan `tokenizer.ggml.merges` oku, merge sıralı BPE uygula
- [ ] Qwen2 ön-bölme (pre-tokenizer regex) — llama.cpp `LLAMA_VOCAB_PRE_TYPE_QWEN2` ile aynı
- [ ] Byte-level eşleme: her bayt → GPT-2 unicode, sonra BPE (Türkçe `ş ğ ı` artık hata vermez)
- [ ] Özel tokenlar metin içinde tanınsın: `<|im_start|>`, `<|im_end|>`, `<think>` vb.
- [ ] BOS: Qwen BOS eklemez — GGUF `add_bos_token` bayrağına uy
- [ ] Doğrulama: 20 prompt (TR, EN, kod, emoji) için token id'leri `llama-tokenize` ile birebir aynı

### 2. Decode doğru olsun
- [ ] GPT-2 unicode → bayt geri çevirme (`Ġ` → boşluk, `Ċ` → satır sonu)
- [ ] Yarım UTF-8 tamponu: çok baytlı karakter iki tokena bölünürse ikisi gelene kadar basma
- [ ] Özel tokenları çıktıya basma (`<|im_end|>` görünmesin)
- [ ] Doğrulama: aynı id dizisi için çıktı metni llama.cpp ile bayt bayt aynı

### 3. Varsayılanları sohbete göre ayarla
- [ ] Grammar varsayılan **kapalı**; `--grammar json` ile açılsın
- [ ] Varsayılan **kayıpsız** (tam model, katman atlama yok)
- [ ] Hız profilleri bayrakla: `--fast` (TG 1000 ayarı), `--prefill-fast` (prefill 2000 ayarı)
- [ ] `ECHO_SKIP`, `ECHO_FFN_KEEP`, `ECHO_FFN_ACT` env'leri CLI bayrağına taşınsın (env yine çalışsın)
- [ ] Doğrulama: varsayılan tg128 kayıpsız baseline'dan düşmesin (tablo ile)

### 4. Chat şablonu
- [ ] GGUF `tokenizer.chat_template`'i oku; Qwen ChatML'i elle uygula (jinja motoru yazma)
- [ ] `system` / `user` / `assistant` rolleri
- [ ] Qwen3.5 düşünme modu: `<think>…</think>` — varsayılan kapalı, `--think` ile açık; açıkken düşünce gizlenebilsin
- [ ] Doğrulama: aynı mesajlar için şablon çıktısı llama.cpp `--chat-template` ile aynı token dizisi

### 5. Durma koşulları
- [ ] EOS + `<|im_end|>` + `<|endoftext|>` → dur
- [ ] `--stop "metin"` (birden fazla)
- [ ] `-n -1` = durma tokenına kadar

### 6. Sampling
- [ ] `--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--seed`
- [ ] `--temp 0` = greedy (bugünkü yol, hız aynı)
- [ ] n-gram/spekülatif kabul sampling ile uyumlu olsun (temp>0'da kabul kuralı dağılımı bozmasın)
- [ ] LM-IVF (nprobe 128) yaklaşık logit veriyor: temp>0'da tam softmax mı, IVF mi — ölç, karar ver
- [ ] Doğrulama: `--seed` sabitken iki koşu aynı çıktı

### 7. Akışlı (streaming) çıktı
- [ ] Token üretildikçe stdout'a bas, flush
- [ ] Loglar/istatistik stderr'e; stdout'ta yalnız metin (pipe'lanabilsin)
- [ ] Sonda tek satır özet: `prompt N tok @ X tok/s | gen M tok @ Y tok/s`

### 8. CLI (llama-cli gibi)
- [ ] `-m` model, `-p` prompt, `-f` prompt dosyası, stdin'den prompt
- [ ] `-n`, `-c` (ctx), `-t` (thread), `--sys` (system prompt)
- [ ] `-cnv` / `-i`: sohbet modu (prompt verilmezse varsayılan)
- [ ] `--help` tam liste, bilinmeyen bayrakta hata (sessizce yutmasın)
- [ ] Hatalar insan dilinde: model yok, ctx doldu, desteklenmeyen mimari

### 9. Sohbet (REPL) modu
- [ ] Çok turlu: geçmiş KV cache'te kalsın, her turda yalnız yeni mesaj işlensin
- [ ] Ctrl+C üretimi kessin, programı değil; ikinci Ctrl+C çıksın
- [ ] `/reset`, `/sys <metin>`, `/quit`, çok satırlı giriş (`\` ile)
- [ ] Ctx dolarsa çökme yok: anlaşılır mesaj + `/reset` önerisi (asıl çözüm sonraki faz)

### 10. Sunucu (OpenAI uyumlu)
- [ ] `echo serve -m model.gguf --port 8080`
- [ ] `POST /v1/chat/completions` (stream: SSE), `POST /v1/completions`, `GET /v1/models`
- [ ] Tek istek kuyruğu (paralel slot yok — gerekirse sonra)
- [ ] Doğrulama: `curl` + bir OpenAI istemcisi (ör. Open WebUI) ile sohbet

### 11. Kurulum ve paketleme
- [ ] Binary adı `echo` (veya `echo-cli` + `echo-server`)
- [ ] `zig build -Doptimize=ReleaseFast` tek komut; `zig build install --prefix ~/.local`
- [ ] Deney hedefleri (deney-a, kolektif, saturn…) varsayılan build'den ayrılsın
- [ ] CPU özelliği yoksa (AVX2 vb.) anlaşılır hata

### 12. Regresyon kapısı (her adımdan sonra)
- [ ] `--llama-bench` tg128 / pp512 tablosu (BENCHMARK şablonu) — hız düşmedi
- [ ] `kalite.sh` — top-1 / ppl değişmedi
- [ ] Greedy çıktı = llama.cpp greedy çıktı (aynı prompt, aynı token)

### 13. README
- [ ] "5 dakikada başla": derle, model indir, sohbet et
- [ ] Bayrak tablosu, hız profilleri ve kalite bedelleri

## Bitti tanımı
- [ ] `echo -m Qwen3.5-0.8B-Q8_0.gguf` → Türkçe soru sor → akıcı, doğru Türkçe cevap akarak gelir
- [ ] Greedy çıktı llama.cpp ile token token aynı
- [ ] Varsayılan modda tg128 kayıpsız baseline'ın altına düşmedi
- [ ] `curl` ile `/v1/chat/completions` çalışıyor

## Sıra
1 → 2 → 3 → 4 → 5 → 7 → 6 → 8 → 9 → 12 → 10 → 11 → 13.
İlk dört adım bitince motor "konuşur" hale gelir; gerisi konfor.
