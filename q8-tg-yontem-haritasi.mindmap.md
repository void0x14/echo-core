---
title: Echo Core — Durum Haritası (8 Kas 2026)
markmap:
  colorFreezeLevel: 2
  maxWidth: 300
  initialExpandLevel: 2
---

# Echo Core — 8 Kas 2026

## 🚩 Buradan başla
- **Ne yapıyoruz?**
  - Qwen3.5-0.8B, yalnız CPU
  - Ryzen 5 3600, kirli ortam
- **Nerede kaldık? (8 Kas)**
  - Hız: TG 1000 ✅, prefill 2000 ✅
  - Kullanılabilirlik ✅ — motor sohbet ediyor
    - `echo-core model.gguf` → sohbet
    - `--server` → OpenAI uyumlu API
    - Tokenizer resmi ile aynı (NFC dahil)
    - `--exact` llama.cpp ile token token aynı
    - Sunucu önceki turları yeniden işlemiyor
- **Modlar (tg tok/s)**
  | Mod | tg | Not |
  |---|---|---|
  | llama.cpp | 30.5 | referans |
  | varsayılan | 34.8 | ppl +%1.1 |
  | `--exact` | 36.5 | llama.cpp ile aynı |
  | `--fast` | 1180 | sohbet setinde (8 Kas akşam) |
- **Açık kalanlar**
  - Varsayılan mod llama.cpp ile birebir değil (H38'den beri)
  - Araç çağırma (tool_calls) yok
- **`--fast` sohbet seti (8 Kas akşam)** ✅ 1180
  - Ölçüm: `python ~/sohbet-seti/olcum.py` (25 sohbet, 2119–5745 tok)
  - 18 katman atla: 109.8 → 749
  - FFN 1920 (aktivasyon dosyası): → 961
  - n-gram derece 2, en kısa 3: → ~1020
  - en uzun taslak 128: → **1180**
  - Kalite ölçülmedi
- **Sıradaki iş**
  - Context window
- **İlk okunacak**
  - `KULLANILABILIRLIK-PLANI.md`
  - `README.md` (5 dakikada başla)
  - `TAVAN-VE-VARSAYIMLAR.md`

## 📊 Barajlar
- **TG (üretim)**
  | Baraj | Durum | Nasıl |
  |---|---|---|
  | 70 → 500 | ✅ | çekirdek + n-gram |
  | 750 | ✅ 762 | kayıpsız |
  | 1000 | ✅ ~1270 | 8 katman atla |
  | 1000 (sohbet seti) | ✅ 1180 | 18 atla + FFN 1920 + n-gram 2/3/128 |
- **Prefill (pp512)**
  | Baraj | Durum | Kalite (top-1) |
  |---|---|---|
  | 750 | ✅ 823 | kayıpsız |
  | 1000 | ✅ 1150 | %27.9 |
  | 1500 | ✅ 1613 | %22.9 |
  | 2000 | ✅ 2109 | %18.1 |
  | 5000+ | ❌ | %0.5 altı |
- **Taban kalite:** top-1 %44.5, ppl 18.2

## ⚙️ Nasıl çalıştırılır
- **Derle**
  - `cd echo-core-zig`
  - `zig build -Doptimize=ReleaseFast`
- **Varsayılan (TG 1000)**
  - Ek ayar gerekmez
  - 8 katman atlanır, nprobe 128
- **Tam model (kayıpsız)**
  - `ECHO_SKIP=` (boş)
  - `--lm-ivf 4096,512`
- **Prefill 2000 ayarı**
  - `ECHO_FFN_KEEP=1920`
  - `ECHO_FFN_ACT=echo-core-zig/benchmarks/ffn-act-qwen35-0.8b-q8.bin`
  - `ECHO_SKIP=20,7,18,11,12,8,5,13,10,9,6,17`
- **Bench**
  - `--llama-bench -m ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf -r 10 -p 0 -n 128 --grammar 0`
  - Prefill: `-p 512 -n 0`
- **Kalite ölç**
  - `echo-core-zig/benchmarks/kalite/kalite.sh`
  - 768 token, ppl + top-1

## 🕐 Son 12 saat
- 17–20: R8 çekirdek, asm → **TG 200**
- 20–22: GDN, iş çalma, n-gram 128 → **TG 500**
- 22–02: faz birleştirme (H55–H66)
- 02–03: flash tek faz, RT öncelik → **TG 750**
- 03–04: katman atlama + budama → **TG 1000**
- 04–05: son katman tek token → **Prefill 2000**
- 05+: stop hook döngüsü, iş yok

## 🧱 Neden 5000 yok
- **Donanım tavanı**
  - 566 G MAC/s (int8, 6 çekirdek)
  - Model: 517 M MAC/token
  - Tam model en çok ~1095 tok/s
- **Şu an**
  - Kayıpsız 823 = tavanın %75'i
- **5000 için**
  - Hesabın ≤ %22'si kalmalı
  - Budayınca model bozuluyor
- **Ölçülen uç ayarlar**
  - 8463 tok/s → top-1 %0.5
  - ~22000 tok/s → top-1 %0.4
  - Çıktı rastgele, işaretlenmedi

## 🔓 Kırılacak varsayımlar
- **Eğitimsiz kırılabilir**
  - V2 her token her katmanda → token budama (~1.5–2.5x)
  - V9 bit-aynılık → bırak (≤1.33x)
  - V1 saat 2.95 GHz → soğutma (≤1.4x)
- **Eğitim ister (V3)**
  - V3 eğitim yok → damıtma / LoRA (2–4x)
  - V4 8 bit → 2 bit + LUT (1.3–2x)
  - V8 yoğun hesap → seyreklik (≤1.3x)
- **Ölçümü değiştirir**
  - V5 kalite eşiği yok → hemen geçer
  - V7 önbellek → sınırsız, hile sayılır
- **Kapsam**
  - V6 küçük model → 135M tavanı ~5340
  - V10 çok makine → doğrusal
- **Ne yeter?**
  - 5000: V3 + V4 + V2
  - 7500: V2 + V3 + V4 + V9
  - 10000: + V1

## 💀 Ölen yollar (tekrar deneme)
- Ağırlık normlu FFN budama (ppl 1089)
- Bileşen atlama (FFN / mixer ayrı)
- Hibrit int8 + FMA (≤%2.5)
- n-gram sırası büyütmek (4 yavaş; 2 en iyi)
- Tiled flash her boyutta
- nprobe < 128 (sohbet setinde de: 64 ≈ 128, 32/16 yavaş)
- 19. katman atlama (en iyi 829 < 925)
- n-gram taslağı 192/256 (128'den yavaş)
- SPLIT=4, hsum transpoze, mwaitx, nice
- SmolLM2-135M (tavan yetmiyor)

## ⚠️ Tuzaklar
- **Kabul 1270 kaliteyi kanıtlamaz**
  - Bozuk model de döngüde kabul alır
  - Her kayıplı ayarda `kalite.sh`
- **Budamada hep `ECHO_FFN_ACT`**
- **Faz içi aralık kümesi**
  - rangeInitS 0/1 ayrı
  - Aynı küme → ABA hatası
- **Kayıpsız değişiklikte**
  - `bayt.sh` + kabul + `pair.sh`
- **Varsayılan artık kayıplı**
  - Tam model: `ECHO_SKIP=`

## 📁 Dosyalar
- `README.md` → baraj listesi
- `TAVAN-VE-VARSAYIMLAR.md` → tavan + V1–V10
- `ISLEYIS-ARASTIRMA.md` → tüm deney logu
- `echo-core-zig/src/llama/qwen35.zig` → model, atlama, budama
- `echo-core-zig/src/llama/ops.zig` → çekirdek fazlar
- `docs/eski-haritalar/` → 6 Kas haritası (Sezaryen planı)
