# Tavan ve varsayımlar (2026-11-08)

Prefill 5000 / 7500 / 10000 barajlarında durduğum yer ve bu duruşun dayandığı varsayımlar. Her varsayım kırılabilir; hangisinin neyi açtığı aşağıda.

## 12 saatlik özet (07 Kas 17:00 → 08 Kas 05:40)

| Saat | Ne oldu | Sonuç |
|---|---|---|
| 17–18 | Doğrulama turu yenilendi (H13, H14), Q8_0 ağırlıkları 8 satır dizildi (H20) | Çekirdek hızlandı |
| 18–19 | Aynı girdili matrisler tek geçişte (H22), thread ayarı (H23), f16 ölçekler (H24) | |
| 19–20 | Taslak uyarlamalı (H25), i16 doygun toplama (H28), asm çekirdek (H34) | **TG 200** (270–277) |
| 20–21 | conv birleştirme, GDN iyileştirmeleri (H35–H39), LM head R8'e taşındı (H40–H41) | |
| 21–22 | GDN kalemleri (H43–H44), önden getirme (H45), iş çalma (H47), n-gram tur 128 (H48–H49) | **TG 500** (507) |
| 22–23 | Kuyruk sütun çekirdeği (H50), LM IVF R8 (H52) | |
| 23–00 | GDN düzeni (H55), R16 çekirdek (H56) | |
| 00–01 | Faz birleştirmeleri (H57, H58, H61, H62) | |
| 01–02 | l2 norm, attention ve GDN tek faz (H63–H66) | 736 → 750 |
| 02–03 | Flash tek faz (H68), RT öncelik + çekirdek sabitleme (H70) | **TG 750** (762) |
| 03–04 | Katman atlama, aktivasyonlu FFN budama, kalite ölçüm aracı | **TG 1000** (1157), **prefill 750/1000/1500** |
| 04–05 | Son katman yalnız son token (N4, N6), ölü yolların ölçümü | **Prefill 2000** (2109) |
| 05–05:40 | Stop hook döngüsü | İş yok |

## Şu anki durum

| Yol | pp512 (tok/s) | tg128 (tok/s) | Top-1 (taban %44.5) |
|---|---|---|---|
| Tam model (`ECHO_SKIP=` boş, `--lm-ivf 4096,512`) | 823 | 821 | %44.5 |
| **Varsayılan** (8 katman atlama, nprobe 128) | 1326 | 1263–1276 | %27.9 |
| Prefill 2000 ayarı (K1920 + 12 katman) | 2109 | — | %18.1 |
| 4 katman kalır + K1024 | 8463 | — | %0.5 |
| 2 katman kalır | ~22000 | — | %0.4 |

Kalite ölçümü: `echo-core-zig/benchmarks/kalite/kalite.sh` (bash man metni, 768 token, ppl + top-1).

## Varsayımlar

| # | Varsayım | Dayanak | Nasıl kırılır | Kırılınca kazanç |
|---|---|---|---|---|
| V1 | Hesap tavanı: çekirdek başına döngüde 1 vpmaddubsw (yalnız FP0, 32 MAC), 6 fiziksel çekirdek, 2.95 GHz (ısıl) → 566 G MAC/s | Sentetik 1.001 döngü/maddubs; SMT ve t≠6 ölçümde kötü; saat ölçümü | Soğutma ile saati yükseltmek (boost 4.2 GHz); CPU dışı birim (kapsam kuralı GPU'yu dışlıyor) | Saat: en çok ~1.4x |
| V2 | Prompt'taki her token tüm katmanlardan geçmeli (token başı 517 M MAC sabit) | Ölçülmedi; varsayım | Prefill'de token budama: derin katmanlarda önemsiz tokenları düşürmek (LazyLLM tarzı). Son katmanda bunu zaten yaptık (N4, N6); diğer katmanlara yaymak. GDN katmanları özyinelemeli olduğu için atılan token durumu değiştirir | Tahmin 1.5–2.5x (literatür; bu modelde ölçülmedi) |
| V3 | Yeniden eğitim yok: budama ve atlama sonrası kalite toparlanmıyor | Hiç eğitim yapılmadı | Budanmış modeli damıtma ya da LoRA ile iyileştirmek (Minitron, Sheared-LLaMA yaklaşımı) | Aynı kalitede 2–4x daha az MAC (tahmin) |
| V4 | Ağırlık 8 bit, çarpma int8 | Q2_K ppl 16.4 → 107; LUT sentetiği doğru birikimle 1.3–2.0x, taşma yok sayılırsa 3.7x | 2 bit + vpshufb LUT çekirdeği; kalite için V3 ile birlikte (2 bite göre eğitim) | 1.3–2x (ölçülen sentetik) |
| V5 | Kalite eşiği: top-1 tabanın en az yarısı, tek İngilizce metin | Benim koyduğum çizgi | Eşiği sen belirle ya da kaldır | Eşik kalkarsa 8463 / ~22000 zaten ölçüldü |
| V6 | Model sabit: Qwen3.5-0.8B | Proje bu modelle ölçüldü | Daha küçük model | SmolLM2-135M bile tavan ~5340 (llama.cpp 845) |
| V7 | Bench'in her tekrarı sıfırdan hesaplanır (önbellek yok) | llama-bench tanımı | Aynı prompt için KV/durum önbelleğini tekrarlar arasında kullanmak | Sınırsız, ama ölçümün anlamı değişir |
| V8 | Hesap yoğun (seyreklik yok) | SiLU modeli; prefill'de token sütunları birleşince aktif kanal kümesi büyür | Eğitimsiz aktivasyon seyrekliği (TEAL) ya da ReLU'ya çevirip eğitmek (V3 ile) | Prefill'de tahmin ≤1.3x |
| V9 | Bit-aynı blok sonu: çekirdek tavanın %75'inde | Blok sonu işlemleri FP0'dan pay alıyor (syn33) | Prefill'de kabul kısıtı yok, bit-aynılık bırakılabilir (ortak ölçek vb.) | ≤1.33x |
| V10 | Tek makine, tek süreç | Kapsam | Birden çok makineye bölmek | Makine sayısıyla doğrusal |

## Hangisini kırmak yetiyor (823 tok/s tam modelden)

| Hedef | Gereken | Tek başına yeten | Birlikte yetebilecek (tahmin, ölçülmedi) |
|---|---|---|---|
| 5000 | 6.1x | V5 (kalite eşiği yok), V7 (önbellek) | V3 + V4 (eğitimli 2 bit, ~2x) × V3 budama (2–3x) ≈ 4–6x; + V2 (1.5–2x) ile güvenli |
| 7500 | 9.1x | V5, V7 | V2 × V3 × V4 × V9: 1.5–2.5 × 2–4 × 1.3–2 × 1.33 ≈ 5–27x |
| 10000 | 12.2x | V5, V7 | Aynı dörtlü + V1 (saat 1.4x); alt tahmin ~7x, üst ~37x |

En büyük kaldıraç V3 (eğitim): V4 ve V8'in kaliteyi bozmadan kırılmasını sağlıyor. Eğitimsiz kırılabilenler V2 ve V9; ikisi birlikte tahminen 2–3.3x, yani tek başına 5000'e yetmez.
