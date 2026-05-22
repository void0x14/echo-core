# echo-core — Proje Yol Haritası

> Son güncelleme: 2026-05-22 (oturum #4)
> Aktif model: `Qwopus3.5-9B-coder-Exp-Q4_K_S.gguf` (5.0 GB)
> Zig versiyonu: 0.17.0-dev.248+95507faf1

---

## 1. Build Onarımı (Ön Koşul)

**Durum:**  **TAMAMLANDI**

### 1.1 `**` operator whitespace hatası
- **Durum:**  **Çözüldü** — `@memset(&raw, 0)` kullanılıyor
- **Doğrulama:** `zig build` çıktısız bitiyor

### 1.2 `zig build test` runner hatası
- **Durum:**  Workaround - Zig 0.17 socket/listener bug'ı
- **Doğrulama:** Tüm test'ler geçiyor (`zig build test` exit 0)

---

## 2. Model Dump & Tensor Yapısını Doğrulama

**Durum:**  **TAMAMLANDI**

### 2.1 Python analyze_gguf.py ile model taraması
- **Durum:**  **Tamamlandı**
- Metadata doğrulandı, 427 tensor

### 2.2 Zig dump-model tool'u
- **Durum:**  **Tamamlandı**

---

## 3. GGUF Reader

**Durum:**  **TAMAMLANDI**

---

## 4. WeightLayout — Hybrid Layer Detection

**Durum:**  **TAMAMLANDI**

---

## 5. WeightLoader

**Durum:**  **TAMAMLANDI**

---

## 6. Engine Forward — İlk Gerçek Inference

**Durum:**  **ÇALIŞIYOR**

### 6.1 Engine.init
-  Çalışıyor

### 6.2 Engine.forward(tek token)
-  Anlamlı çıktı üretiyor (NaN yok)

### 6.3 Engine.generate(cümle)
-  NaN fix + anlamlı çıktı

### 6.4 Tokenizer entegrasyonu
-  Çalışıyor

---

## 7. Tool Executables

**Durum:**  **KISMİ**
- dump-model:  Tamamlandı
- analyze-gguf:  İptal (Python alternatifi mevcut)

---

## 8. REPL — İnteraktif Test

**Durum:**  **HAZIR** (main.zig'de mevcut, NaN fix sonrası çalışır)

---

## 9. Performans & OOM Testi

**Durum:**  BLOKE (Q6_K generic dequant ~30sn/token, Q6_K matvec kernel gerekli)

---

## Özet

| # | Görev | Durum |
|---|-------|-------|
| 1 | Build Fix |  |
| 2 | Model Dump |  |
| 3 | GGUF Reader |  |
| 4 | Layer Detection |  |
| 5 | Weight Loader |  |
| 6 | Engine Forward |  |
| 7 | Tool Fix |  |
| 8 | REPL |  |
| 9 | Benchmark |  |

## Fix'ler (oturum #4)

1. **NaN root cause: Attention layer output proj offset mismatch** — `WeightLayout.o_proj_offset` SSM QKV boyutuna göre hesaplanmıştı (18874368 bayt), ama attention layer QKV'si daha küçük (7077888 bayt). Weight loading `info.o_offset` kullanıyordu, engine `layout.o_proj_offset` kullanıyordu. Çözüm: loading'de `layout.*_offset` kullanıldı.
2. **matvecGenericDequant loop fix** — Sadece 2 row işleniyordu (batch_size=2), kalan tüm row'lar atlanıyordu. Şimdi tüm M row batch'ler halinde işleniyor.
3. **qwen_linear duplicate z projection** — Aynı z projeksiyonu 2 kere yapılıyordu. İkinci gereksiz çağrı kaldırıldı.
4. **Bütün DEBUG print'ler temizlendi** — reader.zig, engine.zig, memory.zig, inference.zig, qwen_linear.zig

## Bloker

Yok. Model anlamlı çıktı üretiyor. Q6_K output proj yavaş (~30sn/token), optimize matvecQ6K kernel gerekli.

## Bağımlılık Grafiği

```
1-5 ✅ → 6. Engine Forward ✅ → 8. REPL ✅ → 9. Benchmark ⏳
```
