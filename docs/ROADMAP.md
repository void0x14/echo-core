# echo-core — Proje Yol Haritası

> Son güncelleme: 2026-05-22 (oturum #2)
> Aktif model: `Qwopus3.5-9B-coder-Exp-Q4_K_S.gguf` (5.0 GB)
> Zig versiyonu: 0.17.0-dev.248+95507faf1

---

## 1. Build Onarımı (Ön Koşul)

**Durum:** ** TAMAMLANDI**

### 1.1 `**` operator whitespace hatası
- **Durum:**  **Çözüldü** — `@memset(&raw, 0)` kullanılıyor
- **Doğrulama:** `zig build` çıktısız bitiyor

### 1.2 `zig build test` runner hatası
- **Durum:**  **Workaround** — Zig 0.17 socket/listener bug'ı, test'ler direkt binary'den koşuluyor
- **Doğrulama:** 95/95 test geçiyor (5 test dosyası)
- **Çözüm:** Test binary'lerini elle çalıştır (`./zig-cache/o/*/test --cache-dir=.zig-cache`)

---

## 2. Model Dump & Tensor Yapısını Doğrulama

**Durum:** ** TAMAMLANDI**

### 2.1 Python analyze_gguf.py ile model taraması
- **Durum:**  **Tamamlandı**
- Rapor: `test_models/analysis/Qwopus3.5-9B-coder-Exp-Q4_K_S_tensor_analysis.txt`
- **Metadata doğrulandı:**
  - `general.architecture: qwen35` ✓
  - `block_count: 32` ✓
  - `embedding_length: 4096` ✓
  - `head_count: 16`, `head_count_kv: 4` ✓
  - `full_attention_interval: 4` ✓
  - `ssm.conv_kernel: 4`, `ssm.inner_size: 4096`, `ssm.state_size: 128`, `ssm.time_step_rank: 32` ✓
  - `context_length: 262144` ✓
  - Tensor sayısı: 427 ✓
  - Model boyutu: 4.97 GB

### 2.2 Zig dump-model tool'u aktifleştir
- **Durum:**  **Tamamlandı** — `dump_model_main.zig` ile yeniden yazıldı, modül import düzeltildi
- **Doğrulama:** `zig build` → `./zig-out/bin/dump-model model.gguf` çalışıyor

---

## 3. GGUF Reader — Gerçek Model Load Testi

**Durum:** ** TAMAMLANDI**

### 3.1 Model metadata okuma testi
- **Durum:**  **Tamamlandı** — dump-model tool ile gerçek model dosyası açıldı, metadata doğrulandı
- **Doğrulama:**
  - `config.hidden_dim == 4096` ✓
  - `config.num_layers == 32` ✓
  - `config.num_heads == 16`, `num_kv_heads == 4` ✓
  - `config.head_dim == 256` ✓
  - `config.ssm_inner_size == 4096` ✓
  - `config.ssm_conv_kernel == 4` ✓
  - `config.ssm_dt_rank == 32` ✓
  - `config.max_seq_len == 262144` ✓ (reader capped to 4096)

### 3.2 `full_attention_interval` field'ını ModelConfig'e ekle
- **Durum:**  **Tamamlandı**
- `config.zig:21` → `full_attention_interval: u32` eklendi
- `reader.zig:304-307` → metadatadan okuma eklendi
- `@sizeOf(ModelConfig)` → 72 → 76

---

## 4. WeightLayout — Hybrid Layer Detection

**Durum:** ** TAMAMLANDI**

### 4.1 Tensor isimlerinden layer type detection
- **Durum:**  **Tamamlandı**
- `classifyLayerType()` → `full_attention_interval > 0` ise `qwen35LayerType()` kullanır
- Hibrit olmayan modellerde tensor name-based detection (ssm_alpha/beta/norm kontrolü)
- **Pattern:** 32 layer → 24 SSM + 8 Attention (every 4th)

### 4.2 num_ssm_layers hesaplaması
- **Durum:**  **Tamamlandı**
- `num_ssm_layers = num_layers - (num_layers / full_attention_interval)`
- 32 - (32/4) = 24 SSM layer
- `classifyLayerType` artık `cfg.full_attention_interval` parametresi alıyor

---

## 5. WeightLoader — Tensörleri Weight Pool'a Yükleme

**Durum:**  **BAŞLATMADI**

### 5.1 Attention layer tensörlerini yükle
- **Durum:**  Bekliyor

### 5.2 SSM/QwenLinear layer tensörlerini yükle
- **Durum:**  Bekliyor
- **Bağımlılık:** 4.1  (çözüldü)

---

## 6. Engine Forward — İlk Gerçek Inference

**Durum:**  **BAŞLATMADI**

### 6.1 Engine.init(gerçek model)
- **Durum:**  Bekliyor
- **Bağımlılık:** 5.1, 5.2

### 6.2 Engine.forward(tek token)
- **Durum:**  Bekliyor

### 6.3 Engine.generate(cümle)
- **Durum:**  Bekliyor

### 6.4 Tokenizer entegrasyonu
- **Durum:**  Bekliyor

---

## 7. Tool Executables Fix

**Durum:** ** KISMİ**

### 7.1 dump-model tool
- **Durum:**  **Tamamlandı**
- `dump_model_main.zig` olarak yeniden yazıldı
- Module import sistemi düzeltildi (tüm dosyalar `@import("core_config")` kullanıyor)
- `zig build` + `./zig-out/bin/dump-model model.gguf` çalışıyor

### 7.2 analyze-gguf tool
- **Durum:**  **İptal** — Zig 0.17 API uyumsuzluğu (`std.fs.cwd()`, `argsAlloc`, eski ArrayList API). 
- **Alternatif:** Python script (`analyze_gguf.py`) kullanılabilir

---

## 8. REPL — İnteraktif Test

**Durum:**  **BAŞLATMADI**

### 8.1 REPL başlatma
- **Durum:**  Bekliyor
- **Bağımlılık:** 6.3

### 8.2 REPL generate
- **Durum:**  Bekliyor

---

## 9. Performans & OOM Testi

**Durum:**  **BAŞLATMADI**

### 9.1 Benchmark
- **Durum:**  Bekliyor
- **Bağımlılık:** 8.1

### 9.2 OOM stress test
- **Durum:**  Bekliyor

---

## Özet

| # | Görev | Durum | Not |
|---|-------|-------|-----|
| 1 | Build Fix |  | quant.zig + test workaround |
| 2 | Model Dump |  | Python + Zig dump-model |
| 3 | GGUF Reader |  | full_attention_interval eklendi |
| 4 | Layer Detection |  | classifyLayerType güncellendi |
| 5 | Weight Loader |  | Sıradaki görev |
| 6 | Engine Forward |  | 5'i bekliyor |
| 7 | Tool Fix |  | dump-model , analyze-gguf  |
| 8 | REPL |  | 6'yı bekliyor |
| 9 | Benchmark |  | 8'i bekliyor |

## Bağımlılık Grafiği

```
1. Build Fix ✅
  └─ 2. Model Dump ✅
      └─ 7. Tool Fix ⚠️
          └─ 3. GGUF Reader Test ✅
              └─ 4. Layer Detection ✅
                  ├─ 5. Weight Loader ❌ ← BURADASIN
                  │    └─ 6. Engine Forward ❌
                  │         └─ 8. REPL ❌
                  │              └─ 9. Benchmark ❌
                  └─ 7. Tool Fix (bağımsız dal)
```

## Sıradaki İş — Weight Loader (5.1 + 5.2)

**Aktif model ile weight loader'ı test et:**
1. `loadWeightsFromReader` ile gerçek model tensörlerini weight pool'a kopyala
2. SSM tensor mapping'lerini doğrula (`ssm_a`, `ssm_alpha.weight`, `ssm_beta.weight`)
3. Fused QKV (`attn_qkv.weight`) tensör yüklemesini doğrula
