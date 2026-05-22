# echo-core — Proje Yol Haritası

> Son güncelleme: 2026-05-22 (oturum #3)
> Aktif model: `Qwopus3.5-9B-coder-Exp-Q4_K_S.gguf` (5.0 GB)
> Zig versiyonu: 0.17.0-dev.248+95507faf1

---

## 1. Build Onarımı (Ön Koşul)

**Durum:**  **TAMAMLANDI**

### 1.1 `**` operator whitespace hatası
- **Durum:**  **Çözüldü** — `@memset(&raw, 0)` kullanılıyor
- **Doğrulama:** `zig build` çıktısız bitiyor

### 1.2 `zig build test` runner hatası
- **Durum:**  **Workaround** — Zig 0.17 socket/listener bug'ı, test'ler direkt binary'den koşuluyor
- **Doğrulama:** 95/95 test geçiyor (5 test dosyası)
- **Çözüm:** Test binary'lerini elle çalıştır (`./zig-cache/o/*/test --cache-dir=.zig-cache`)

---

## 2. Model Dump & Tensor Yapısını Doğrulama

**Durum:**  **TAMAMLANDI**

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
- Tensor tipleri: 241×q4_k, 177×f32, 8×q5_k, 1×q6_k

---

## 3. GGUF Reader — Gerçek Model Load Testi

**Durum:**  **TAMAMLANDI**

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

**Durum:**  **TAMAMLANDI**

### 4.1 Tensor isimlerinden layer type detection
- **Durum:**  **Tamamlandı**
- `classifyLayerType()` → `full_attention_interval > 0` ise `qwen35LayerType()` kullanır
- Layer 0-2,4-6,8-10,...: qwen_linear (24 adet)
- Layer 3,7,11,15,19,23,27,31: attention (8 adet)
- Pattern: `(layer_idx + 1) % 4 == 0 → .attention`

### 4.2 num_ssm_layers hesaplaması
- **Durum:**  **Tamamlandı**
- `num_ssm_layers = num_layers - (num_layers / full_attention_interval)`
- 32 - (32/4) = 24 qwen_linear/ssm layer

### 4.3 Fused QKV offset fix
- **Durum:**  **Tamamlandı**
- `memory.zig:357-365` — K/V offset'leri fused QKV data region'ın içine bakacak şekilde override
- `k_proj_offset = q_proj_offset + q_dim × row_size`
- `v_proj_offset = q_proj_offset + (q_dim + kv_dim) × row_size`

---

## 5. WeightLoader — Tensörleri Weight Pool'a Yükleme

**Durum:**  **TAMAMLANDI** (temel işlev çalışıyor)

### 5.1 Tüm tensor'leri load
- **Durum:**  Çalışıyor — 427 tensor yüklendi, weight pool 5.3 GB
- **Debug çıktısı:** "Weights loaded successfully"

### 5.2 Fused QKV dtype propagation
- **Durum:**  **Tamamlandı**
- `inference.zig:910-915` — K/V ayrı tensor olarak bulunamazsa, fused QKV dtype'ı K/V slot'larına kopyalanır
- **Öncesi:** K/V dtype'ları .f16 kalıyordu → attention yanlış compute
- **Sonrası:** K/V dtype'ları fused QKV'nin gerçek dtype'ı (q5_k/q4_k) oluyor

### 5.3 SSM tensor dtype'ları
- **Durum:**  **Dtype'lar doğru yükleniyor** — slot 11-18 doğru dtype'ları alıyor
- **Ama SSM path (.ssm layer'lar) hala matvecDispatch (dtype'siz) kullanıyordu → FIX EDILDI**
- **qwen_linear path dtype'ları doğru geçiyor**

---

## 6. Engine Forward — İlk Gerçek Inference

**Durum:**  **KISMİ — NaN sorunu var**

### 6.1 Engine.init(gerçek model)
- **Durum:**  Çalışıyor
- Weight pool allocation: 5.3 GB başarılı
- KV cache, SSM states, temp buffers initialize ediliyor

### 6.2 Engine.forward(tek token)
- **Durum:**  Çalışıyor (token embedding → 32 layer → final norm → output proj)
- **Ama:** output logits NaN
- **Tespit edilen:** `ssmForward()` dtype'siz matvec kullanıyordu → FIX EDILDI
- **SSM path fix:** ssmForward artık dtype alıp `matvecDispatchQuant` kullanıyor
- **qwen_linear path:** dtype'lar doğru geçiyor, `matvecDispatchQuant` kullanılıyor

### 6.3 Engine.generate(cümle)
- **Durum:**  **Çalışmıyor** — logits NaN → argmax hep 0 dönüyor → "Merhaba" çıktısı
- **Öncesi:** K/V dtype bug'ı + SSM dtype bug'ı → NaN
- **Şimdi:** dtype fix'leri yapıldı, ama hala NaN
- **Şüpheli:** qwen_linear conv1d boyut uyumsuzluğu — conv_dim=8192 ama conv weight hidden_dim=4096
- **Q6_K output proj:** generic dequant yolu kullanılıyor, doğru ama yavaş

### 6.4 Tokenizer entegrasyonu
- **Durum:**  **KISMİ** — tokenizer çalışıyor ama generate loop'u NaN yüzünden test edilemedi

---

## 7. Tool Executables Fix

**Durum:**  **KISMİ**

### 7.1 dump-model tool
- **Durum:**  **Tamamlandı**
- `dump_model_main.zig` olarak yeniden yazıldı
- Module import sistemi düzeltildi (tüm dosyalar `@import("core_config")` kullanıyor)
- Tensor tipi sayımı eklendi (q4_k/f32/q5_k/q6_k dumping)

### 7.2 analyze-gguf tool
- **Durum:**  **İptal** — Zig 0.17 API uyumsuzluğu. Python alternatifi mevcut.

---

## 8. REPL — İnteraktif Test

**Durum:**  **BLOKE** — 6.3 NaN sorununu bekliyor

### 8.1 REPL başlatma
- **Durum:** Bekliyor
- **Bağımlılık:** 6.3

---

## 9. Performans & OOM Testi

**Durum:**  **BLOKE** — 8'i bekliyor

### 9.1 Benchmark
- **Durum:** Bekliyor
- **Not:** Mevcut Q6_K generic dequant yolu ~40sn/token, Q6_K matvec kernel gerekli

### 9.2 OOM stress test
- **Durum:** Bekliyor

---

## Özet

| # | Görev | Durum | Not |
|---|-------|-------|-----|
| 1 | Build Fix |  | quant.zig + test workaround |
| 2 | Model Dump |  | Python + Zig dump-model |
| 3 | GGUF Reader |  | full_attention_interval eklendi |
| 4 | Layer Detection |  | classifyLayerType güncellendi |
| 5 | Weight Loader |  | Fused QKV dtype propagation eklendi |
| 6 | Engine Forward |  | init+forward OK, generate NaN |
| 7 | Tool Fix |  | dump-model , analyze-gguf  |
| 8 | REPL |  | 6.3 NaN blokesi |
| 9 | Benchmark |  | 8'i bekliyor |

## Yapılan Fix'ler (oturum #3)

1. **matvecQ5K kernel** — 176-byte bloklar, qh high-bit handling
2. **matvecGenericDequant** — Q6_K/Q3_K/IQ tipleri için generic dequant fallback
3. **Fused QKV K/V dtype propagation** — K/V slot'larına fused tensor'ın dtype'ı kopyalanıyor
4. **SSM dtype pass-through** — ssmForward artık dtype alıp matvecDispatchQuant kullanıyor
5. **qwen_linear dtype'ları** — doğru slot mapping ile yükleniyor ve kullanılıyor

## Bloker: NaN in logits

- **Semptom:** Tüm logit'ler NaN → sampleGreedy hep 0 dönüyor
- **Dtype fix'leri yapıldı** ama NaN devam ediyor
- **En olası neden:** qwen_linear conv1d boyut uyumsuzluğu — conv_dim=8192 kullanılıyor ama conv weight hidden_dim=4096
- **Alternatif şüphe:** token embedding stride, SSM state init, final norm dtype
- **Öneri:** 1-token prefill sonrası gizli state'i NaN-check ile izle

## Bağımlılık Grafiği

```
1. Build Fix ✅
  └─ 2. Model Dump ✅
      └─ 7. Tool Fix ⚠️
          └─ 3. GGUF Reader Test ✅
              └─ 4. Layer Detection ✅
                  ├─ 5. Weight Loader ✅
                  │    └─ 6. Engine Forward ⚠️
                  │         └─ 8. REPL 🔴
                  │              └─ 9. Benchmark 🔴
                  └─ 7. Tool Fix (bağımsız dal)
```

## Sıradaki İş — NaN Fix (6.2/6.3)

1. **qwen_linear conv1d dim fix** — weight_idx formülünde conv_dim yerine hidden_dim kullanımını araştır
2. **1-token prefill + hidden_state dump** — NaN'nin hangi layer'da başladığını izle
3. **Q6_K matvec kernel** — output.weight için optimize matvecQ6K yaz (öncelik düşük, doğrulukla ilgili değil)
