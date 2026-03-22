# Handoff Document — echo-core SimpleTokenizer Task

## Görev Özeti
GGUF'tan okunan vocab ile çalışan `SimpleTokenizer` sınıfı oluşturmak. Başlangıç prompt'u bu dosyanın sonunda.

## Yapılan İşler (TAMAMLANDI)

### 1. include/tokenizer.h (YENİ)
- `TokenData` struct: text, score, type
- `SimpleTokenizer` sınıfı: id_to_token, token_to_id, sorted_tokens_, bos_id, eos_id, tokenizer_type
- Metodlar: `encode()`, `decode()`, `bos()`, `eos()`, `vocab_size()`, `type()`

### 2. src/tokenizer.cpp (YENİ)
- **Constructor**: GGUFReader'dan metadata okuma
  - `tokenizer.ggml.model` → "spm" veya "bpe" belirleme
  - `reader.tokens()` ile vocab alma (metadata map'te move edilmiş oluyor, `reader.tokens()` kullanılmalı)
  - scores, token_type, bos/eos ID okuma
  - `sorted_tokens_` oluşturma (longest-first, greedy matching için)
- **Encode**:
  - SPM: space → ▁ dönüşümü + başında ▁ ekleme
  - BPE: ham metin, GPT-2 byte-to-unicode mapping ile byte fallback
  - Greedy longest-match algoritması
  - BOS token'ı başa ekleme
- **Decode**:
  - Token text'lerini birleştirme
  - BOS/EOS/token_type==3 atlanır
  - BPE: single-byte token'lar GPT-2 reverse mapping ile orijinal byte'a dönüştürülür
  - SPM: ▁ → space, baştaki space trim

### 3. src/inference.cpp (MODİFİYE)
- `reset()` metodu eklendi (KVCache reset)
- `load_weights_from_gguf` sonunda dequantization eklendi:
  - Q8_0 → FP16 (dequantize_q8_0_to_fp16)
  - Q4_K → FP16 (dequantize_q4_K_to_fp16)
  - Q2_K → FP16 (dequantize_q2_K_to_fp16)
  - F16 → kopyalama
  - F32 → FP16 (fp32_to_fp16_row)

### 4. include/inference.h (MODİFİYE)
- `reset()` deklarasyonu eklendi
- `matvec_d()` deklarasyonu eklendi (fused dequant denemesi, şu an kullanılmıyor)
- `weight_dtype_` map eklendi (fused denemesi, kullanılmıyor)

### 5. include/kernels/quant.h (MODİFİYE)
- `dequantize_q8_0_to_fp16()` deklarasyonu
- `dequantize_q4_K_to_fp16()` deklarasyonu
- `dequantize_q2_K_to_fp16()` deklarasyonu

### 6. src/kernels/quant.cpp (MODİFİYE)
- `dequantize_q8_0_to_fp16()` implementasyonu
- `dequantize_q4_K_to_fp16()` implementasyonu
- `dequantize_q2_K_to_fp16()` implementasyonu

### 7. include/kernels/matvec.h (MODİFİYE)
- `matvec_q8_0()`, `matvec_q4_K()`, `matvec_q2_K()` deklarasyonları (fused denemesi)

### 8. src/kernels/matvec.cpp (MODİFİYE)
- Scalar fused dequant+matvec implementasyonları (kullanılmıyor, performans sorunu)

### 9. src/main.cpp (MODİFİYE)
- GGUF parse → tokenizer oluştur → roundtrip test → single forward pass → interactive REPL

### 10. CMakeLists.txt (MODİFİYE)
- `src/tokenizer.cpp` echo_core target'a eklendi

### 11. .gitignore (MODİFİYE)
- `*.gguf` eklendi

## KALAN SORUNLAR

### 1. Engine 5x Yavaş (KRİTİK)
- **Bizim engine**: 789ms/token (1.27 tok/s)
- **llama.cpp**: ~154ms/token (6.5 tok/s)
- **Sebep bilinmiyor**: Dequantization 10.8s (bir kez, loading'de) kaldırıldı ama forward pass hâlâ yavaş
- **Şüphelenilen nedenler**:
  1. Single-threaded matvec (llama.cpp multi-threaded)
  2. FP16 matvec AVX2 tiling verimsiz (her row için input vector yeniden okunuyor)
  3. Output projection [248320, 2560] çok büyük (635M elements, ~1.2 GB FP16)
- **Düzeltme**: Fused dequant+matvec denendi ama scalar kernel AVX2 FP16'dan 8x yavaştı (138ms/layer vs 16.6ms/layer). AVX2 vectorized kernel gerekiyor.

### 2. Fused Dequant+Matvec Crash
- Scalar `matvec_q2_K` SIGBUS crash veriyor (muhtemelen alignment sorunu)
- Bu kod hâlâ matvec.cpp'de duruyor ama inference.cpp tarafından çağrılmıyor (dequantization approach kullanılıyor)

### 3. max_gen Değeri
- Şu an `main.cpp`'de `max_gen = 16`
- 789ms/token ile 16 token = 12.6 saniye
- Kullanıcı interactive'da metin girdiğinde ~30 saniye beklemesi gerekiyor

## DURUM ÖZETİ
- **Tokenizer**: TAMAM, çalışıyor, 5/5 roundtrip OK
- **Dequantization**: TAMAM, çalışıyor, Q8_0/Q4_K/Q2_K destekleniyor
- **Engine performansı**: SORUNLU, 5x yavaş, ayrı task olarak ele alınmalı
- **Commit**: ATILDI (`eef5206`)

## YENİ OTURUM İÇİN PROMPT

Aşağıdaki prompt'u yeni oturumun başına ver:

```
echo-core projesinde SimpleTokenizer implementasyonu tamamlandı (commit eef5206).
Şimdi engine performansını 5x hızlandırmam gerekiyor.

Mevcut durum:
- Bizim engine: 789ms/token (1.27 tok/s)
- llama.cpp aynı modelde: ~154ms/token (6.5 tok/s)
- Model: Qwen3.5-4B Q8_0 mixed-quant (Q8_0/Q4_K/Q2_K)
- Forward pass: 32 layer × 17ms/layer = 544ms layers + output projection = 789ms total
- Output projection: [248320, 2560] = 635M elements = 1.2 GB FP16

Yapılması gerekenler:
1. src/inference.cpp'deki forward pass profil et (hangi matvec en yavaş?)
2. Multi-threading ekle (std::thread ile matvec paralelleştirme)
3. Output projection matvec optimizasyonu (en büyük bottleneck)
4. Fused dequant+matvec AVX2 vectorized kernel (memory bandwidth tasarrufu)

Dosyalar:
- include/inference.h, src/inference.cpp: forward pass
- include/kernels/matvec.h, src/kernels/matvec.cpp: matvec kernel'ları
- include/kernels/quant.h, src/kernels/quant.cpp: dequant fonksiyonları
```

## BAŞLANGIÇ PROMPT'U (orijinal)

```
Görev: GGUF'tan okunan vocab ile çalışan bir 'SimpleTokenizer' sınıfı oluştur.

--- GGUF'tan Okunacak Alanlar ---
Şu metadata key'lerini oku:
  tokenizer.ggml.model      → string: tokenizer algoritması ("llama"=SPM, "gpt2"/"qwen2"=BPE vb.)
  tokenizer.ggml.tokens     → string array: token metinleri, index = token ID
  tokenizer.ggml.scores     → float32 array: SPM log-probability skorları (BPE'de kullanılmaz)
  tokenizer.ggml.token_type → int32 array: token türleri
                               1=normal, 2=unknown, 3=control, 4=user_defined, 6=byte
  tokenizer.ggml.bos_token_id → uint32: sequence başına eklenecek BOS token ID
  tokenizer.ggml.eos_token_id → uint32: sequence sonu EOS token ID

--- Veri Yapıları ---
struct TokenData {
    std::string text;
    float       score;
    int32_t     type;
};

class SimpleTokenizer {
    std::vector<TokenData>              id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    uint32_t bos_id, eos_id;
    std::string tokenizer_type;
};

--- Encode Algoritması ---
1. SPM ise: boşlukları ▁ ile değiştir, başına ▁ ekle
2. Greedy longest-match uygula
3. Eşleşme yoksa byte fallback
4. Sonuç başına bos_id ekle

--- Decode ---
- Token text'lerini birleştir
- SPM ise: ▁ → space, baştaki boşluk temizle
- type==3 control token'ları dahil etme
```
