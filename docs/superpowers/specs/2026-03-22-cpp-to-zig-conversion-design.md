# echo-core C++ → Zig Dönüşüm Tasarımı

**Tarih:** 2026-03-22
**Zig Sürümü:** 0.16.0-dev (master)
**Durum:** Tasarım Onaylandı

---

## 1. Felsefi Temel

### Terry Davis
Her şeyi kendin yaz, tam kontrol, minimal sistem. TempleOS felsefesi: kodun her satırını anla, her satırdan sorumlu ol.

### Ken Thompson
"Trusting Trust" - güvenilmeyen kodu çalıştırma. Her bağımlılığı incele, anla, kontrol et. Güncelleme geldiğinde körce merge etme.

### Linus Torvalds
"Good taste" - soyutlama sadece gerektiğinde. Edge case'leri kaldır, data structure'ı düşün. Gereksız abstraction düşmandır.

---

## 2. Zero-Surprise Prensibi

```
Zig std'si → Cerrahi Operasyon → Senin Kodun
     ↓              ↓                ↓
  İncele        Optimize et      Kontrol et
  Anla          Gereksız çıkar   Sorumlu ol
  Güvenme       Kendine dahil    Güncelleme: manuel incele
```

**Ne demek:**
1. Zig std'sinden sadece ihtiyacın olan parçaları al
2. Optimize et, gereksız kısımları çıkar
3. Kendi koduna dahil et (fork etme değil, cerrahi operasyon)
4. Her satırı anla, her satırdan sorumlu ol
5. Yeni Zig sürümü çıktığında: LLM ile incele → diff al → manuel entegre et
6. Körce merge yok, tam kontrol var

---

## 3. KISS Prensipleri

1. **Tek bir amaç:** GGUF LLM inference
2. **Gereksız feature yok** (YAGNI)
3. **Her modül tek bir sorumluluk**
4. **Debug edilebilirlik > performans** (ilk aşamada)
5. **Okunabilirlik > cleverness**
6. **Karmaşıklık sadece gerektiğinde**

---

## 4. Hexagonal Mimari (Zig Adaptasyonu)

### Mimari Diagram

```
                    ┌─────────────────┐
                    │  REPL Adapter    │
                    │ (stdin/stdout)   │
                    └────────┬────────┘
                             │
┌──────────────────┐    ┌────┴────┐    ┌──────────────────┐
│  GGUF Adapter    │───▶│  PORT   │◀───│  Test Adapter    │
│  (file I/O)      │    │ (iface) │    │ (synthetic data) │
└──────────────────┘    └────┬────┘    └──────────────────┘
                             │
                    ┌────────┴────────┐
                    │  DOMAIN CORE    │
                    │ - Inference     │
                    │ - Tokenizer     │
                    │ - Kernels       │
                    │ - KV Cache      │
                    │ - Config        │
                    │ - Memory        │
                    └─────────────────┘
```

### Domain Core (İç Hexagon)
- Hiçbir I/O yok, hiçbir "side effect" yok
- Sadece data transform, pure functions
- Test edilebilir, güvenilir, saf

### Ports (Arayüzler)
- `ModelLoader` port: model yükleme interface
- `TokenizerPort` port: tokenize interface
- `InferencePort` port: inference interface

### Adapters (Dış Bağlantılar)
- `GGUFAdapter`: GGUF dosyasından model yükler
- `REPLAdapter`: stdin/stdout ile kullanıcı etkileşimi
- `TestAdapter`: synthetic data ile test

### Bağımlılık Kuralı
Bağımlılıklar dışarıdan içeriye akar. Domain core hiçbir dış bağımlılığı bilmez.

---

## 5. Modül Yapısı

```
echo-core-zig/
├── build.zig
├── src/
│   ├── main.zig                    # Entry point (adapter)
│   ├── core/
│   │   ├── types.zig               # fp16_t, base types
│   │   ├── config.zig              # ModelConfig, CacheTileConfig
│   │   ├── memory.zig              # AlignedMemoryPool
│   │   └── math.zig                # sqrt, exp, tanh (cerrahi operasyon)
│   ├── data/
│   │   ├── arraylist.zig           # Minimal ArrayList (cerrahi operasyon)
│   │   ├── hashmap.zig             # Minimal HashMap (cerrahi operasyon)
│   │   └── string.zig              # String utilities
│   ├── kernels/
│   │   ├── quant.zig               # Quantization/dequantization
│   │   └── matvec.zig              # AVX2 matvec kernels (inline assembly)
│   ├── gguf/
│   │   └── reader.zig              # GGUF v3 parser
│   ├── inference/
│   │   └── engine.zig              # InferenceEngine
│   ├── tokenizer/
│   │   └── tokenizer.zig           # SimpleTokenizer
│   ├── kv_cache/
│   │   └── cache.zig               # KVCache
│   └── ports/
│       ├── loader.zig              # ModelLoader port
│       └── inference.zig           # InferencePort port
└── tests/
    ├── test_gguf.zig
    ├── test_tokenizer.zig
    ├── test_inference.zig
    └── test_kernels.zig
```

---

## 6. Cerrahi Operasyon Planı (Zig Std'den Alınacaklar)

### Alınacak Parçalar

| Parça | Kaynak | Optimizasyon |
|-------|--------|-------------|
| Allocator interface | `std.mem.Allocator` | Interface'i al, implementasyonu kendin yaz |
| ArrayList | `std.array_list.zig` | Minimal versiyon, gereksız methodları çıkar |
| HashMap | `std.hash_map.zig` | Open addressing, FNV-1a hash |
| Math | `std.math.zig` | Sadece sqrt, exp, tanh, sqrtf |
| SIMD | `std.simd.zig` | @Vector operations, AVX2 için |
| File I/O | `std.fs.File.zig` | Sadece open/read/close |

### Kendimiz Yazacağımız
- Syscall wrapper'ları (Linux x86_64 inline assembly)
- AVX2 inline assembly kernel'lar
- Memory pool (aligned allocation)
- GGUF parser (binary format)
- Tokenizer (SPM/BPE)
- KV cache (INT8/FP32)
- Inference engine (forward pass)

---

## 7. Binary Boyutu Optimizasyonu

### Mevcut Durum
C++ → Zig portunda %77 daha büyük (Factor VM örneği, Mart 2026).

### Optimizasyon Stratejisi

1. **ReleaseSmall mode** kullan (ReleaseFast değil)
2. **Dead code elimination** → sadece kullanılan fonksiyonları dahil et
3. **Comptime'da gereksız kodları çıkar** → unused code compile edilmeyecek
4. **Inline assembly ile minimal syscall wrapper'ları** → küçük binary
5. **Sadece kullanılan std fonksiyonlarını dahil et** → cerrahi operasyon
6. **Strip symbols** → debug bilgilerini çıkar
7. **Gereksız dependency'leri kaldır** → cerrahi operasyon

### Hedef
C++ binary boyutundan daha küçük veya aynı.

---

## 8. Versiyon İzolasyonu

```
Yeni Zig sürümü → LLM ile incele → Diff al → Manuel entegre et
     ↓                 ↓              ↓            ↓
  0.17.0 çıktığında   İncele        Değişiklikleri   Beğendiğin
  hatırla              her satırı    belirle          parçaları
                       anla                         ekle
```

### Güncelleme Akışı
1. Zig güncellese bile senin kodun değişmiyor
2. Yeni sürümü LLM'lerle birlikte inceliyorsun
3. Diff alıyorsun (eski std vs yeni std)
4. Beğendiğin parçaları manuel olarak entegre ediyorsun
5. Körce merge yok, tam kontrol var

---

## 9. Implementasyon Sırası

1. `core/types.zig` → fp16_t, base types
2. `core/config.zig` → ModelConfig, CacheTileConfig
3. `core/memory.zig` → AlignedMemoryPool
4. `core/math.zig` → sqrt, exp, tanh (cerrahi operasyon)
5. `data/arraylist.zig` → Minimal ArrayList (cerrahi operasyon)
6. `data/hashmap.zig` → Minimal HashMap (cerrahi operasyon)
7. `data/string.zig` → String utilities
8. `kernels/quant.zig` → Quantization/dequantization
9. `kernels/matvec.zig` → AVX2 matvec kernels (inline assembly)
10. `gguf/reader.zig` → GGUF v3 parser
11. `kv_cache/cache.zig` → KVCache
12. `tokenizer/tokenizer.zig` → SimpleTokenizer
13. `inference/engine.zig` → InferenceEngine
14. `ports/loader.zig` → ModelLoader port
15. `ports/inference.zig` → InferencePort port
16. `main.zig` → REPL + test harness

---

## 10. Test Stratejisi

1. Unit test: her modül için
2. Integration test: GGUF → inference
3. Performance benchmark: C++ vs Zig
4. Cross-version test: farklı Zig sürümleri

---

## 11. Başarı Kriterleri

1. Zero-surprise: her satırı anlıyorum, her satırdan sorumluyum
2. Binary boyutu: C++'dan daha küçük veya aynı
3. Performans: C++'dan en az %80 hız
4. Versiyon izolasyonu: yeni Zig sürümünden etkilenmeme
5. Sürdürülebilir, temiz kod
6. KISS prensiplerine uygunluk
