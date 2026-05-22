# echo-core — Proje Yol Haritası

> Son güncelleme: 2026-05-22 (oturum #4)
> Aktif model: `Qwopus3.5-9B-coder-Exp-Q4_K_S.gguf` (5.0 GB)
> Zig versiyonu: 0.17.0-dev.248+95507faf1
> Güncel hız: **0.02 tok/s** (hedef: 50-100 tok/s)

---

## 1-5: Altyapı

**Durum:**  **TAMAMLANDI**

- Build fix (quant.zig whitespace)
- Model dump (Python + Zig)
- GGUF Reader (full_attention_interval)
- Layer Detection (hybrid qwen35)
- Weight Loader (fused QKV dtype propagation)

---

## 6. Engine Forward

**Durum:**  **TAMAMLANDI**

-  Engine.init çalışıyor
-  forward (tek token) anlamlı çıktı
-  generate (cümle) anlamlı çıktı
-  Tokenizer çalışıyor

---

## 7. Tool Executables

**Durum:**  **TAMAMLANDI**
- dump-model:  Tamam
- analyze-gguf:  Python alternatifi

---

## 8. REPL

**Durum:**  **TAMAMLANDI**
- main.zig'de mevcut, çalışıyor
- :quit, :reset, :stats komutları

---

## 9. Performans (OPTIMIZATION)

**Durum:**  **BLOKE — Q6_K AVX2 kernel gerekli**

### 9.1 Benchmark (oturum #4)
```
prefill: 166241ms (4 token) = 0.02 tok/s
decode:  41117ms (1 token) = 0.02 tok/s
```

### 9.2 Q6_K bottleneck analysis
- output.weight: Q6_K, shape [151665, 4096]
- suan: generic dequant → fp16 → f32 → scalar dot
- tahmini AVX2 kazanimi: ~40000ms → ~5ms

---

## Bloker

**Q6_K output projection** — suanki hizin ~99%'i bu.
Cozum: Zig `@Vector` + inline asm ile AVX2 kernel.

---

## Bağımlılık Grafiği

```
1-8 ✅ → 9. Performance (Q6_K AVX2) → 50-100 tok/s
```
