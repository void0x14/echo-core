# Qwen3.5 Verification Matrix

Bu belge varsayim degil, yeniden uretilebilir yerel kanit kaydidir.

## Dogrulanmis Kapsam

- Hedef model: `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`
- Hedef repo: `echo-core-zig`
- Hedef iddia: onceki `load -> runtime -> matvec/ssm segfault` zinciri artik bu modelde tekrar olusmuyor.

## Yerel Tensor Kaniti

### Linear katman kaniti

Yerel analizden dogrulanan `blk.0` tensorleri:

- `blk.0.attn_qkv.weight`
- `blk.0.attn_gate.weight`
- `blk.0.ssm_a`
- `blk.0.ssm_alpha.weight`
- `blk.0.ssm_beta.weight`
- `blk.0.ssm_dt.bias`
- `blk.0.ssm_conv1d.weight`
- `blk.0.ssm_norm.weight`
- `blk.0.ssm_out.weight`

Bu set, full-attention degil `qwen_linear` katman sozlesmesine gore yukleniyor.

### Full-attention katman kaniti

Yerel parser ile `blk.3` ve `blk.7` icin dogrulanan tensorler:

- `attn_q.weight`
- `attn_k.weight`
- `attn_v.weight`
- `attn_output.weight`
- `attn_q_norm.weight`
- `attn_k_norm.weight`

Bu nedenle full-attention katmanlari fused `attn_qkv` degil, ayri Q/K/V yolunda kaliyor.

## Komutlar ve Sonuclar

### 1. Build

Komut:

```bash
zig build
```

Gozlenen sonuc:

- Basarili cikis
- Derleme hatasi yok

### 2. Inference testleri

Komut:

```bash
zig test src/inference_tests.zig
```

Gozlenen sonuc:

```text
All 39 tests passed.
```

### 3. GGUF testleri

Komut:

```bash
zig test src/gguf_tests.zig
```

Gozlenen sonuc:

```text
All 8 tests passed.
```

### 4. Tum test suiti

Komut:

```bash
zig build test --summary all
```

Gozlenen sonuc:

```text
Build Summary: 11/11 steps succeeded; 94/94 tests passed
```

Not:

- Cikti icindeki `failed command: ./.zig-cache/.../test --listen=-` satirlari gercek assertion failure degil.
- Bunlar Zig'in dahili test runner alt komutlari.
- Esas sonuc satiri yukaridaki `94/94 tests passed` ozetidir.

### 5. Gercek model repro

Komut:

```bash
./zig-out/bin/echo-core-zig "../../../Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" --prompt "hi" --max-tokens 1
```

Gozlenen sonuc:

```text
Output: hi
```

Ek gozlem:

- Weight load tamamlanir
- Eski `matvec.zig -> ssm.zig` segfault zinciri gorulmez

## Kod Seviyesi Kanit Noktalari

- `echo-core-zig/src/core/config.zig`
  `LayerType.qwen_linear` eklendi.
- `echo-core-zig/src/core/memory.zig`
  Packed non-attention layer index hesabina `qwen_linear` dahil edildi.
- `echo-core-zig/src/gguf/reader.zig`
  `qwen35.ssm.group_count` ve `general.alignment` kullaniliyor.
- `echo-core-zig/src/kernels/matvec.zig`
  `.f32` agirliklar icin ayri `matvecF32Fp32` yolu var.
- `echo-core-zig/src/kernels/qwen_linear.zig`
  Qwen linear stateful runtime operatoru mevcut.
- `echo-core-zig/src/inference/engine.zig`
  `qwen_linear` branch aktif ve embedding row decode dtype-aware.
- `echo-core-zig/src/ports/inference.zig`
  Qwen linear tensor classification ve exact loader mapping mevcut.

## Raw Quantized Kaniti

Bu bolumde "hangi agirliklar ham quantized isleniyor" sorusuna kod seviyesi kanit verilir.

### Ham byte yukleme

Dosya:

- `echo-core-zig/src/ports/inference.zig:528-564`

Kanit:

- `loadTensorIfPresent()` tensor dtype bilgisini slot'a yazar.
- Ardindan yorum satirinda acikca `Load raw tensor data directly - NO DEQUANTIZE` yazar.
- `reader.loadTensorInto(...)` ile ham byte kopyalanir.

### Runtime raw quant matvec

Dosya:

- `echo-core-zig/src/kernels/matvec.zig:71-101`

Kanit:

- `.q8_0 => matvecQ80`
- `.q4_k => matvecQ4K`
- `.q2_k => matvecQ2K`

Bu yollar raw quantized bloklari dogrudan dot-product icinde kullanir.

### Qwen linear katmaninda raw quantized kalan tensorler

Dosya:

- `echo-core-zig/src/kernels/qwen_linear.zig:141-180,261-269`
- `echo-core-zig/src/inference/engine.zig:719-743`

Kanit:

- `attn_qkv.weight` -> `weights.qkv` -> `matvecDispatchQuant(...)`
- `attn_gate.weight` -> `weights.z` -> `matvecDispatchQuant(...)`
- `ssm_alpha.weight` -> `weights.alpha` -> `matvecDispatchQuant(...)`
- `ssm_beta.weight` -> `weights.beta` -> `matvecDispatchQuant(...)`
- `ssm_out.weight` -> `weights.out` -> `matvecDispatchQuant(...)`

### Bilerek quantized kalmayan yollar

Bu kisim varsayim degil, koddan dogrudan gorulen davranistir:

- `token_embd.weight` runtime'da satir bazli `f32`'ye acilir:
  `echo-core-zig/src/inference/engine.zig:21-54,365-369`
- `ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight`, `ssm_norm.weight` scalar/f32-f16 yolundan okunur:
  `echo-core-zig/src/kernels/qwen_linear.zig:76-84,193-199,216-217,255`

Yani "hicbir yerde conversion yok" iddiasi bu repo icin yanlistir.
Dogru iddia sunudur:

- buyuk Qwen linear matmul agirliklari ham quantized isleniyor
- embedding ve scalar/norm/conv agirliklarinda kontrollu f32/f16 okuma var

## Leak Kaniti

Bu bolumde target repro komutunda agirlik havuzunun surec sonunda geri verildigine dair OS seviyesi kanit kaydedilir.

### Komut

```bash
strace -f -s 0 -e trace=mmap,munmap,brk,mremap -o "/tmp/qwen35_strace.log" ./zig-out/bin/echo-core-zig "../../../Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" --prompt "hi" --max-tokens 1
```

### Gozlenen buyuk allocation ve release

```text
mmap(0x..., 5150711808, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x...
munmap(0x..., 5150711808) = 0
```

Bu boyut `Engine.init()` logundaki `raw_pool_size=5150711808 bytes` ile ayni buyukluk sinifindadir.

### Kod seviyesi sahiplik zinciri

- `main.zig:152-176`
  `InferencePort.init(...)` sonrasi `defer port.deinit(...)`
- `ports/inference.zig:191-211`
  `InferencePort.deinit()` -> `loader.deinit()` + `tokenizer_.deinit()`
- `inference/engine.zig:206-295`
  `Engine.deinit()` -> `weight_pool`, temp buffer'lar, `kv_cache`, `ssm_states`, `qwen_states` free edilir

### Sinir

Bu kanit sunu ispatlar:

- hedef repro komutunda buyuk weight pool surec sonunda `munmap` ile geri veriliyor

Bu kanit su iddiayi tek basina ispatlamaz:

- tum kod yollarinda hic retention yok
- uzun yasayan REPL oturumlarinda hic bellek tutma davranisi yok

## Bilerek Sinirli Olan Iddialar

Asagidaki iddialar bu belgede KANITLANMIS DEGILDIR:

- Tum Qwen3.5 quant/export varyantlari destekleniyor
- Hugging Face referans implementasyonuyla sayisal parity garanti
- Uzun generation kalitesi veya benchmark sonuclari
- Q5_K / Q6_K / IQ varyantlarinda ayni runtime dogrulugu

Bu belge sadece yerelde test edilen hedef GGUF ve hedef crash zinciri icin kanit sunar.
