# HANDOFF — echo-core CPU inference / llama.cpp parite çalışması

> Bu dosya sohbet kaydından ayıklanarak yazılmıştır: kullanıcının verdiği hedefler,
> başlangıç tabloları, oturum boyunca üretilen ölçümler, çürütülen hipotezler ve
> yapılan bütün değişiklikler istisnasız listelenmiştir. Sohbet export aracı mevcut
> olmadığı için kaynak; konuşma geçmişi + `git diff` kanıtıdır.
>
> Tarih: 2026-11-03 · Dizin: `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/`
> Baz commit: `46ffd47 perf: Q8_K integer dot-product path for Q4_K matvec + thread pool hardening`
> Makine: AMD Ryzen 5 3600 (6 çekirdek / 12 thread), AVX2+FMA+F16C, governor=performance

---

## 1. Kullanıcının verdiği görev (aynen)

- Çalışma dizini: `echo-core/`. CPU-only inference; GPU kapsam dışı.
- Amaç: Zig inference motorunu (`echo-core-zig`) llama.cpp CPU baseline'ına **1:1 eşitlemek**.
- Tolerans: llama.cpp'den **%1 bile kötü sonuç kabul edilmez**; eşit veya üstü kabul.
- Bu başarıya ulaşmadan iş bitmiş sayılmaz (kullanıcı ifadesi).

### llama.cpp baseline (kullanıcının verdiği, CPU)

| Model | Boyut / Kuant | pp128 | pp512 | tg32 | tg128 |
|---|---|---|---|---|---|
| unc-0.8b-q5km.gguf | 380.88 MiB, Q5_K_M | ~153.3 | ~136.4 | ~42.3 | ~47.3 |
| Qwen3.5-0.8B-Q8_0.gguf | 784.52 MiB, Q8_0 | ~155.7 | ~171.1 | ~31.8 | ~30.7 |
| Qwen3.5-0.8B-UD-Q8_K_XL.gguf | 1.11 GiB, UD-Q8_K_XL | ~124.7 | ~128.8 | ~23.1 | ~20.1 |

### Başlangıçta Zig motorunun durumu (kullanıcının verdiği)

| Model | Durum | Prefill | Generation | Not |
|---|---|---|---|---|
| Qwen3.5-0.8B-UD-Q8_K_XL.gguf | ÇALIŞTI | ~1.26 t/s | ~1.28 t/s | Uçtan uca token üretti (18 prompt, 10 decode) |
| unc-0.8b-q5km.gguf | PANIC | - | - | `loadEmbeddingRowToF32`: embedding Q6_K, motorda karşılığı yok (unreachable) |
| Qwen3.5-0.8B-Q8_0.gguf | SEGFAULT | - | - | `0x7bbafaf20000` adresinde crash (SSM layer slot/ofset şüphesi) |

Kullanıcının başlangıç değerlendirmesi: forward adımları SIMD/AVX2 çekirdeklere ve
multithread pool'a tam bağlı değil, tek thread scalar/dequantize döngüsünde ~1.2-1.3 t/s.

## 2. Sonuç hükmü (başarı ölçütüne göre)

**HEDEF TUTMADI.** Hiçbir modelde %1 toleransla llama paritesi yok.

- Generation: Q8_0 hedefe en yakın (~28.5 ölçülene karşı 30.7). UD dalgalı olarak
  hedef civarında (koşuya göre 16.9-22.5; hedef 20.1). unc belirgin geride (~27'ye karşı 42.3).
- Prefill: üç model de hedefin yaklaşık üçte birinde (uzun promptta 45-52'ye karşı 125-171).
- Üretilen metin anlamsal olarak **yanlış** (bkz. Bölüm 7). Hız sayıları, yanlış üreten
  bir motora aittir; bu gizlenmemiştir.

## 3. Kullanıcı emriyle silinenler ("zero test")

Kullanıcı açıkça emretti: aktif bütün testler ve driven-development skill'lerinin tamamı silinecek.

Silinen Zig test hedefleri/dosyaları (commit'e dahil):

- `echo-core-zig/src/gguf_tests.zig`, `inference_tests.zig`, `kernel_tests.zig`,
  `kv_cache_tests.zig`, `tokenizer_tests.zig` (dosya olarak silindi)
- Kaynak içi bütün `test "..."` blokları: `src/core/{config,math,memory,types}.zig`,
  `src/data/{arraylist,hashmap,string}.zig`, `src/gguf/reader.zig`,
  `src/inference/engine.zig`, `src/kernels/{avx2_norm,avx2_q4k,avx2_q5k,avx2_q6k,avx2_softmax,int_dot,matvec,quant,qwen_linear,simd,ssm}.zig`,
  `src/kv_cache/cache.zig`, `src/main.zig`, `src/ports/inference.zig`, `src/tokenizer/tokenizer.zig`
- `echo-core-zig/build.zig` içindeki Zig test hedefleri
- Kök `CMakeLists.txt` içindeki `inference_test` / `quant_test` hedefleri

Silinen skill'ler:

- `~/.config/opencode/skills/test-driven-development` (symlink) + `~/.codex/aegis/skills/test-driven-development` (gövde)
- `~/.config/opencode/skills/subagent-driven-development` (symlink) + `~/.codex/aegis/skills/subagent-driven-development` (gövde)

Sonuç: repoda **hiç test yok**. Doğrulama yalnızca build + gerçek model bench +
geçici harness'larla yapıldı (geçici harness'lar kullanıldıktan sonra silindi, bkz. Bölüm 5).

## 4. Yapılan düzeltmeler (kök nedenleriyle)

### 4.1 Q8_0 SEGFAULT — giderildi
- GGUF taraması (Python ile tensör envanteri): `qwen35.block_count=25` ama
  `qwen35.nextn_predict_layers=1`. `blk.24` ana katman değil, MTP (next-token-prediction) bloğu;
  içinde `nextn.*` tensörleri var ve deseni ana bloklardan farklı.
- Ana katman sayısı 24, desen `LLLALLLALLLALLLALLLALLLA` (L=qwen_linear, A=attention).
- Düzeltme: `echo-core-zig/src/gguf/reader.zig` `populateConfig` içine
  `num_layers = block_count - nextn_predict_layers` çıkarımı. SSM katman sayısı 19 yerine
  doğru değer olan 18 raporlanıyor (`INFO: ... SSM layers=18`).

### 4.2 unc PANIC — giderildi
- `unc-0.8b-q5km.gguf` token embedding tensörü Q6_K (`token_embd.weight (1024, 48736) q6_k`).
- `echo-core-zig/src/inference/engine.zig` `loadEmbeddingRowToF32` içine `.q6_k`
  dequantize dalı eklendi (210 byte blok: ql[128] + qh[64] + scales[16] i8 + d f16 @208).
  Blok formülü, GGUF Q6_K spesifikasyonuyla ve `avx2_q6k.zig` içindeki mevcut düzenle eşleşir.

### 4.3 Attention'da yanlış format yorumu — giderildi (büyük doğruluk hatası)
- `engine.zig` `attention()` fonksiyonu, dtype'a hiç bakmadan bütün modellerde
  `parallelMatvecQ4KPre` (Q4_K integer yolu, 144 byte blok) çağırıyordu. Q8_0/Q5_K/Q6_K/F16
  ağırlıklar Q4_K gibi yorumlanıyordu → attention çıkışı her modelde bozuktu.
- Düzeltme: q/k/v/o projeksiyonları `parallel.parallelMatvec(..., dtypeForTensor(slot))`
  ile doğru dtype'da hesaplanıyor. Slot eşlemesi (`src/ports/inference.zig`):
  1=attn_q, 2=attn_k, 3=attn_v, 4=attn_output/attn_gate, 6=ffn_gate, 7=ffn_up, 8=ffn_down,
  11=ssm_out, 12=ssm_alpha, 13=ssm_beta, 14=ssm_a, 15=ssm_dt.bias, 16=ssm_conv1d, 17=ssm_norm.
- Not: Q4_K yolu artık attention'da kullanılmıyor; `parallelMatvecQ4KPre` tanımı duruyor.

### 4.4 GQA eşlemesi — giderildi
- `attention()` içinde `kv_head = h % num_kv_heads` idi (8 başlık, 2 KV başlığı için yanlış).
  Doğrusu `h / (num_heads / num_kv_heads)`. Hem `attention()` hem batched prefill kopyasında düzeltildi.

### 4.5 qwen_linear conv_dim — giderildi
- Tensör gerçeği: `ssm_conv1d.weight` `[4, 6144]`, `attn_qkv` 6144 satır. Kod `conv_dim=4096`
  (`key_dim = ssm_inner/2 = 1024`) varsayıyordu; qkv matvec'i ve konvolüsyon eksik işliyordu.
- Düzeltme (`src/kernels/qwen_linear.zig`): `conv_dim`, conv1d tensörünün bayt uzunluğundan
  türetiliyor (`bytes / (conv_kernel * elem_bytes)`); `key_dim = (conv_dim - value_dim)/2 = 2048`,
  `num_k_heads=16`, tekrar faktörü 1.
- Bölge uzunluğundan türetme denendi ve **çürütüldü**: bazı katmanlarda slot bölgesi dtype'a
  göre şişirilmiş olduğundan 6144 yerine 7330 bulunuyordu (unc katman 4, q5_k) ve assert patlıyordu.
  Bu yüzden kaynak conv1d tensörüdür.
- `engine.zig` tarafı: `qwen_tmp_mixed_qkv`/`qwen_tmp_conv_out` tamponları `ssm_inner*2`den
  `*3`e çıkarıldı; `QwenLinearState.init` conv_dim argümanı `key_dim*2 + value_dim` (6144) oldu.

### 4.6 RoPE — DÜZELTİLMEDİ (şüphe olarak duruyor)
- `applyRoPE` ve batched kopyası `applyRoPETo`: bütün 256 boyutu, baz 10000 ile döndürüyor.
- GGUF metadata: `qwen35.rope.dimension_count=64`, `rope.freq_base` okuması şüpheli
  (okuyucu f32 metadata'yı tam sayı gibi basıyordu; örn. epsilon `897988541` bit deseni).
  llama qwen35'te kısmi RoPE (64/256) uygular. Bu düzeltilmedi; anlamsal bozukluğun
  baş şüphelilerinden biri.

## 5. Performans işleri (yapılan + ölçüm kanıtı)

### 5.1 Persistent thread pool — `src/kernels/parallel.zig` tamamen yeniden yazıldı
- Önceki hâli: matvec başına `std.Thread.spawn` (token başına ~193 spawn; perf'te
  `compiler_rt.memset` %20, TLS hazırlığı spawn içindeydi).
- Yeni hâli: generation/futex tabanlı kalıcı pool; ana thread katılımcı; worker'lar önce
  spin (2048), sonra 1 ms futex bekleme. Toplam katılımcı = mantıksal CPU/2 (=bu makinede 6).
- Ölçülen: 12 katılımcı belirgin yavaş (tg 16.33'e karşı 21.7 aynı gün, aynı model).
  Kod `MAX_TOTAL=64`, toplam = `(cpu+1)/2`.

### 5.2 Q8_0 integer-dot matvec — `matvec.zig: matvecQ80Int`
- x, 32'lik blok başına int8'e kuantize ediliyor (`xq`/`xs` tamponları), ağırlıklarla
  integer dot; ölçekli f32 fallback (`matvecQ80`) korundu, dispatch koşulu `K%32==0 ve K<=16384`.
- Sonraden: 4 kısmi toplam (s0..s3) + `@prefetch` (16 blok ileri, locality 2) eklendi.
  Etki: Q8_0 tg 18.91 → 21.55 (aynı oturum ölçümleri).

### 5.3 F16/F32 vektör matvec — `matvec.zig: matvecFp16Vec / matvecF32Vec`
- 4 vektör akümülatörü, 32'lik adımlar, kuyruk skaler. Dispatch `.f16/.f32` bunlara bağlı.
- İçlerine `@prefetch` eklendi (etkisi ölçüm gürültüsü içinde, nötr).
- Etki: UD tg 4.16 → 16.78 (UD'de 394M f16 eleman var; çıktı projeksiyonu da f16 embedding kopyası).

### 5.4 Q5_K / Q6_K kernel yeniden yazımı — `avx2_q5k.zig`, `avx2_q6k.zig`
- Eski hâli: blok başına çok sayıda yatay `@reduce` (Q5: 8/blok, Q6: 16/blok).
- Yeni hâli: 8 elemanlık parçalar hâlinde ağırlık vektörü üretiliyor, reduce satır başına
  bir kez (4 akümülatör, chunk paritesiyle dönüşümlü). Blok başına `@prefetch` var.
- Sayısal doğrulama (geçici harness, kullanıldıktan sonra silindi): rastgele bloklarda
  `quant.dequantizeRow` referansına karşı Q5 rel fark ~4e-7, Q6 ~1.1e-4
  (referansın fp16 yuvarlaması sınırında). Eşleme (segment sırası) bu testle doğrulandı.
- Etki: unc tg 12.25 → 21.71 (aynı gün ardışık ölçüm).

### 5.5 Batched prefill (GEMM) — yeni
Yeni parçalar:

- `src/kernels/matvec.zig`: `matmatDispatchQuant` + `matmatDispatchQuantStrided`
  (token-major Y, stride/ofset destekli). Çekirdekler:
  - `matmatQ80Float`: Q8_0 için f32-dequant GEMM (x kuantize edilmez; decode yolu ile
    arasındaki ~%0.35 fark kuantizasyondandır, doğrulama harness'ında ölçüldü).
  - `matmatFloat(is_f16)`: f16/f32.
  - `int_dot.zig: matmatQ4K_int`: Q8_K'ye ön-kuantize + `dotQ4K`.
  - `avx2_q5k.zig: matmatQ5K_avx2`, `avx2_q6k.zig: matmatQ6K_avx2`:
    ağırlık vektörü parça başına bir kez üretilir, tokenlara dağıtılır.
  - Desteklenmeyen dtype'larda token başına matvec fallback'i (Y önce sıfırlanır;
    matmat semantiği "üzerine yaz", "+=" değil).
- `src/kernels/parallel.zig`: `parallelMatmat` (M satırları pool'a bölünür,
  her worker `matmatDispatchQuantStrided` ile kendi satır dilimini bütün tokenlar için yazar).
- `src/inference/engine.zig`:
  - `prefillBatched`: embedding bütün tokenlara; katman katman batched işlem;
    final norm + çıktı projeksiyonu yalnızca son token için bir kez
    (önceden forwardToken her prompt tokenında 248320 satırlık çıktı projeksiyonu yapıyordu).
  - `layerForwardBatched`: norm satır satır; qwen_linear için qkv/z matmat +
    alpha/beta (dequantize→f16 matmat) + token başına `forwardCore` + out matmat;
    attention için q/k/v matmat + token başına head-norm/RoPE/KV-append/score/softmax +
    o-projeksiyon matmat; SSM (eski tip) katmanlarda token başına eski yola düşer.
  - `ffnBatched`, `applyRoPETo` (pozisyon parametreli RoPE kopyası), `BatchScratch`.
  - `prefill()` artık `prefillBatched` çağırıyor; `generate()` prompt'u `prefill` ile işliyor.
  - T<4 için batched yol sıralı `forwardToken` fallback'ine düşer.
- `src/kernels/qwen_linear.zig`: `forwardCore` eklendi (mevcut `forward` korunarak;
  conv+recurrent kısmının kopyası). Tek token yolu (`forward`) ile batched yol aynı
  recurrent matematiğini kullanır ama kod kopyadır — birleştirilmedi.

- Etki (Q8_0, 18 prompt token): pp 15.7 → ~26-31 bandı. 321 tokenda pp ~34 (önce)
  → recurrent vektörleştirmesiyle ~51-52 (bkz. 5.6).

### 5.6 Recurrent adımın vektörleştirilmesi — `qwen_linear.zig` (forward + forwardCore)
- Eski düzen: `kv_mem` ve `attn` hesapları v-dış/k-iç döngüyle, state'e adımlı (stride'lı) erişim.
- Yeni düzen: k-dış döngü; state satırı `[k*head_v_dim ..]` üzerinde ardışık vektör FMA
  (`kv_mem[v] += srow[v]*kval`, `attn_arr[v] += srow[v]*qval`). State güncelleme döngüsü zaten ardışıktı, korundu.
- Etki: Q8_0 uzun prompt pp 40.7 → 51.2; kısa decode tg da iyileşti (27 bandı).

### 5.7 Matmat token bloğu 4 → 8
- `matmatQ80Float`, `matmatFloat`, `matmatQ5K_avx2`, `matmatQ6K_avx2`: token bloğu 8,
  akümülatör/token=1 vektör (8 bağımsız FMA zinciri). Tam blok/kuyruk ayrımı Q80Float'ta
  dalları (`tt < tn`) sıcak döngüden çıkardı (etkisi nötr ölçüldü).
- Etki: UD pp 39.4 → 48.5 (321 token). Q8/unc'ta nötr-gürültü aralığı.
- Doğrulama: geçici harness ile matmat vs token-başına-matvec karşılaştırması:
  q5/q6/f16 rel ~2e-7; q8_0 rel %0.35 (tasarım farkı: matmat x'i kuantize etmez).
  Harness ve build.zig'deki geçici hedef iş bitince silindi (build.zig eski hâline döndü).

## 6. Çürütülen / işe yaramayan denemeler (tekrar denemeyin)

| Deneme | Sonuç |
|---|---|
| 12 thread (SMT dahil) | tg 16.33'e karşı 6 thread 21.7 — daha yavaş, geri alındı |
| `taskset -c 0-5` afinite | Fark yok (27.3/26.6 vs 27.3/26.0) — migration darboğaz değil |
| Q8 2-satır iç içe (mikro-bench) | 5.92 GB/s'e karşı tek satır 8.90 — register baskısı, daha yavaş |
| Q8 i16/i16-pairwise varyantları (mikro-bench) | A(i32)=0.22ms, B(i16)=0.27, D(pair)=0.34 ms/rep — mevcut i32 yolu en hızlısı |
| Q5/Q6/F16'ya prefetch | Ölçüm gürültüsü içinde, kalıcı kazanç yok (kodda duruyor, zararsız) |
| Matmat'ta `tt<tn` dallarını kaldırma | pp 51.2 → 52.3 — nötr |
| conv_dim'i qkv bölge baytından türetme | Bölge şişirmesi yüzünden 7330/assert — çürütüldü, conv1d kaynağına geçildi |
| Q5/Q6→Q8_0 yükleme-anında repack | Denetlenip **reddedildi**: boyutlandırma iki ayrı sistemde (`core/memory.zig` layout'u katman-0 tensöründen, `ports/inference.zig` katman-katman info) — uyumsuzluk riski yüksek bulundu, uygulanmadı |

Ölçüm altyapısı notları: `perf` kullanılabilir. `/tmp/opencode` yazılamaz durumda;
profil çıktıları `mktemp -d` altına alındı. `perf report` darboğazları:
matvec/matmat dağılımı, `Engine.prefill` içi recurrent, memset.

## 7. Doğruluk durumu (açık yara)

`--prompt 'The capital of France is' --max-tokens 16` çıktısı üç modelde de prompt
yankısı + anlamsız tokenlar. Yani motor artık EOS'a hemen düşmüyor, token üretiyor
ama üretim semantically yanlış. Doğrulanmış şüpheliler:

1. RoPE kısmi boyut/frekans bazı uygulanmıyor (Bölüm 4.6).
2. Attention katmanlarında `attn_q` tensörü 4096 satır (q+gate olası); motor yalnızca
   ilk 2048 satırı q olarak kullanıyor, gate yarısını uygulamıyor (llama qwen35 gated
   attention'da gate sigmoid ile çarpılır — bu kodda yok).
3. GGUF metadata okuyucusunda f32 alanların (epsilon vb.) bit deseni olarak basılması;
   `populateConfig` sayısal ayrıştırması ayrıca denetlenmedi.

llama-cli ile token-token karşılaştırma yapılmadı. `llama-cli` sistemde var;
interaktif beklememesi için `-st` bayrağı şartıyla çalıştırıldı
(`llama-cli -st -m <model> -p '...' -n 16 --temp 0 -t 6`).

Hız hedefi tutsa bile bu madde kapanmadan motor "doğru" sayılmaz.

## 8. Ölçüm geçmişi (oturum boyunca, ReleaseFast, 32 token aksi belirtilmedikçe)

### Generation tg (t/s) — kısa prompt (18-19 token)
| Aşama | Q8_0 | UD | unc |
|---|---|---|---|
| Başlangıç | SEGFAULT | 1.43 | PANIC |
| SEGFAULT+PANIC düzeltmeleri sonrası | 1.67 | ~1.4 | 1.68 |
| Pool + Q8 int-dot sonrası | 21.90 | 4.16 | 12.54 |
| f16/f32 vektör sonrası | 22.39 | 16.78 | 12.25 |
| Q5/Q6 yeniden yazım sonrası | - | - | 21.71-23.48 |
| Doğruluk düzeltmeleri (attn dtype + conv_dim) sonrası* | 18.91 | 16.79 | 18.19 |
| Q8 kısmi toplam + prefetch sonrası | 21.55 | 17.58 | 19.58 |
| Batched prefill + recurrent vektörleştirme sonrası | 26.8-28.5 | 19.3-22.5 | 27.2-30.7 |

* Doğruluk düzeltmeleri hızı düşürdü çünkü motor artık gerçek işi yapıyor
(qkv 4096 yerine 6144 satır; attention'da ucuz-yanlış Q4_K yolu yerine gerçek dtype).

### Prefill pp (t/s)
| Prompt | Q8_0 | UD | unc |
|---|---|---|---|
| Kısa (18-19 tok), sıralı ön hâl | 15.28 | 3.80 | 10.44 |
| Batched prefill ilk hâli | ~25.8 | ~26.4 | ~21.1 |
| 321 token, batched ilk hâli | 34.27 | 36.16 (321) | 39.12 (369) |
| Recurrent vektörleştirme sonrası (321) | 51.16 | - | 48.31 |
| TB=8 sonrası (321) | 51.12 | 48.53 | 44.64 |

Son doğrulama koşusu (kısa prompt, metin çıktısı dahil): üç model de EXIT 0,
`{"prompt_tokens":18,...,"pp_tps":30.98,"tg_tps":28.47}` (Q8_0),
`pp 31.52 / tg 22.49` (UD), `pp 28.23 / tg 27.19` (unc).

**Varyans uyarısı:** Aynı binary ile ardışık koşularda ±%15 oynama var
(makinede arka plan yükü: load average ~2-3.5 gözlendi). Tek koşuluk sayılar
karar için kullanılmamalı; medyan protokolü kurulmadı (kalan iş).

### Hedeflerle son karşılaştırma
| Model | tg hedef | tg ölçülen (bant) | pp hedef (pp128) | pp ölçülen (321 tok) |
|---|---|---|---|---|
| unc | 42.3 | 27-31 | 153 | 45-48 |
| Q8_0 | 30.7 | 27-28.5 | 156 | 51 |
| UD | 20.1 | 17-22.5 | 125 | 48 |

Mikro-bench referansları (tek thread, streaming): Q8 mevcut çekirdek 8.90 GB/s,
prefetch'li 9.55, saf okuma tavanı (SUM) 14.64 GB/s. Decode'da etkin bant:
Q8_0 ~22 GB/s (820 MB/token varsayımıyla), unc ~13.5 GB/s — Q5/Q6 compute-bound.

## 9. Kalan iş (öncelik sırasıyla, dürüst boyutlandırma)

1. **Anlamsal doğruluk** (hızdan önce kapanmalı): RoPE kısmi boyut + frekans bazı;
   attention gate yarısının uygulanması; llama-cli çıktısıyla token-token diff.
   Bunlar değişince hız sayıları da değişir (muhtemelen düşer).
2. **Prefill 3x açık**: Mevcut matmat ~50 GMAC/s üretiyor; llama ~120 GMAC/s.
   Gerçekçi yol: ağırlıkların yükleme anında GEMM-dostu paketlenmiş düzene çevrilmesi
   (repack) + integer (maddubs-tarzı) GEMM çekirdeği. Bu ayrı, baştan planlanacak bir iş;
   "birkaç ince ayar" değil. Repack için Bölüm 6'daki boyutlandırma denetimi yeniden açılmalı.
3. **unc generation**: Q5/Q6 integer-dot (`vec_dot_q*_K_q8_K` tarzı) veya repack.
   Mevcut f32 yolu parça başına ~12 vektör işi/8 eleman ile compute-bound.
4. **Q8/UD generation son %10-30**: sıralı bölümler (recurrent, norm, attention skor
   döngüleri) ve pool senkronizasyonu profillenip sıkılmalı; önce varyansı ezen ölçüm
   protokolü (sabit pp128/pp512 promptu, N koşu medyanı) kurulmalı.
5. **Ölçüm protokolü**: llama baseline pp128/pp512 ile aynı prompt uzunluklarında,
   aynı thread sayısında, medyan değerlerle karşılaştırma yapılmadan parite ilan edilemez.

## 10. Komutlar

Build:
```bash
cd echo-core-zig && zig build -Doptimize=ReleaseFast
```

Bench (JSON):
```bash
./zig-out/bin/echo-core-zig ../ArastirmaModelleri/<model>.gguf \
  --prompt 'Hello world, what is the meaning of life?' --max-tokens 32 --bench --json
```

Uzun prompt (pp ölçümü için, ~321 token):
```bash
P=$(python3 -c "print('The quick brown fox jumps over the lazy dog. '*16)")
./zig-out/bin/echo-core-zig ../ArastirmaModelleri/<model>.gguf --prompt "$P" --max-tokens 32 --bench --json
```

Profil:
```bash
d=$(mktemp -d); perf record -F 999 -g -o $d/perf.data ./zig-out/bin/echo-core-zig <args>
perf report -i $d/perf.data --stdio --no-children --sort=symbol
```

llama karşılaştırması (interaktif beklememesi için `-st` zorunlu):
```bash
llama-cli -st -m ../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf \
  -p 'The capital of France is' -n 16 --temp 0 -t 6 --no-warmup
llama-bench -m ../ArastirmaModelleri/<model>.gguf
```

## 11. Ortam ve sürüm notları

- Zig: `0.17.0-dev.332+4a65cc4ac`. `std.Thread.Condition/Mutex` yok; pool senkronizasyonu
  `std.os.linux.futex_*` + atomic spin ile yazıldı. Zaman için `std.Io.Timestamp`.
- `std.heap.smp_allocator` matmat scratch tahsislerinde kullanıldı (Q4_K yolu).
- Modeller `ArastirmaModelleri/` altında, **git'te izlenmiyor ve commit'e dahil edilmedi**
  (kullanıcı talimatı). İçinde `Qwen3.5-0.8B-Q8_0_tensor_analysis.txt` de untracked.
- GGUF envanteri için oturumda Python (stdlib struct) ile tensör/metadata dökümü kullanıldı;
  `analyze_gguf.py` çıktısındaki bazı alanlar güvenilmez (dtype/shape baskısı kısımları) —
  ham parse tercih edildi.

## 12. Bu commit'in içeriği

- 32 dosya değişti (HANDOFF öncesi diff: +1189/-2270). Değişikliklerin tamamı tek commit'te;
  baz: `46ffd47`.
- Silinen test dosyaları ve CMake/Zig test hedefleri dahildir (Bölüm 3).
- Büyük değişim dosyaları: `src/inference/engine.zig` (batched prefill),
  `src/kernels/matvec.zig` (int-dot + matmat), `src/kernels/parallel.zig` (pool + matmat),
  `src/kernels/avx2_q5k.zig` / `avx2_q6k.zig` (yeniden yazım),
  `src/kernels/qwen_linear.zig` (conv_dim + recurrent + forwardCore),
  `src/gguf/reader.zig` (nextn çıkarımı; ayrıca test bloklarının silinmesi diff'i büyütüyor).
- Repoda test yoktur; CI/bench doğrulaması Bölüm 10'daki komutlarla elle yapılır.
---

## 13. Yeni yön: Sezaryen Modu

- Amaç: autoregressive zinciri kırmak; dışarıda üretilen token/özellikleri modelin içine püskürtmek; motor seviyesinde model-agnostik tarif.
- Belgeler: docs/sezaryen-modu/00-MANIFESTO.md, 01-ARASTIRMA.md, 02-FORMUL-VE-HACK.md, 03-DENEY-VE-MIMARI.md, 04-KAYNAKLAR.md.
- Mindmap (tek dosya): q8-tg-yontem-haritasi.mindmap.html; "Sezaryen Modu — Tam Harita" dalı ana haritanın içinde, 11 alt dal (30 saniyelik özet, duvar, A/B, kanıt, hile, karar ağacı, deney, ilk adım, risk, sözlük, kaynaklar).
- Duvar: MLSys 2026 hızlanma denklemi E = (1-alfa^(k+1))/((1-alfa)(k c+1)).
- Kanıt bandı: PLD 2.4x, Lyanna 3.3x, ANPD 3.67x, MLSys adaptif 4.9x, DFlash 6.1x, CopyLoop 10.7x (özel model).
- İlk somut adım: KV rollback/truncate + tek zincir doğrulama + n-gram drafter; sonra ESP embedding injection, sonra hidden-state reuse.
- Kırmızı çizgi: model dosyası ve quantizasyon değişmez; adaptör motorun içinde; lossless ve task ayrı etiketlenir.
