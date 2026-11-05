# TG-MAX — Q8 throughput denetimi ve ilerleme kaydı

Hedef sırası (yalnız **Q8** modelinde geçerli; Q5 ile baraj geçilmiş sayılmaz):

- TG: 70 → 100 → 150 → 200 → 300 → 500 → 750 → 1000
- Prefill: 750 → 1000 → 1500 → 2000 → 5000 → 7500 → 10000 (TG'den sonra)

Kapsam: **yalnız CPU-only inference.** GPU/Vulkan yolu kapsam dışıdır; onunla alınan sayı baraj geçmek sayılmaz.

Sayaç modeli: `Qwen3.5-0.8B-Q8_0.gguf`. Ölçüm: `--llama-bench` (rastgele token, her tokende logit).

## Durum

| Yol | tg32 / tg128 | Not |
|---|---|---|
| Başlangıç (görsel) CPU t=6 | 39,09 / 36,77 | yük düşükken |
| Şimdiki CPU t=6, tam LM head | 29,61 / 36,85 | RAM duvarı |
| **Şimdiki CPU t=6, IVF LM head (yaklaşık, top-1 %96,2)** | **46,47 / 54,48** | **70 geçilmedi** |

## Temel fizik (ölçüldü, 2026-11-05)

| Ölçüm | Değer |
|---|---|
| Token başına okunan ağırlık (Q8_0, 24 katman + LM head) | 802 MB |
| Saf okuma, t=1 / t=6 | 21 GB/s / 34 GB/s (`bun` 3–4,5 çekirdek yerken) |
| DDR4-3000 çift kanal teorik | 48 GB/s |
| CPU tavanı, tam yol | 802 MB ÷ 34 GB/s = 42 tok/s |
| CPU tavanı, LM head sıfır maliyetli | 532 MB ÷ 34 GB/s = 64 tok/s |

Sonuç: CPU'da tam yolla 70 fiziksel olarak imkânsız. 70'in üstü ancak token başına okunan baytı azaltan bir teknikle gelir.

## Token başına süre dökümü (Q8_0, tg32, `fp16` düzeltmesi sonrası)

| Parça | Pay | t=1 ms | t=6 ms |
|---|---:|---:|---:|
| LM head | %32 | 19,4 | 8,9 |
| ffn_gate + ffn_up + ffn_down | %34 | ~20,8 | ~10 |
| attn_qkv | %14 | 8,8 | 4,2 |
| wgate + ssm_out | %10 | ~6 | ~3,5 |
| wq + wk + wv + wo | %6 | ~3,4 | ~2,2 |
| GDN + conv | %3–4 | ~2,8 | ~1,4 |
| rms + küçük op + barrier | %1,5 | ~0,2 | ~0,6 |

## 100 madde

Durum: `BİTTİ` uygulandı ve ölçüldü · `ÇÜRÜDÜ` denendi, kazanç yok · `AÇIK` sırada · `SİZDE` donanım/BIOS, kod dışı · `DIŞI` barajı geçerli sayılmaz.

### A. Bant genişliği duvarı

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 1 | t=6 Q8 TG RAM duvarında: t=3 → 34, t=4 → 35,7, t=6 → 33,5 | `-t` taraması | kök neden | KANIT |
| 2 | Her token tüm ağırlıkları yeniden okuyor (802 MB) | tensör dökümü | kök neden | KANIT |
| 3 | Prefill 7 kat hızlı çünkü batch ağırlık okumasını paylaşıyor; TG paylaşamıyor | 285 / 39 oranı | yön | KANIT |
| 4 | RAM 3000 MT/s, tek rank, 2×8 GiB; 3200/3600 ve FCLK eşlemesi BIOS'tan | `dmidecode` | %6–20 | SİZDE |
| 5 | Tek rank DIMM: bank grubu geçişi az, bant genişliği çift rank'tan düşük | `dmidecode` | %5–10 | SİZDE |
| 6 | Arka plan yükü (`bun` %280–450) RAM duvarını paylaşıyor; ölçümü %15'e kadar oynatıyor | `ps`, tekrar ölçümler | gürültü | SİZDE |
| 7 | Takas dolu (15,1/15,9 GiB) ve bellek parçalı: huge page alınamıyor | `free`, `HugePages` 11 | engel | SİZDE |
| 8 | Tctl 92–95 °C: Zen 2 Tjmax sınırı, boost düşüyor | `sensors` | %5–10 | SİZDE |
| 9 | TG için 6 thread gereksiz: t=4 aynı hızı veriyor, daha az ısı | `-t` taraması | ısı | AÇIK |
| 10 | Dinamik chunk 64 satır: thread'ler sık zıplıyor, prefetcher fazla okuyor | `mulMatChunks` | %0–10 | AÇIK |
| 11 | `L3` 32 MiB iki CCX'e bölük; thread'ler CCX'e göre yerleşmiyor | `lscpu` | %0–5 | AÇIK |
| 12 | Yazma trafiği: logits 1 MB, GDN durumu 18 MB/token | hesap | %2 | AÇIK |
| 13 | Yazılım prefetch (t=1 +%2, t=6 sıfır) | A/B ölçüm | yok | ÇÜRÜDÜ |
| 14 | Kayıpsız ağırlık sıkıştırma: Q8_0 blok ölçeği maks/127, entropi ~7,5 bit | hesap, ölçülecek | <%8 | AÇIK |
| 15 | Q8'i açılışta Q4'e çevirmek | kural | yasak | DIŞI |
| 16 | Daha düşük kuantalı dosya kullanmak | kural | yasak | DIŞI |
| 17 | Rastgele değil gerçek metinde ölçüm yok; spekülasyon görünmüyor | bench kodu | ölçüm | AÇIK |

### B. Kernel kod üretimi

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 18 | `fp16`: `vpinsrw` akümülatöre bağımlı, 20 döngü/blok | `perf annotate` | t=1 +%91 | BİTTİ |
| 19 | Q5_K/Q6_K dot'larında da `fp16` var; aynı düzeltme geçerli | kod | ölçülecek | BİTTİ |
| 20 | Her blokta iki `vcvtph2ps` + `vmulss` + `vbroadcastss`; girdi tarafı sabit, önceden açılabilir | C mikrobench 6,2 → 4,3 | %30 hesap | AÇIK |
| 21 | Girdi `vpsignb` her satırda yeniden; `|x|` önceden hesaplanabilir | mikrobench V2 | yukarıda | AÇIK |
| 22 | 4 satır birlikte kernel (4 akış): RAM'de 15,5 döngü/blok, 10,4'ten kötü | mikrobench V1 | negatif | ÇÜRÜDÜ |
| 23 | 34 bayt adım: 32 baytlık yüklerin ~%47'si önbellek satırını kesiyor | hesap, mikrobench V2 | %15 hesap | AÇIK |
| 24 | `d` ve `q` ayrık, hizalı yerleşim (V2): DRAM'de 8,8 döngü/blok | mikrobench | %15 t=1 | AÇIK |
| 25 | `mulMatOneChunk` `tmp` + `memcpy`: n=1 için gereksiz | kod | <%1 | AÇIK |
| 26 | `hsum` satır başına; satır 32 blok, önemsiz | kod | <%1 | AÇIK |
| 27 | `ssmConv` skaler döngü, conv 0,8 ms t=1 | sayaç | %1 | AÇIK |
| 28 | `gatedDeltaNet` üç geçiş; satır başına tek geçişe indirgenebilir, sonuç aynı | kod | %2 | AÇIK |
| 29 | `softplusF`/`sigmoidF` skaler libm `expf`/`logf` | kod | <%0,5 | AÇIK |
| 30 | `gateAttn` skaler `sigmoidF`; `vExpf` varken kullanılmıyor | kod | <%0,5 | AÇIK |
| 31 | `rope` `cosf`/`sinf` çağırıyor; TG'de token başı önbelleklenebilir | kod | <%0,2 | AÇIK |
| 32 | `flashOneChunk`'ta `vkq16` F16 aritmetiği: her adımda çevrim | kod | uzun bağlam | AÇIK |
| 33 | `dotF16`/`madF16` tek-başlık-tek-hücre; TG'de 64 hücreli tile yok | kod | uzun bağlam | AÇIK |
| 34 | `visible` hücre başına dal; TG'de sınır tek karşılaştırma | kod | <%0,2 | AÇIK |
| 35 | `mulMat` `@memcpy` ile küçük kopyalar çağrı üretiyor | profil `memcpy` %0,14 | <%0,2 | AÇIK |
| 36 | `dequantizeQ80` token başına tek satır, sorun değil | kod | yok | ÇÜRÜDÜ |
| 37 | `rmsNorm` çift duyarlık toplam: ggml eşitliği için, tam yolda kalmalı | kod | yok | ÇÜRÜDÜ |
| 38 | AVX-512 yok; bu CPU'da uygulanamaz | `lscpu` | yok | ÇÜRÜDÜ |
| 39 | Zen 2'de `vpmaddubsw` tek port, ymm 1/döngü; alt sınır ~2 döngü/blok | port tablosu | tavan | KANIT |
| 40 | PGO/`-mcpu=znver2` zamanlaması denenmedi | build | %0–3 | AÇIK |

### C. Sayfa ve yerleşim

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 41 | File-backed mmap: 4 KiB sayfa, token başına ~200 K dTLB miss | `perf stat` 21 M / 97 tok | ölçülecek | AÇIK |
| 42 | Önceki "THP %1–3 kötü" çürütmesi geçersiz: istek karşılanmıyor (814 MB'ın 28–43 MB'ı) | `smaps` | yöntem hatası | KANIT |
| 43 | `compact_memory` sonrası da huge page yok | `bw` testi | engel | KANIT |
| 44 | `nr_hugepages=150` isteği 11 sayfa verdi | `/proc/meminfo` | engel | KANIT |
| 45 | `drop_caches` + compaction + reserve denenmedi | sırada | ölçülecek | AÇIK |
| 46 | Saf okumada file-mmap = anon 4K (21,9 / 20,1 GB/s): sayfa tipi t=1'de belirleyici değil | `bw` | zayıf | KANIT |
| 47 | Açılışta `POPULATE` var, ısınma ilk token'ı bozmuyor | kod | yok | ÇÜRÜDÜ |
| 48 | Ağırlıkları kullanım sırasına dizmek: katman sırası zaten dosya sırası değil | hesap | prefetch | AÇIK |
| 49 | Tampon hizası 64; 2 MiB hizalı arena yok | kod | küçük | AÇIK |
| 50 | Bağlam tamponları anon; `b_*` THP alabiliyor | kod | yok | ÇÜRÜDÜ |

### D. Thread ve barrier

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 51 | Token başına ~770 barrier | sayım | %3–5 | AÇIK |
| 52 | t=6'da CPU döngülerinin %15,8'i `graph` içindeki spin'de | `perf` | teşhis | KANIT |
| 53 | `mulMat` n=1'de girdi kuantalamayı 6'ya bölüp barrier koyuyor; 1024 eleman için zararlı | kod | %1–2 | AÇIK |
| 54 | Aynı `b_norm` için 4–6 kez yeniden kuantalama (qkv, alpha, beta, wgate; ffn_gate, ffn_up) | kod | %1–2 | AÇIK |
| 55 | Tek satırlık op'lar (`add`, `mul`, `silu`) ayrı barrier; thread 0'da birleşebilir | kod | %2 | AÇIK |
| 56 | n_t=1'de `rmsNorm`+`mul` tek geçiş zaten; `add`+`rmsNorm` birleşmiyor | kod | <%1 | AÇIK |
| 57 | Barrier `seq_cst` fetchAdd + çıkış fence: CCX'ler arası gidip geliyor | kod | %1–2 | AÇIK |
| 58 | Spin `pause`; `mwaitx` denenmedi | `lscpu` mwaitx | ısı | AÇIK |
| 59 | Thread'ler CPU'ya bağlı değil; eski "pinning" notu bu yükte `pinned 208–266` | not | ölçülecek | AÇIK |
| 60 | t=12 SMT: 16 tok/s; bağlı/yarışan sıçrama | `-t` taraması | negatif | ÇÜRÜDÜ |
| 61 | Yardımcı thread + prefetch: B deneylerinde aynı thread'e göre kötü | Rapor 5 | negatif | ÇÜRÜDÜ |
| 62 | `gatedDeltaNet`/`flashAttn` chunk'ları ayrı barrier açıyor | kod | <%0,5 | AÇIK |

### E. LM head

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 63 | LM head tüm 248 320 satırı okuyor: 270 MB, %33 | sayaç | t=6 8,9 ms | KANIT |
| 64 | Örnekleme için tüm logit gerekmiyor: argmax/top-K yeterli | `port.zig` greedy | LM head → ~%5 | AÇIK |
| 65 | PCA/projeksiyon eskizi + aday yeniden hesap (MIPS) | sırada | +%50 TG | AÇIK |
| 66 | Küme tabanlı (IVF) aday seçimi | sırada | +%50 TG | AÇIK |
| 67 | Üst sınır budama (norm sıralı, Cauchy-Schwarz) | sırada | tam argmax | AÇIK |
| 68 | Sözlük kısma: `unc-0.8b` 48 736 token'lı, bu yüzden hızlı; Q8 248 320 | tensör dökümü | model farkı | KANIT |
| 69 | `Q5 ≠ Q8` karşılaştırması: Q5 modeli 548 M param, Q8 773 M | tensör dökümü | yorum hatası | KANIT |
| 70 | `greedy` skaler tarama: 248 K, 0,1 ms | kod | %0,4 | AÇIK |
| 71 | Rastgele token bench'i logit'i kullanmıyor; yine de hesaplanmalı (kural) | `bench.zig` | dürüstlük | KANIT |
| 72 | `output.weight` = `token_embd` paylaşımlı, repack edilmemiş | kod | yok | ÇÜRÜDÜ |

### F. GPU yolu (KAPSAM DIŞI, maddeler geçersiz)

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 73 | Motorda GPU yolu yok; aynı kartta llama.cpp Q8 TG 77, pp 828 | belge | >70 ve >750 | AÇIK |
| 74 | llama.cpp Vulkan'ın 13 ms'sinin ~5,8'i bant dışı (dispatch, barrier) | Q8/Q5 doğrusal uyum | +%60 | AÇIK |
| 75 | Birleşik shader: norm+mul, add+norm, conv+silu | sırada | barrier azalır | AÇIK |
| 76 | Aynı düzey matmul'lar tek barrier (`ffn_gate`, `ffn_up`) | sırada | küçük | AÇIK |
| 77 | Kuantalanmış girdi yerine float girdi: GPU'da gereksiz maliyet | llama.cpp shader | hız | AÇIK |
| 78 | int8 → float `cvt_f32_ubyte`: Polaris'te tam hız | ISA | hız | AÇIK |
| 79 | 34 baytlık bloklar GPU'da hizasız: `q` ve `d` ayrılmalı | tasarım | hız | AÇIK |
| 80 | Satır başına tek dalga (64 şerit = 1024 eleman), `subgroupAdd` | tasarım | önbellek | AÇIK |
| 81 | Komut tamponu önceden kaydedilip tekrar kullanılabilir | tasarım | CPU maliyeti | AÇIK |
| 82 | KV ve GDN durumu VRAM'de kalmalı, PCIe'den geçmemeli | tasarım | zorunlu | AÇIK |
| 83 | Logit geri okuma 1 MB × PCIe 3.0 x8 = 0,15 ms | `lspci` | %1–2 | AÇIK |
| 84 | Argmax GPU'da yapılırsa geri okuma 4 bayt | tasarım | %1–2 | AÇIK |
| 85 | LM head eskizi GPU'da da geçerli: VRAM trafiği %33 düşer | hesap | +%50 | AÇIK |
| 86 | sclk 1224 MHz'e çıkmıyor olabilir, güç profili denetlenmeli | `pp_dpm_sclk` | %15 | AÇIK |
| 87 | VRAM 2 GiB, masaüstü 658 MiB tutuyor; Q8 sığıyor | `mem_info_vram_used` | sınır | KANIT |
| 88 | CPU + GPU aynı anda (hibrit) tek token içinde: senkron gecikmesi kazancı yiyor | hesap | negatif | ÇÜRÜDÜ |
| 89 | fp16 yok: F16 KV yerine F32 KV ya da paket açma | `vulkaninfo` | tasarım | AÇIK |

### G. Ölçüm ve doğrulama

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 90 | Tek tur ölçüm yükle %20 oynuyor; A/B aynı anda dönüşümlü yapılmalı | tekrar ölçümler | yöntem | KANIT |
| 91 | Logit referansı repoda yoktu | `tools/logit_check.py` | güvence | BİTTİ |
| 92 | Q5'te logit thread sayısına bağlı (t=3 ≠ t=6); Q8, UD değil | `logit_check -t 3` | hata | AÇIK |
| 93 | Bench `n_ctx = n_gen`: 128 tokende en fazla 128 hücre, uzun bağlam TG ölçülmüyor | kod | ölçüm | AÇIK |
| 94 | `doc_parity` tek CPU; GPU ayrı sütun gerek | araç | ölçüm | AÇIK |
| 95 | Operasyon başına süre sayacı repoda yok | geçici yama | teşhis | AÇIK |

### H. Prefill (TG'den sonra)

| # | Bulgu | Kanıt / test | Etki | Durum |
|---|---|---|---:|---|
| 96 | Q8 pp 285; GPU'da llama.cpp 828 | belge | >750 | AÇIK |
| 97 | `TinyBlasQ0` thread'leri statik bölüyor | kod | %5 | AÇIK |
| 98 | Tile flash `n_t<64`'te devreye girmiyor | kod | küçük batch | AÇIK |
| 99 | n_ubatch 512 sabit; 2048 ile barrier ve kuantalama payı azalır | kod | %3–5 | AÇIK |
| 100 | Batch kuantalama `mulMat` başına tekrar (bkz. 54) | kod | %3 | AÇIK |

## Kayıt

### 2026-11-05: `fp16` bağımlılığı

Kanıtlanan:
- `perf annotate`: `vpinsrw $0,mem,%xmm0,%xmm2` ve `vfmadd231ps ...,%ymm0`. `xmm0` akümülatör; zincir 20 döngü/blok.
- Düzeltme sonrası t=1 Q8_0 tg32 8,6 → 16,4 tok/s, LM head 37,9 → 19,4 ms. `tools/logit_check.py`: 9/9 özet bit bit aynı.

Çürütülen:
- t=6'da kazanç yok: 33,7 → 33,7. Sınır artık RAM.
- Yazılım prefetch: t=1 +%2, t=6 sıfır.
- 4 satırlı kernel (V1): RAM'de ters etki.

### 2026-11-05: Kapsam düzeltmesi

Vulkan yolunu kendi kararımla ekledim (99c73e9, f268571, 265a31e); hedef CPU-only. Üçü geri alındı. O kayıtlardaki "70 geçildi" ve 83,8–98,7 tok/s baraj sayılmaz.

O çalışmadan CPU için geçerli kalan bulgular:
- Q8 çıktısı 1 ULP'ye duyarlı: norm ölçeğine 1 ULP oynama, CPU'da kendine karşı tok0 logit farkı 0,23–0,46 ve 739/1400 greedy eşleşme veriyor. Kaynak 8 bit girdi nicemlemesinin eşik atlamaları. Yaklaşık yolların doğruluğu bu gürültü tabanına göre okunmalı.
- Madde 67 ÇÜRÜDÜ: en iyi logit ~19, `|w|·|x|` ~72; satır ve blok Cauchy-Schwarz sınırı hiçbir satırı elemiyor.
- Ağırlık entropisi 7,67 bit; kayıpsız sıkıştırma yok. `d` tablosu tensör başına 834'e kadar farklı değer.
- Kimlik öneki yetersiz: üretilen tokenlerin %99'u için sözlüğün %80'i gerekiyor. 8000 tokende yalnız 2115 farklı token var.
- n-gram tahmin isabeti %25–28 (20 metin, 8000 token). Doğrulanmış spekülatif batch ağırlık okumasını paylaşır, çıktıyı değiştirmez; CPU için aday.

### 2026-11-05: IVF LM head (CPU)

Yaklaşık LM head: satırlar küresel k-means ile 4096 kümeye ayrıldı (`src/llama/ivf.zig`, önbellek `<model>.ivf4096`, 5,4 MB). Gizli vektör Q8 merkezlerle karşılaştırılır, en yakın 256 kümenin satırları okunur. `--lm-ivf 4096,256`; varsayılan yol tam LM head olarak kaldı.

Kanıtlanan (Q8_0, t=6, 5 tekrar, aynı oturum):

| LM head | tg32 | tg128 |
|---|---:|---:|
| Tam | 29,61 | 36,85 |
| **IVF 4096/256 (yaklaşık)** | **46,47** | **54,48** |

- Üretim arama kodu, 20 gerçek metnin 8000 greedy tokeninde top-1 %96,21; satırların %3,36'sı okunuyor. nprobe 128: %93,43, satır %1,66. nprobe 512: %97,75, satır %7,08.
- nprobe 64–512 arası tg128 52,8–53,6: artık sınır LM head değil, gövde.
- **Sınır:** IVF logit dizisi üretmez, yalnız top-1 verir; kaçan %3,8 tokende çıktı tam yoldan ayrılır. Top-k/sıcaklık örneklemesi henüz yok.

Thread: t=3 50,6, t=5 49,8, t=6 51,5–54,5 (t=4'te 37,8 tek sapma, arka plan yükü). 70 geçilmedi.

`tools/logit_check.py` Zig'e taşındı: `--logit-check [--save]`, referans `benchmarks/logit-ref.txt`. `tools/doc_parity.py` kaldırıldı.

### 2026-11-05: Gövde duvarı (CPU)

IVF sonrası LM head ~0,5 ms; token süresi 18,35 ms, gövde 532 MB ÷ 34 GB/s = 15,6 ms. LM head bedava olsa bile tavan ~64 tok/s. **70 için token başına gövde okuması azalmalı ya da bant genişliği artmalı.**

Çürütülen (hepsi ölçüldü, kod geri alındı):

| Deney | Sonuç |
|---|---|
| SwiGLU kapısıyla `ffn_up` satır atlama | τ=%5: top-1 %71; τ=%10: %46; τ=%20: %17, hız 42,8 → 49,0. Hız için doğruluk yıkılıyor |
| Thread'leri fiziksel çekirdeğe bağlama | 53,4'e karşı 53,0, gürültü içinde |
| `mulMat` chunk 8–512 satır | 32–256 düz; 8 ve 16 biraz kötü |
| Aynı girdiye `mulMat` grubu (tek nicemleme, az bariyer) | logit 9/9 aynı, A/B ≤%1–2, gürültü içinde |
| Doğrulanmış spekülatif batch | Batch süresi n=1 18,35, n=2 22,8, n=3 25,95, n=4 23,25 ms. n-gram k=4'te 1,33 token/batch, 59 tok/s tahmini; token başına 19 MB GDN durum kopyası eklenince ~51 tok/s, kazanç yok |
| `MADV_COLLAPSE` ile huge page | 420 bloktan 4'ü; bellek parçalı, swap 11,7 GB dolu |

Kod dışı kalan yollar (donanım/açılış): açılışta `hugepages=420`, RAM zamanlaması/frekansı.

### 2026-11-05: Prefill, dinamik tile

`TinyBlasQ0.gemm4xN` statik tile yerine 4 tile'lık dinamik satır bloklarıyla çalışıyor. Logit 9/9 aynı. A/B (6 tur, 3 tekrar, Q8_0, t=6): pp128 263 → 269, pp512 264 → 275; en kötü tur 214 → 254. 750 barajı geçilmedi.

Çürütülen: satır bloğu büyütme (rb=16 ~245, rb=64 ~200, rb=256 ~105).
