# TG-MAX — Q8 throughput denetimi ve ilerleme kaydı

Hedef sırası (yalnız **Q8** modelinde geçerli; Q5 ile baraj geçilmiş sayılmaz):

- TG: 70 → 100 → 150 → 200 → 300 → 500 → 750 → 1000
- Prefill: 750 → 1000 → 1500 → 2000 → 5000 → 7500 → 10000 (TG'den sonra)

Sayaç modeli: `Qwen3.5-0.8B-Q8_0.gguf`. Ölçüm: `--llama-bench` (rastgele token, her tokende logit).

## Durum

| Yol | tg32 / tg128 | Not |
|---|---|---|
| Başlangıç (görsel) CPU t=6 | 39,09 / 36,77 | yük düşükken |
| Şimdiki CPU t=6 | aşağıdaki kayıtta | RAM duvarı |

## Temel fizik (ölçüldü, 2026-11-05)

| Ölçüm | Değer |
|---|---|
| Token başına okunan ağırlık (Q8_0, 24 katman + LM head) | 802 MB |
| Saf okuma, t=1 / t=6 | 21 GB/s / 34 GB/s (`bun` 3–4,5 çekirdek yerken) |
| DDR4-3000 çift kanal teorik | 48 GB/s |
| RX 460 GDDR5 (1750 MHz, 128 bit) teorik | 112 GB/s |
| CPU tavanı, tam yol | 802 MB ÷ 34 GB/s = 42 tok/s |
| CPU tavanı, LM head sıfır maliyetli | 532 MB ÷ 34 GB/s = 64 tok/s |
| GPU tavanı, tam yol | 802 MB ÷ 112 GB/s = 139 tok/s |

Sonuç: CPU'da tam yolla 70 fiziksel olarak imkânsız. 70'in üstü ancak GPU yolu ya da okunan baytı azaltan bir teknikle gelir.

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

### F. GPU yolu

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

Sıradaki tek adım: GPU bant genişliği POC'u (Vulkan, Zig).
