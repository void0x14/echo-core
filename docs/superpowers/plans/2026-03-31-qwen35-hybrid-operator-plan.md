# Qwen3.5 Hibrit Operator Uygulama Plani

> **Ajanik uygulayicilar icin:** ZORUNLU ALT-SKILL: Bu plani gorev gorev uygulamak icin `superpowers:subagent-driven-development` (onerilen) veya `superpowers:executing-plans` kullanin. Adimlar takip icin checkbox (`- [ ]`) soz dizimi kullanir.

**Amac:** OOM/layout tarafi buyuk olcude kirilmis Qwen3.5-4B GGUF yuklemesinden sonra, runtime'da dogru hibrit mimariyle ilerleyip full-attention katmanlarini mevcut attention path ile, Qwen linear katmanlarini ise ayri ve dogru operator path ile calistirmak.

**Mimari:** Mevcut `kernels/ssm.zig` generic Mamba-benzeri operatorunu Qwen3.5'e zorlamak yerine, Qwen3.5 linear katmanlari icin ayri bir layer/operator contract tanimlanacak. Loader, weight layout ve runtime dispatcher, `full_attention` ile `qwen_linear` katmanlarini acikca ayiracak; dtype handling `q8_0` ve `f32` tensor gercegine gore tanimlanacak.

**Teknoloji Yigini:** Zig 0.16, GGUF reader, custom weight layout, quant matvec kernels, Qwen3.5 GGUF metadata/tensor contracts

---

**Dogrulanmis Kanit**

- Yerel GGUF metadata komutu `python3 "analyze_gguf.py" "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf"` su alanlari verdi: `qwen35.embedding_length=2560`, `qwen35.attention.head_count=16`, `qwen35.attention.head_count_kv=4`, `qwen35.attention.key_length=256`, `qwen35.full_attention_interval=4`, `qwen35.ssm.group_count=16`, `qwen35.ssm.inner_size=4096`, `qwen35.ssm.time_step_rank=32`.
- Ayni komut, yerel tensor setinin `attn_qkv.weight`, `attn_gate.weight`, `post_attention_norm.weight`, `ssm_a`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_dt.bias`, `ssm_conv1d.weight`, `ssm_out.weight` oldugunu; `ssm_x.weight`, `ssm_dt.weight`, `ssm_D.weight` gibi klasik Mamba tensorlerinin bulunmadigini gosterdi.
- `echo-core-zig/src/gguf/reader.zig:266-292` metadata'dan `hidden_dim`, `num_heads`, `num_kv_heads` okuyor ve `attention.key_length` varsa `head_dim` olarak onu kullaniyor. Bu nedenle bu model icin birincil yerel kanit `head_dim=256`dir; `hidden_dim / num_heads = 160` turevleri ikincil ve yanlis olabilir.
- `echo-core-zig/src/ports/inference.zig:20-24` ve `107-109` explicit `head_dim` varsa onu koruyor; `detectActualDimensions()` artik `8192` eksenini `hidden_dim` sanmiyor.
- Yerel repro komutu `./zig-out/bin/echo-core-zig "../Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" --prompt "hi" --max-tokens 1` su durumu verdi: weight load tamamlandi, `raw_pool_size=5150711808 bytes (4912.10 MB)` ve crash sonrasinda stack `matvec.zig:26 -> ssm.zig:167 -> engine.zig:534` olarak dustu.
- `echo-core-zig/src/kernels/ssm.zig:141-160` generic Mamba-benzeri bir contract bekliyor: `ssm_x`, `ssm_dt.weight`, `ssm_A/B/C/D`, `ssm_conv1d_b`.
- `echo-core-zig/src/kernels/ssm.zig:167-245` ve `echo-core-zig/src/kernels/matvec.zig:33-43` SSM yolunda tum projection'lari `matvecDispatch()` ile fp16 dense gibi okuyor.
- `echo-core-zig/src/inference/engine.zig:507-559` `.ssm` layer'i dogrudan bu generic `ssmForward` yoluna bagliyor.
- `echo-core-zig/src/ports/inference.zig:567-604` fused `attn_qkv` split helper'ina sahip, ama `777-794` araligindaki aktif load path halen ayri `attn_q/k/v.weight` bekliyor.
- `echo-core-zig/src/gguf/reader.zig:303-322` SSM metadata okuyor, ancak `ssm.num_groups` anahtarini bekliyor; yerel GGUF metadata ise `qwen35.ssm.group_count=16` veriyor. Bu ayri bir sozlesme acigi.
- Context7 tarafinda resmi Transformers dokumani Qwen3-Next icin hibrit yapinin `Gated DeltaNet + Gated Attention` oldugunu dogruluyor. Ancak Context7 sorgulari `full_attention_interval` ve `layer_types` alanlarinin tam config tanimini net dondurmedi; bu alanlar icin yerel GGUF metadata daha guclu kanit.

**Kisa Root-Cause Ozeti**

- Cozulen problem: OOM/layout ve weight byte sizing zinciri. Kanit: mevcut binary artik modeli yukluyor, `raw_pool_size` yaklasik 4.9 GB seviyesinde ve crash load oncesi degil load sonrasi geliyor.
- Cozulmeyen problem: Qwen3.5 `linear_attention` operator/mathematics mismatch. Kanit: crash, ilk token forward sirasinda `ssm.ssmForward()` icinden geliyor ve fp16 dense matvec, Qwen'in mevcut `q8_0`/`f32` tensorlerini yanlis yorumluyor.
- Teknik sonuc: Bu gorev artik OOM fix gorevi degil. Ana is, Qwen3.5 hibrit katmanlarini dogru operator tasarimina baglamak.

**Karar**

- `kernels/ssm.zig` uzerine kucuk alias yamalari ana cozum olmamali.
- Qwen3.5 linear katmanlari icin ayri bir `qwen_linear` layer/operator path tasarlanmasi ana teknik yon olmali.
- Mevcut attention path, sadece full-attention katmanlari icin korunmali.
- Generic `ssm.zig`, gercekten Mamba-benzeri modeller icin ayrik kalmali.

**Neden Bu Yonu Seciyoruz**

- Yerel tensor seti generic Mamba contract'i ile birebir ayni degil.
- Yerel dtype gercegi `q8_0 + f32`; mevcut SSM runtime'i ise fiilen `fp16 + Mamba` varsayiyor.
- `ssm_x`, `ssm_dt.weight`, `ssm_D.weight` yokken bunlari pointer seviyesinde zorlamak segfault riskini yapisal olarak surdurur.
- Duzgun ayrim yapilirsa OOM/layout kazanimi korunur ve kalan runtime bug'i mimari olarak izole edilir.

### Gorev 1: Canonical Qwen3.5 Layer Contract

**Dosyalar:**
- Olustur: `docs/superpowers/specs/2026-03-31-qwen35-linear-operator-contract.md`
- Uygulama sirasinda oku: `qwen3.5-full-research.md`
- Uygulama sirasinda oku: `echo-core-zig/src/gguf/reader.zig:263-326`
- Uygulama sirasinda oku: `echo-core-zig/src/ports/inference.zig:20-133,567-604,796-930`
- Uygulama sirasinda oku: `echo-core-zig/src/kernels/ssm.zig:127-245`

- [ ] Yerel GGUF metadata ve tensor setini tek bir canonical contract dokumanina dondur.
- [ ] `full_attention` ve `qwen_linear` layer semantiklerini ayri basliklarla tanimla.
- [ ] Her tensor icin su alanlari sabitle: isim, shape, dtype, beklenen rol, runtime'da raw mi tutulacak yoksa convert mi edilecek.
- [ ] Bilinmeyenleri acik yaz: ozellikle `attn_qkv` ic ayrisimi ve `ssm_a/alpha/beta/dt.bias` matematikteki tam rolleri.
- [ ] Bu dokumanda su karar net olsun: `qwen_linear` generic `ssm` degildir.

**Dogrulama:**
- Yerel GGUF komut ciktilari ile dokuman birebir uyusmali.
- Dokumanda `ssm_x.weight`, `ssm_dt.weight`, `ssm_D.weight` zorunlu tensor olarak gecmemeli.
- Dokumanda `head_dim=256` kaniti `attention.key_length` metadata'sina baglanmali; `160` turetiminin yanlis/ikincil oldugu not dusulmeli.

### Gorev 2: Layer Kind ve Metadata Sozlesmesini Duzelt

**Files:**
- Duzenle: `echo-core-zig/src/core/config.zig`
- Duzenle: `echo-core-zig/src/gguf/reader.zig`
- Duzenle: `echo-core-zig/src/ports/inference.zig`
- Istege bagli test guncellemesi: `echo-core-zig/src/gguf_tests.zig`

- [ ] `LayerType` semantigini genislet: en kucuk dogru degisiklik olarak `.attention`, `.qwen_linear`, `.ssm` gibi ayri bir layer/operator ayrimi getir.
- [ ] `reader.zig` tarafinda `full_attention_interval`, `attention.key_length`, `attention.value_length`, `ssm.group_count`, `ssm.state_size`, `ssm.time_step_rank` gibi alanlari canonical sekilde oku.
- [ ] `detectActualDimensions()` icinde metadata override mantigini Qwen3.5 uyumlu kalacak sekilde koru; explicit `head_dim` uzerine yazma.
- [ ] Layer classification kurali tanimla: mevcut model icin paterni metadata + tensor presence ile belirle; salt `.ssm_out.weight var -> ssm` mantigini Qwen-specific hale getir.

**Verification:**
- `zig build`
- Gerekirse gecici dump/test ile su degerler gorulmeli: `hidden_dim=2560`, `head_dim=256`, `num_layers=32`, `num_ssm_layers=24`, `ssm_num_groups=16`.
- Layer classification, yerel paternle uyusmali: 24 `qwen_linear`, 8 `attention`.

### Gorev 3: Weight Layout ve Loader'i Qwen Linear Icin Ayir

**Files:**
- Duzenle: `echo-core-zig/src/core/memory.zig`
- Duzenle: `echo-core-zig/src/ports/inference.zig`
- Istege bagli olustur: `echo-core-zig/src/inference/qwen_linear_layout.zig`

- [ ] `WeightLayout` icinde Qwen linear tensorleri icin generic SSM offset'lerinden ayri, acik bir layout tanimla.
- [ ] Fused `attn_qkv.weight` icin aktif load path yaz; mevcut `splitFusedQKV()` ya kullanilsin ya da qwen-specific loader ile degistirilsin.
- [ ] `attn_gate.weight`, `post_attention_norm.weight`, `ssm_a`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_dt.bias`, `ssm_conv1d.weight`, `ssm_out.weight` yukleme zincirini ayri slotlarla tanimla.
- [ ] `weight_dtypes` slot haritasini gercek layout'a bagla; `engine.zig:20-27` ile `ports/inference.zig:614-930` arasindaki sabit indeks varsayimini kaldir.
- [ ] Qwen linear layer'lar icin eksik tensor fallback'larini sessizce fp16'e dusurme; eksik mandatory tensor varsa acik hata ver.

**Verification:**
- `zig build`
- Loader-only repro veya tam repro sirasinda weight load tamamlanmali.
- `raw_pool_size` mevcut 4.9 GB civarindan dramatik sekilde sapmamali; buyuk sapma varsa layout regresyonu olarak ele alinmali.
- `WARN`/fallback loglari mandatory Qwen linear tensorlerinde gorunmemeli.

### Gorev 4: Qwen Linear Runtime Path Tasarla

**Files:**
- Olustur: `echo-core-zig/src/kernels/qwen_linear.zig`
- Duzenle: `echo-core-zig/src/inference/engine.zig`
- Istege bagli olustur: `echo-core-zig/src/inference/qwen_linear_state.zig`
- Generic path'i koru: `echo-core-zig/src/kernels/ssm.zig`

- [ ] `engine.layerForward()` icinde `attention` ve `qwen_linear` dallarini ayir.
- [ ] `qwen_linear` icin ayri state modeli tanimla: conv/state buffer'lari ve varsa layer-specific recurrent buffer'lar.
- [ ] `full_attention` katmanlarinda mevcut KV cache yolunu koru.
- [ ] Generic `ssmForward` u Qwen katmanlarina cagirtma; Qwen linear icin ayrik kernel arayuzu kullan.
- [ ] State init/reset mantigini layer kind bazli calistir.

**Verification:**
- `zig build`
- Repro komutu artik `matvec.zig:26` civarinda segfault vermemeli.
- Ilk token forward en azindan `ssmForward` kaynakli pointer/format crash'ini asmali.

### Gorev 5: Dtype ve Quantization Strategy'yi Acikca Sabitle

**Files:**
- Duzenle: `echo-core-zig/src/kernels/matvec.zig`
- Duzenle: `echo-core-zig/src/kernels/quant.zig`
- Duzenle: `echo-core-zig/src/kernels/qwen_linear.zig`
- Gerekirse duzenle: `echo-core-zig/src/ports/inference.zig`

- [ ] Qwen linear icin tensor bazli dtype stratejisini netlestir:
- [ ] `q8_0` agirliklar raw quantized tutulup quant-aware matvec ile islenecek.
- [ ] `f32` norm, bias, scalar veya conv tensorleri native `f32` okunacak ya da yukleme sirasinda kontrollu convert edilecek.
- [ ] Hiçbir `f32` tensor fp16 byte dizisi gibi reinterpret edilmeyecek.
- [ ] `matvecDispatchQuant()` veya yeni yardimci yol, Qwen linear kernel'in ihtiyac duydugu `f32` ve `q8_0` kombinasyonlarini desteklemeli.

**Verification:**
- Tensor-type bazli unit testler: `q8_0`, `f32`, gerekiyorsa `q4_k/q2_k` regression.
- Kod incelemesinde `@ptrCast(... types.fp16_t ...)` sadece gercek fp16/bf16 kaynaklarda kalmali.
- Qwen linear runtime'i, `ssm_conv1d.weight` ve `ssm_a` gibi `f32` tensorleri fp16 gibi okumamali.

### Gorev 6: Gercek Verification Matrix ve Repro Zinciri

**Files:**
- Olustur: `docs/superpowers/specs/2026-03-31-qwen35-verification-matrix.md`
- Varsa testleri duzenle: `echo-core-zig/src/gguf_tests.zig`
- Istege bagli olustur: `echo-core-zig/src/tools/dump_model.zig` veya ilgili test helper'lari

- [ ] Build dogrulamasi: `zig build`
- [ ] End-to-end repro: `./zig-out/bin/echo-core-zig "../Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf" --prompt "hi" --max-tokens 1`
- [ ] Loader verification: mandatory tensorlerin bulundu gunu, mandatory olmayan klasik Mamba tensorlerinin ise aranmadigini acik log veya test ile kanitla.
- [ ] Layer-kind verification: 32 katmanda 24 `qwen_linear`, 8 `attention`.
- [ ] Non-Qwen regression verification: mevcut generic attention model path'i bozulmamis olmali; generic `ssm.zig` kullanan model varsa ona regression kontrolu eklenmeli.

**Verification:**
- Build sifir compile error ile bitmeli.
- Repro komutu segfault'suz tamamlanmali.
- Basarili durumda en az bir token uretilmeli veya yeni hata, artik operator crash degil, daha ileri seviyede deterministik bir uyumsuzluk olarak gorulmeli.

**Degismesi Muhtemel Dosyalar**

- `echo-core-zig/src/core/config.zig`
- `echo-core-zig/src/gguf/reader.zig`
- `echo-core-zig/src/core/memory.zig`
- `echo-core-zig/src/ports/inference.zig`
- `echo-core-zig/src/inference/engine.zig`
- `echo-core-zig/src/kernels/matvec.zig`
- `echo-core-zig/src/kernels/quant.zig`
- `echo-core-zig/src/kernels/ssm.zig` sadece generic path'i korumak veya ayirmak icin minimal dokunus
- `echo-core-zig/src/kernels/qwen_linear.zig` yeni dosya
- `docs/superpowers/specs/2026-03-31-qwen35-linear-operator-contract.md`
- `docs/superpowers/specs/2026-03-31-qwen35-verification-matrix.md`

**Riskler ve Bilinmeyenler**

- En buyuk bilinmeyen, `attn_qkv`, `attn_gate`, `ssm_a/alpha/beta/dt.bias` tensorlerinin matematikteki kesin roludur. Tensor isimleri ve sekilleri tek basina tam operator denklemini vermeyebilir.
- Context7 dokumani hibrit mimariyi dogruluyor, ancak tam config/schema ve operator-level ayrinti vermiyor. Bu nedenle implementasyon oncesi authoritative referans kod okuma adimi sart.
- `ssm.group_count` metadata anahtari mevcut reader tarafinda yanlis adla okunuyor; bu sadece crash fix degil, state shape dogrulugu riski de tasiyor.
- `weight_dtypes` slotleme mantigi hibrit modellerde kolayca kayabilir; bu gizli numerical bug uretebilir.
- `splitFusedQKV()` mevcut ama atil durumda; yarim entegrasyon yeni layout hatalari uretebilir.

**Subagent'lara Bolunebilir Gorevler**

- A: Yerel GGUF / tensor semantigi dogrulama
  Cikti: canonical tensor sozlugu, hangi tensor var/yok, shape/dtype tablosu, metadata anahtar listesi.
- B: Mevcut engine + loader sozlesme analizi
  Cikti: config -> layout -> load -> runtime call chain dokumu, hangisi OOM/layout, hangisi runtime/operator.
- C: Qwen3.5 linear_attention matematik ve tensor contract tasarimi
  Cikti: operator denklemi, tensor rol haritasi, state modeli, full-attention ile interface siniri.
- D: Dtype / quantization / dequant strategy
  Cikti: tensor bazli dtype policy, hangi yolda raw quantized, hangi yolda native f32, hangi noktada conversion yapilacak.
- E: Verification matrix ve gercek model repro komutlari
  Cikti: build, loader, runtime, regression checklist'i; beklenen ciktilar ve failure triage kurallari.

**Ilk Uygulanacak Milestone**

- Milestone 1: `qwen_linear` canonical contract ve layer-kind ayrimi.
- Bu milestone bitmeden kernel yazilmamali.
- Basari kosulu: ekip, `qwen_linear` katmanlarinin generic `ssm` olmadigi konusunda kod seviyesinde acik bir model elde etmis olmali; `head_dim=256`, `ssm_num_groups=16`, 24/8 layer ayrimi ve mandatory tensor listesi tek bir canonical belge ve config/load contract'i uzerinde sabitlenmis olmali.

**Bu Plan Neden Hotfix Degil, Gercek Cozum**

- Sorunu `OOM/layout` ve `operator/runtime` diye acikca ayiriyor.
- Crash'i susuturmak icin alias eklemek yerine, yanlis semantik eslemeyi ortadan kaldiriyor.
- Generic `ssm.zig` ile Qwen3.5 linear katmanlarini ayni sepetten cikartiyor.
- Dtype politikasini acik tanimlayip `f32` tensorleri fp16 gibi yorumlama hatasini yapisal olarak kapatiyor.
- Verification matrix ile sadece build degil, gercek GGUF ve gercek repro komutunu kabul kriteri yapiyor.

**Kisa Sonuc**

- Mevcut `kernels/ssm.zig` yamalanarak bitirilmemeli.
- Dogru yon, mevcut attention yolunu full katmanlar icin koruyup Qwen3.5 linear katmanlari icin ayri operator + layer execution path tasarlamaktir.
- Ilk is kernel yazmak degil, canonical Qwen linear contract'i sabitlemektir.
