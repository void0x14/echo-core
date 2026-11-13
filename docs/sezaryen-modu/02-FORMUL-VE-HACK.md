# Sezaryen Modu — Formül, Varsayımlar ve Hile Kataloğu

Bu dosya, MLSys 2026 hızlanma denklemini **duvar** olarak alır ve duvarın neresinden kırılabileceğini tek tek yazar.

---

## 1. Duvar

E = (1 - alfa^(k+1)) / ((1 - alfa) * (k * c + 1))

- **E:** hedef modele göre wall-time hızlanma.
- **alfa:** taslak tokenların kabul oranı.
- **k:** önerilen taslak uzunluğu.
- **c:** taslak üretiminin, hedef modelin tek forward'ına göre maliyet oranı.

Denklem, *Speculative Decoding: Performance or Illusion?* (MLSys 2026) tarafından üretim ortamında doğrulandı. Aynı çalışma, alfa = 1 idealinde bile doğrulama maliyetinin baskın kaldığını ve adaptif kombinasyonla üretimde 4.9x tavan oluştuğunu gösteriyor.

---

## 2. Denklemin beş varsayımı

Duvar, denklemin kendisi değil; şu beş varsayım:

1. **Doğrulama hedef model tarafından yapılmak zorunda.**
2. **Hedef modelin k+1 tokenı işlemesi, bir tokenı işlemesiyle aynı maliyette.**
3. **Taslak maliyeti c, kabul oranı alfa'dan bağımsız.**
4. **Kabul oranı alfa, pozisyondan ve içerikten bağımsız, i.i.d.**
5. **Çıktı, hedef dağılımla birebir aynı olmak zorunda.**

Bu varsayımlardan biri kırılırsa, denklemin dışına çıkılır. Aşağıdaki hile kataloğu, bu beş varsayımı hedef alır.

---

## 3. Terim terim saldırı yüzeyi

### 3.1 alfa'yı yükseltmek

alfa, "taslak doğru mu?" sorusudur. Yükseltme yolları:

- **Hidden state / feature drafter:** token yerine özellik tahmin et; hedefin iç temsili daha zengin. EAGLE, Lyanna, DFlash.
- **Token-info injection:** hidden state'e token bilgisini geç enjekte et; reddedilen taslağı çöpe atma. Lyanna.
- **Retrieval/copy:** prompt'ta/geçmişte aynı dizi varsa, gelecek token zaten biliniyor. PLD, REST, ANPD.
- **Grammar/constraint:** zorunlu metin varsa, orada tahmin yok; kesinlik var. llguidance fast-forward.
- **Çoklu drafter:** içerik bazında en iyi drafter'ı seç. Hybrid Spark, MetaSD.
- **Adaptif k:** kolay yerde uzun, zor yerde kısa taslak. SVIP, AdaEDL, PEARL.

### 3.2 k'yı büyütmek

k, aynı forward'da doğrulanan token sayısıdır. Büyütme yolları:

- **Ağaç/tree attention:** birden fazla dalı tek forward'da doğrula.
- **Blok drafter:** bir bloğu paralel üret; DFlash.
- **Adaptif k:** kabul olasılığı yüksekken k büyüt, düşükken küçült.
- **Kopya span'leri:** uzun tekrar eden dizilerde k doğal olarak büyür.

Ama k büyüdükçe doğrulama maliyeti ve ağaç yönetimi büyür. Denklemdeki k * c + 1 paydası bunu temsil eder.

### 3.3 c'yi küçültmek

c, taslak maliyetidir. Küçültme yolları:

- **n-gram/retrieval:** modele hiç girmeden aday üret; c ~= 0.
- **Hidden-state reuse:** reddedilen taslağın hidden state'ini yeniden kullan; yeniden draft etme. Lyanna.
- **Tiny drafter:** 1 katmanlı, küçük head; EAGLE.
- **Erken çıkış / katman atlama:** daha az katmanla taslak üret; dikkat, hedefle uyum riski.
- **Cache:** aynı bağlamda tekrar eden taslakları önbellekle.

### 3.4 Paydayı küçültmek: doğrulama maliyeti

Denklemdeki 1, hedef modelin doğrulama forward'ıdır. Küçültme yolları:

- **Seçici doğrulama:** her tokenı değil, kabul olasılığı yüksek olanları doğrula.
- **Batch/weight reuse:** k+1 tokenı tek forward'da işle; ağırlık okumasını paylaş. Bu repodaki ölçüm: B=256'da ~300 tok/s, B=1'de 55 tok/s.
- **IVF LM head:** LM head'i yaklaşıklaştır; zaten yapıldı (52-55 tok/s).
- **Ağaç budama:** düşük olasılıklı dalları doğrulamadan çıkar.
- **Doğrulama füzyonu:** yeniden örneklenen ağacı bir sonraki doğrulamaya birleştir; Lyanna.

### 3.5 Dağılım eşdeğerliğini bırakmak

Eğer hedef dağılım birebir korunmak zorunda değilse, alfa'nın tanımı değişir:

- **Görev doğruluğu:** kod testi, JSON şeması, hesap doğruluğu.
- **Format kısıtı:** grammar; yanlış token zaten üretilemez.
- **Kopya/retrieval:** kaynak metin doğruysa, hedefin ne diyeceğini beklemek anlamsız.

Bu modda "lossless" etiketi kullanılmaz. Etiket: **task**.

---

## 4. Varsayım kırma tablosu

| Varsayım | Kırma yolu | Örnek | Beklenen kazanç | Risk |
|---|---|---|---|---|
| Doğrulama şart | Dış kesinlik bölgesi | grammar fast-forward, retrieval | 5-10x (kısıtlı) | Yanlış token sınırı, bağlam kaybı |
| k+1 ~= 1 token | Batch/weight reuse | B=256 batch | ~5.4x (CPU) | Bellek, gecikme |
| c bağımsız | Hidden-state reuse | Lyanna | 1.4x ek | Eğitim, karmaşıklık |
| alfa i.i.d. | Yapısal kopya | PLD/REST/ANPD | 2.4-3.7x | Görev bağımlı |
| Dağılım eşdeğerliği | Task modu | kod/JSON | 5-10x | Kalite garantisi yok |

---

## 5. Hile kataloğu

Her hile için: ne yapar, nerede kullanılır, beklenen etki, maliyet, risk, etiket.

### H01 — Prompt lookup / n-gram copy
- **Ne:** Prompt ve üretilen geçmişte son n-gram'ı ara; devamını aday yap.
- **Nerede:** Özetleme, QA, kod düzenleme, tekrar eden metin.
- **Etki:** 2.4x (PLD), 3.67x (ANPD).
- **Maliyet:** Günler; eğitim yok.
- **Risk:** Kopya olmayan serbest metinde alfa düşer.
- **Etiket:** LİTERATÜR.

### H02 — Retrieval tabanlı taslak
- **Ne:** Harici veri deposundan n-gram/parça getir.
- **Nerede:** Domain içi, doküman ağırlıklı işler.
- **Etki:** 1.62-2.36x (REST).
- **Maliyet:** Haftalar; depo yönetimi.
- **Risk:** Depo kalitesi, güncellik.
- **Etiket:** LİTERATÜR.

### H03 — ESP mask-token probing
- **Ne:** Prompt embedding ortalamasını mask token olarak diziye ekle; gelecek tokenları tek forward'da sor.
- **Nerede:** Eğitimsiz, model-agnostik genel amaç.
- **Etki:** 1.22-1.48x.
- **Maliyet:** Haftalar.
- **Risk:** Kısa ufuk; ağaç yönetimi.
- **Etiket:** LİTERATÜR.

### H04 — Hidden-state reuse (Lyanna)
- **Ne:** Taslağı hidden state seviyesinde üret; reddedilen hidden state'i yeniden kullan; token bilgisini örneklemede enjekte et.
- **Nerede:** Eşdeğerlik korunacak genel amaç.
- **Etki:** SD'ye göre 3.3x; en iyi baseline'a göre 1.4x.
- **Maliyet:** Haftalar-aylar; 1 katman eğitim (68K örnek).
- **Risk:** Eğitim, bellek, karmaşıklık.
- **Etiket:** LİTERATÜR.

### H05 — DFlash blok drafter
- **Ne:** Hedefin çok katmanlı özelliklerini küçük blok-difüzyon drafter'a KV ver; bloğu paralel üret.
- **Nerede:** GPU, küçük batch, bellek-bağımlı.
- **Etki:** 6.1x (Qwen3-8B, H200).
- **Maliyet:** Aylar; eğitim.
- **Risk:** CPU'da, büyük batch'te düşer.
- **Etiket:** LİTERATÜR.

### H06 — Grammar fast-forward
- **Ne:** Zorunlu metni (JSON anahtarı, ayraç, sabit ifade) token token seçtirmeden blok ekle.
- **Nerede:** JSON, DSL, kod şablonu, yapılandırılmış çıktı.
- **Etki:** Kısıtlı işlerde 5-10x; serbest metinde 1x.
- **Maliyet:** Günler-haftalar.
- **Risk:** Token sınırı, bağlamı işleme maliyeti.
- **Etiket:** LİTERATÜR + HIPOTEZ.

### H07 — Adaptif k
- **Ne:** Taslak entropisine göre k'yı büyüt/küçült.
- **Nerede:** Uzun reasoning, değişken zorluk.
- **Etki:** Sabit k'ya göre %10-57 (AdaEDL); SVIP %17-22.
- **Maliyet:** Günler.
- **Risk:** Entropi tahmini, ölçüm gürültüsü.
- **Etiket:** LİTERATÜR.

### H08 — Seçici doğrulama
- **Ne:** Her adayı değil, kabul olasılığı yüksek olanları doğrula.
- **Nerede:** Ağaç/batch doğrulama.
- **Etki:** Denklemdeki payda küçülür; MLSys'in önerdiği yön.
- **Maliyet:** Haftalar.
- **Risk:** Yanlış eleme, dağılım sapması.
- **Etiket:** LİTERATÜR + HIPOTEZ.

### H09 — Çoklu drafter scheduler
- **Ne:** n-gram + hidden-state + grammar drafter'larını içerik bazında seç.
- **Nerede:** Karışık iş yükü.
- **Etki:** Hybrid Spark model-based'e göre 1.10-1.62x; MetaSD tek drafter'ı geçer.
- **Maliyet:** Haftalar.
- **Risk:** Fiyatlandırma/gecikme, karmaşıklık.
- **Etiket:** LİTERATÜR.

### H10 — Görev-doğruluğu modu
- **Ne:** Hedef dağılım yerine kod testi/JSON şeması/hesap doğruluğu optimize et.
- **Nerede:** Kod, yapılandırılmış çıktı, doğrulanabilir işler.
- **Etki:** Kısıtlı işlerde 5-10x.
- **Maliyet:** Günler-haftalar.
- **Risk:** "Lossless" değil; kalite garantisi yok.
- **Etiket:** HIPOTEZ.

### H11 — Batch/weight reuse
- **Ne:** k+1 tokenı tek forward'da işle; ağırlık okumasını paylaş.
- **Nerede:** CPU/GPU, bellek-bağımlı.
- **Etki:** CPU'da ~5.4x (B=256, bu repo).
- **Maliyet:** Günler.
- **Risk:** Gecikme, bellek, attention maliyeti.
- **Etiket:** ÖLÇÜLDÜ.

### H12 — IVF LM head (mevcut)
- **Ne:** LM head'i küme tabanlı yaklaşıklaştır; top-1.
- **Nerede:** Tek model, Q8_0.
- **Etki:** 38 -> 52-55 tok/s.
- **Maliyet:** Yapıldı.
- **Risk:** Logit yok; örnekleme yok.
- **Etiket:** ÖLÇÜLDÜ.

### H13 — Erken çıkış / katman atlama
- **Ne:** Taslağı erken katmandan üret; kalan katmanları atla.
- **Nerede:** Derin modeller.
- **Etki:** c düşer; alfa riski.
- **Maliyet:** Haftalar.
- **Risk:** Hedefle uyumsuzluk; eğitim gerekebilir.
- **Etiket:** LİTERATÜR.

### H14 — Token Recycling
- **Ne:** Reddedilen adayları sakla; sonraki turda yeniden kullan.
- **Nerede:** Ağaç/batch doğrulama.
- **Etki:** Ek kabul; c düşüşü.
- **Maliyet:** Haftalar.
- **Risk:** Bellek, karmaşıklık.
- **Etiket:** LİTERATÜR.

### H15 — CopyLoop / özel model ailesi
- **Ne:** Looped-Transformer'da bağlam kopyalama + sığ eleme + örtüşen yürütme.
- **Nerede:** Yalnız o model ailesi.
- **Etki:** 2.98-10.70x.
- **Maliyet:** Model ailesine özel.
- **Risk:** Genel modele taşınamaz.
- **Etiket:** LİTERATÜR.

---

## 6. En az efor, en yüksek hile sıralaması

1. **H01/H02 — n-gram/retrieval.** Eğitim yok, model değişmiyor, 2-3.7x. İlk prototip.
2. **H06 — grammar fast-forward.** Eğitim yok, kısıtlı işlerde 5-10x. JSON/kod için ilk hedef.
3. **H03 — ESP.** Eğitim yok, 1.2-1.5x. Genel amaçlı yedek.
4. **H04 — Lyanna-benzeri hidden-state reuse.** Eğitim var, 3.3x. Ciddi yatırım.
5. **H05 — DFlash-benzeri blok drafter.** Eğitim var, 6x. Uzun vadeli.
6. **H09 — çoklu drafter scheduler.** Hepsini birleştirir; en yüksek tavan.

---

## 7. 4-6x için ne gerekir?

| Hedef | Gereken kombinasyon | Nerede gerçekçi |
|---|---|---|
| 2x | PLD/REST veya ESP | Her yerde, kolay |
| 3x | Lyanna-benzeri hidden-state reuse | GPU/CPU, eğitimle |
| 4x | Lyanna + adaptif k + seçici doğrulama | GPU, küçük batch |
| 5x | Batch/weight reuse + IVF + seçici doğrulama | CPU'da sınırlı |
| 6x | DFlash-benzeri blok drafter | GPU, Qwen3-8B |
| 10x | Grammar/copy kesinlik + blok fast-forward | Kısıtlı işler; genel değil |

---

## 8. 10x için gereken koşullar

1. **alfa ~= 1** (devam dışarıdan kesin) veya **c ~= 0** (taslak bedava).
2. **k ~= 9-10** (uzun kabul bloğu).
3. **Hedef forward'ı k+1 tokenı ~1 token maliyetinde işlemeli** (GPU küçük batch; CPU'da değil).
4. **Doğrulama seçici olmalı** (her tokenı doğrulama).
5. **Dağılım eşdeğerliği ya korunmalı ya da bilinçli bırakılmalı** (task modu).

CPU'da bu koşulların 1, 2 ve 3'ü aynı anda sağlanmıyor. Bu yüzden 4-6x gerçekçi, 10x ise kısıtlı rejimlerde.

---

## 9. Hile tanımı

Burada "hile" kaliteyi kandırmak değil. Hile:

- Denklemin varsayımını ölçülebilir şekilde kırmak.
- Kazancı aynı etiketle değil, doğru etiketle raporlamak.
- Kayıpsızsa lossless, değilse task demek.

Yanlış etiket, en büyük hile kaybıdır. Çünkü bir kez "lossless" deyip saparsan, bütün ölçümler güvenilirliğini kaybeder.
