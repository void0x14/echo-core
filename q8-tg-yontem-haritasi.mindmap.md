---
title: Echo Core — Q8 TG ve Sezaryen Modu
markmap:
  colorFreezeLevel: 2
  color:
    - "#ff6b6b"
    - "#4ecdc4"
    - "#45b7d1"
    - "#96ceb4"
    - "#feca57"
    - "#ff9ff3"
    - "#54a0ff"
    - "#5f27cd"
    - "#48dbfb"
    - "#1dd1a1"
  maxWidth: 520
  initialExpandLevel: 2
  duration: 350
  lineWidth: 2
  spacingHorizontal: 100
  spacingVertical: 8
  nodeMinHeight: 18
  paddingX: 12
  autoFit: true
  fitRatio: 0.95
  maxInitialScale: 2
---

# Q8 TG: 70 yolu tıkandı

## 🗺️ Harita nasıl okunur
- **30 saniyelik özet**
  - Q8 tarafı: 70 barajı için denenen yollar ve tıkanma
  - Sezaryen tarafı: autoregressive zinciri kırma planı
- **Etiketler**
  | Etiket | Anlam |
  |---|---|
  | **KANIT** | Bu repoda ölçüldü |
  | **LİTERATÜR** | Başka yerde yayınlandı |
  | **HİPOTEZ** | Henüz denenmedi |
  | **ÇÜRÜDÜ** | Denendi, kazanç yok |
- **İlk açılacak dal**
  - **Sezaryen Modu — Tam Harita** → **0. 30 saniyelik özet**

## 📊 Tablo: hiçbir baraj geçilmedi
- **Özet tablo**
  | Yol | tok/s | Not |
  |---|---:|---|
  | Tam yol | **38** | RAM duvarı |
  | IVF 4096/512 | **52–54** | top-1 %97,75 |
  | Prefill pp512 | **282** | baraj 750 |
- **Araştırma:** 23 yöntem, 230 kaynak

## 🧮 Ana denklem
- **Formül**
  ```text
  TG = etkin bant / token baytı
  ```
- TG = etkin bant ÷ token baytı
- Token süresi ~18,3 ms
  - `mulMat` ~16 ms, tepe bantta
  - Spin ~1,6 ms
  - GDN ve çizge ~1,25 ms
- Gövde dışı sıfır olsa bile ~60–64
- 70 için (hesap)
  - Bütçe 14,29 ms
  - Gövde 389–443 MB
  - Kesim 89–143 MB

## 🧱 İp 1: Baytı değiştirmeyen elendi
- Bant tavanında çekirdek kazancı sıfır
  - M01 `fp16`: t=1 iki kat, t=6 aynı
  - M02 T0 prefetch: t=6 sıfır
  - M04 4 satırlı kernel: RAM'de kötü
  - M05 hizalı yerleşim: bayt aynı
  - M06 girdi önhesabı: bayt aynı
- Önbellek kaldıracı yok
  - M03 NTA: 802 MB ≫ 32 MB
  - M13 yardımcı thread: yanlış teşhis
  - M21 bloklama: TG'de satır 1

## ⏱️ İp 2: Gövde dışı süre (ölçüldü)
- M07 sayfa yürüyüşü: varsayımla tavan %1–3,6
- M08 `majflt`: 0, çürüdü
- M09 thread yerleşimi: etkisiz
- M10 t=7, t=8: 51,7 ve 50,2, kazanç yok
- M11 chunk boyutu: etkisiz
- M12 spin %8,6: thread'lere eşit, seri bölüm yok
- M23 durum erişim düzeni: TG'de ölçülmedi

## 🧬 İp 3: Bayt azaltan tek kol
- M14 IVF LM head: **+%47, bitti**
  - Kaçanların p95 farkı ~13 logit
  - Yaklaşık yol aynı metni üretmez
- M15 Cauchy-Schwarz: eleme yok
- M16 sözlük öneki: sıklık sırası denenmedi
- M17 kayıpsız: pay ~%4, tek başına yetmez
- M18 FFN kanal atlama: **70 için öldü**
  - %55 kapı: top-1 %64,9
  - %55 oracle: top-1 %88,7
  - %88 oracle: top-1 %58,5
  - %5–20 kapı: %97,9–92, 8–32 MB

## ➗ İp 4: Paydayı bölen
- M19 n-gram batch: **70 için öldü**
  - Gereken isabet ≥%54 (hesap)
  - Ölçülen %25–28, tavan ~58
  - Kopya metinde geçerli değil

## 🧪 Kenarda: prefill
- TG bitmeden dokunulmaz
- M20 dinamik tile: +%4,2
- M21 `rb=16`: düşüş; 256×256 Q8 denenmedi
- M22 + M23: +%3,3, ayrılmadı
- Denenmedi: concat üretmeden `ssm_conv`

## 🧭 Tek yol: durum
- [x] 1. Gövde dışı süre ayrıldı
- [x] 2. Spin onarımı düştü
- [x] 3. nprobe 512 yapıldı
- [x] 4. M18 oracle öldü
- [ ] 5. Denenen yöntemlerle 70 yok

## 🧰 Denenmemiş kaldıraçlar
- **Kaldıraç tablosu**
  | Kaldıraç | Beklenen | Durum |
  |---|---|---|
  | Bant | bellek hızı | donanım |
  | GDN durum trafiği | ~36 MB | açık |
  | Küçük FFN kesimi | 8–32 MB | açık |
  | Tek başına 70 | yetmez | hesap |
- **Not:** Süre tepe banda dayalı.

## 🧠 Sezaryen Modu — Tam Harita
- **Tek bakışta**
  | Soru | Cevap |
  |---|---|
  | Duvar nedir? | MLSys hızlanma denklemi |
  | Neden 1–2x? | alfa/c aynı Pareto'da |
  | En ucuz hile? | Deterministik bölge + n-gram |
  | İlk kod adımı? | KV rollback + tek zincir + n-gram |

### 0. 30 saniyelik özet
- Problem
  - Model her adımda tek token üretir
  - Token geri girer, zincir büyür
  - Zincir büyüdükçe darboğaz büyür
  - CPU'da token süresi ~18,3 ms
- Tez
  - Modelin kurallarına uymak zorunda değiliz
  - Karnını aç, üretim boyunca açık tut
  - Dışarıda hızlı token/özellik üret
  - İç katmanlara püskürt
  - Motor model-agnostik olsun
- Kırmızı çizgi
  - Model mimarisi değişmez
  - Quantizasyon değişmez
  - Manuel model ameliyatı yok
  - Adaptör motorun içinde
  - Lossless ve task ayrı etiket
- Duvar
  - MLSys 2026 hızlanma denklemi
  - E = (1 - alfa^(k+1)) / ((1 - alfa)(k c + 1))
  - Duvar denklem değil, beş varsayım
- İlk adım
  - KV rollback/truncate
  - Tek zincir doğrulama
  - n-gram drafter
  - Benchmark: alfa, c, k
- Bugün karar ver
  - Lossless mı, task mı?
  - İlk deney: n-gram + tek zincir
  - 2x hedefiyle başla

### 1. Tez ve kırmızı çizgi
- Tez, uzun hali
  - Autoregressive zincir doğal darboğaz
  - Modelin kendi kurallarına uymak zorunda değiliz
  - Dışarıda çok daha hızlı üretici olabilir
  - Üretilenler doğrudan iç katmana girebilir
  - Model dosyasına dokunmadan
- Neden "sezaryen"
  - Karnını aç: iç katmanlara erişim
  - Açık tut: üretim boyunca hook açık
  - Kapat: iş bitince normal yola dön
  - Metafor, mimari değişiklik değil
- Neden "takyon çarpıştırıcısı"
  - Dışarıda sürekli aday üreten hızlandırıcı
  - Token veya özellik üretebilir
  - Hedef modelin doğrulaması sonlu
  - Sonsuz akış doğrudan modele verilemez
- Kırmızı çizgi
  - Model mimarisi değişmez
    - Katman sayısı, attention, tokenizer aynı
  - Quantizasyon değişmez
    - Q8_0 dosyası Q8_0 kalır
  - Manuel model ameliyatı yok
    - Kullanıcı modelin içine elle girmez
  - Adaptör motorun içinde
    - Model dosyasının dışında
    - Capability adapter
  - Lossless/task ayrı etiket
    - Lossless: hedef dağılım korunur
    - Task: görev doğruluğu yeterli
- İki kol
  - Kol 1: eşdeğerlik korunacak
    - PLD/REST, ESP, Lyanna, DFlash
    - Hedef 2-4x, iyi senaryoda 6x
  - Kol 2: görev doğruluğu yeterli
    - Grammar, JSON, kod, retrieval
    - Hedef kısıtlı alanda 5-10x
  - Etiketler karışmaz
    - Lossless sonucu task sonucuyla aynı tabloya girmez

### 2. Duvar: MLSys denklemi
- Denklem
  - E = (1 - alfa^(k+1)) / ((1 - alfa)(k c + 1))
  - E: wall-time hızlanma
  - alfa: kabul oranı
  - k: taslak uzunluğu
  - c: taslak/hedef forward maliyeti
- Denklem ne söylüyor
  - Taslak kalitesi ve maliyeti bağlı
  - Daha iyi taslak daha pahalı
  - Daha ucuz taslak daha zayıf
  - Doğrulama maliyeti baskın
  - İdeal alfa=1 bile 4.9x tavan
- Beş varsayım
  - Doğrulama hedef modelde
    - Kırma: dış kesinlik bölgesi
  - k+1 ~= 1 token
    - Kırma: batch/weight reuse
  - c bağımsız
    - Kırma: hidden-state reuse
  - alfa i.i.d.
    - Kırma: yapısal kopya/retrieval
  - Dağılım eşdeğer
    - Kırma: task modu
- Duvarın yeri
  - Denklem kanun değil
  - Varsayımlar kırılabilir
  - Hile = varsayım kırma
  - En ucuz hile: deterministik bölge
- Denklemin tuzakları
  - Sabit k varsayımı
  - Bağımsız alfa varsayımı
  - Taslak maliyeti sabit varsayımı
  - Hedef forward'ın k+1 tokenı 1 token sayması
  - Dağılım eşdeğerliği şartı

### 3. A/B tartışması
- A tarafı: eğitimsiz + embedding
  - Neden eğitimsiz
    - Sıfır eğitim maliyeti
    - Model-agnostik
    - Quantizasyon/mimari değişmez
    - Hızlı prototip
  - Neden embedding
    - Modelin kendi giriş uzayı
    - Dağılım içi mask token
    - Yeni ağırlık gerekmez
  - Örnekler
    - ESP: mask token probing
    - PLD: n-gram
    - REST: retrieval
  - Zayıf taraf
    - alfa düşük
    - Kısa ufuk
    - Ağaç yönetimi
- B tarafı: eğitimli + hidden state
  - Neden eğitimli
    - Hidden state daha zengin
    - Gelecek token bilgisi taşır
    - Küçük projeksiyon öğrenilebilir
  - Neden hidden state
    - Token bilgisi değil, bağlam bilgisi
    - Future Lens: %48 tahmin
  - Örnekler
    - EAGLE: hidden + token
    - Hidden Transfer: projeksiyon
    - FIRP: lineer tahmin
  - Zayıf taraf
    - c yüksek
    - Eğitim gerekir
    - Model değişince yeniden eğitim
- A/B yanlış eksen
  - ESP hidden state'ten yararlanır
  - EAGLE token embedding kullanır
  - PaSS: eğitimli embedding
  - CLaSp: eğitimsiz hidden state
  - Lyanna: hidden draft + token-info
- C var
  - Lyanna
    - Hidden-state draft
    - Token-info injection
    - Reddedilen hidden state reuse
    - 3.3x (SD'ye karşı)
    - 1.4x (en iyi baseline'a karşı)
  - DFlash
    - Çok katmanlı hidden bağlam
    - Blok difüzyon drafter
    - 6.1x (Qwen3-8B)
  - Hybrid Spark
    - Model-based + model-free
    - İçerik bazında seçim
    - 1.10-1.62x ek
  - MetaSD
    - Çoklu drafter
    - Bandit seçimi
  - Falcon / ParallelSpec
    - Yarı-AR / paralel drafter
    - 2.84-3.51x
- Sonuç
  - A ve B rakip değil
  - Aynı hattın iki kesiti
  - Hibritler zaten var
  - Tartışma ekseni: hangi belirsizlik, hangi seri iş, hangi garanti

### 4. Kanıt bandı
- **Bant özeti**
  | Bant | Temsilci | Hız |
  |---|---|---:|
  | 1–2x | ESP, EAGLE | 1.2–3x |
  | 2–4x | PLD, Lyanna, ANPD | 2.4–3.7x |
  | 4–6x | MLSys adaptif, DFlash | 4.9–6.1x |
  | 10x | CopyLoop, grammar | 10.7x / kısıtlı |
- 1-2x
  - ESP 1.22-1.48x
  - Lookahead 1.8x
  - EAGLE 2-3x
  - Medusa 2-3x
- 2-4x
  - PLD 2.4x
  - REST 1.62-2.36x
  - ANPD 3.67x
  - Lyanna 3.3x
  - Falcon 2.91-3.51x
  - ParallelSpec 2.84x
- 4-6x
  - MLSys adaptif 4.9x
  - DFlash 6.1x
  - PEARL 4.43x
- 10x
  - CopyLoop 2.98-10.70x
    - Looped-Transformer
    - Özel model ailesi
    - RTX 5090
  - Grammar fast-forward
    - Kısıtlı çıktı
    - JSON/DSL/kod
- Bu repoda
  - Tam yol 38 tok/s
  - IVF 52-55 tok/s
  - Batch B=256 ~300 tok/s
  - Yani ~5.4x
- Uyarı
  - Sayılar farklı donanım/model/görev
  - Bu repoda aynı sayı garanti değil
  - Her yeni deney kendi etiketini taşır

### 5. Hile kataloğu
- **Hızlı seçim**
  | Hile | Etki | Eğitim | Ne zaman |
  |---|---|---|---|
  | H01 n-gram | 2.4x | yok | kopya |
  | H04 Lyanna | 3.3x | var | lossless |
  | H05 DFlash | 6.1x | var | GPU |
  | H06 grammar | 5–10x | yok | JSON/kod |
- H01 n-gram/PLD
  - Ne: prompt/geçmişte son n-gram'ı ara
  - Etki: 2.4x
  - Maliyet: günler, eğitim yok
  - Risk: kopya olmayan metinde alfa düşer
  - Ne zaman: özetleme, QA, kod
  - Hangi varsayım: alfa i.i.d. değil
- H02 retrieval/REST
  - Ne: harici depodan parça getir
  - Etki: 1.62-2.36x
  - Maliyet: haftalar
  - Risk: depo kalitesi
  - Ne zaman: domain içi, doküman
- H03 ESP
  - Ne: prompt embedding ortalaması mask token
  - Etki: 1.22-1.48x
  - Maliyet: haftalar
  - Risk: kısa ufuk
  - Ne zaman: eğitimsiz genel amaç
- H04 Lyanna
  - Ne: hidden-state draft + token-info + reuse
  - Etki: 3.3x
  - Maliyet: 1 katman eğitim
  - Risk: eğitim/bellek
  - Ne zaman: lossless, c düşük, alfa yüksek
- H05 DFlash
  - Ne: hidden bağlam + blok difüzyon
  - Etki: 6.1x
  - Maliyet: GPU/eğitim
  - Risk: CPU'da yok
  - Ne zaman: GPU, küçük batch
- H06 grammar fast-forward
  - Ne: zorunlu metni blok ekle
  - Etki: 5-10x kısıtlı
  - Maliyet: günler-haftalar
  - Risk: token sınırı
  - Ne zaman: JSON/DSL/kod
  - Hangi varsayım: doğrulama şart değil
- H07 adaptif k
  - Ne: entropiye göre k büyüt/küçült
  - Etki: %10-57
  - Maliyet: günler
  - Risk: tahmin gürültüsü
- H08 seçici doğrulama
  - Ne: sadece yüksek olasılıklı adayı doğrula
  - Etki: payda küçülür
  - Maliyet: haftalar
  - Risk: yanlış eleme
- H09 çoklu drafter
  - Ne: n-gram + hidden + grammar seç
  - Etki: +1.1-1.6x
  - Maliyet: haftalar
  - Risk: scheduler overhead
- H10 task modu
  - Ne: kod testi/JSON şeması optimize
  - Etki: 5-10x kısıtlı
  - Maliyet: günler-haftalar
  - Risk: lossless değil
- H11 batch/weight reuse
  - Ne: k+1 tokenı tek forward
  - Etki: ~5.4x CPU
  - Maliyet: günler
  - Risk: gecikme/bellek
- H12 IVF LM head
  - Ne: LM head'i yaklaşıklaştır
  - Etki: 38 -> 52-55
  - Maliyet: yapıldı
  - Risk: logit yok
- H13 erken çıkış
  - Ne: erken katmandan taslak
  - Etki: c düşer
  - Risk: alfa düşer
- H14 token recycling
  - Ne: reddedilen adayı sakla
  - Etki: ek kabul
  - Risk: bellek
- H15 CopyLoop
  - Ne: looped modelde kopya + örtüşme
  - Etki: 10.7x
  - Risk: genel modelde yok

### 6. Karar ağacı
- Çıktı deterministik mi?
  - Evet: grammar fast-forward
    - JSON, DSL, kod şablonu
    - 5-10x
  - Hayır: devam
- Kaynak/geçmişte kopya var mı?
  - Evet: n-gram/retrieval
    - PLD/REST/ANPD
    - 2.4-3.7x
  - Hayır: devam
- Lossless şart mı?
  - Evet: ESP veya Lyanna
    - ESP 1.2-1.5x
    - Lyanna 2-3x
  - Hayır: task modu
    - 5-10x kısıtlı
- GPU var mı?
  - Evet: DFlash araştır
    - 6.1x
  - Hayır: batch/IVF
    - ~5.4x CPU
- Eğitim bütçesi var mı?
  - Evet: Lyanna/DFlash
  - Hayır: ESP/n-gram/grammar
- En az eforla en yüksek hile
  - 1. n-gram/retrieval
  - 2. grammar fast-forward
  - 3. ESP
  - 4. Lyanna
  - 5. DFlash
  - 6. hibrit scheduler

### 7. Deney planı
- [ ] Faz 0: PLD/REST
  - Hedef: 2x
  - İlk adım: KV rollback + tek zincir
  - Başarı: >=1.8x
  - Durma: 1.3x altı
- [ ] Faz 1: ESP
  - Hedef: 1.2-1.5x
  - Başarı: >=1.2x lossless
  - Durma: 1.1x altı
- [ ] Faz 2: Lyanna
  - Hedef: 2-3x
  - Başarı: >=2.5x
  - Durma: eğitim bütçesi yok
- [ ] Faz 3: DFlash
  - Hedef: 4-6x GPU
  - Başarı: >=4x
  - Durma: CPU-only hedef
- [ ] Faz 4: grammar
  - Hedef: 5-10x kısıtlı
  - Başarı: >=3x, şema %100
  - Durma: token sınırı
- [ ] Faz 5: hibrit
  - Hedef: 4x+ karışık
  - Başarı: regresyon yok
  - Durma: scheduler overhead
- Metrikler
  - alfa: kabul/öneri
  - c: taslak/hedef süre
  - k: öneri uzunluğu
  - speedup: baseline/spec
  - kabul dağılımı
  - taslak/doğrulama/cache ayrı
  - kalite: lossless/task
- Rapor
  - Her deney mod etiketi taşır
  - Sayı + kaynak + etiket
  - Yanlış etiket = geçersiz

### 8. İlk kod adımı
- **Komut iskeleti**
  ```bash
  # 1) KV rollback/truncate
  # 2) tek zincir doğrulama
  # 3) n-gram drafter
  # 4) benchmark: alfa, c, k
  ```
- 1. KV rollback/truncate
  - Neden: Context.head ileri gidiyor
  - Nerede: qwen35.zig Context
  - Risk: yanlış rollback sessiz sapma
- 2. Tek zincir doğrulama
  - Aday k tokenı tek decode
  - Her pozisyon logit
  - En uzun eşleşen prefix
  - Neden: ağaç yok, en basit
- 3. n-gram drafter
  - Prompt + geçmiş
  - Trie/lookup
  - Faz 0
- 4. Benchmark
  - Baseline vs spec
  - alfa, c, k
  - Aynı prompt seti
- 5. ESP embedding injection
  - Input embedding yolu
  - Mask token
  - Ağaç sonra
- 6. Hidden-state reuse
  - Draft model
  - Token-info injection
  - Lyanna benzeri
- Sıra neden bu
  - En az mühendislik
  - En yüksek kanıt
  - Model dosyasına dokunmaz

### 9. Riskler ve tuzaklar
- KV rollback yanlış
  - Sessiz bağlam sapması
  - Yanlış kabul
- RoPE/pozisyon yanlış
  - Yanlış bağlam
  - Uzun bağlamda çöküş
- Tokenizer sınırı
  - Doğru metin yanlış token
  - Grammar fast-forward bozulur
- Ağaç maliyeti
  - Kazancı yer
  - Batch küçükse
- Eğitimli drafter
  - Model değişince çöp
  - Adaptör bakımı
- Task modu
  - Lossless değil
  - Kalite garantisi yok
- Yanlış etiket
  - Güven kaybı
  - Bütün ölçümler şüpheli
- Farklı donanım sayısı
  - CPU/GPU karışır
  - Karar yanlış

### 10. Sözlük
- alfa
  - Taslak tokenların kabul oranı
- c
  - Taslak forward / hedef forward
- k
  - Aynı turda önerilen token sayısı
- lossless
  - Hedef dağılım korunur
- task
  - Görev doğruluğu yeterli, dağılım değil
- drafter
  - Aday token/özellik üretici
- tree attention
  - Ağaç dallarını tek forward'da doğrulama
- KV rollback
  - Reddedilen tokenların cache etkisini geri alma
- token-info injection
  - Token bilgisini örnekleme aşamasında ekleme
- hidden-state reuse
  - Reddedilen taslağın hidden state'ini yeniden kullanma
- capability adapter
  - Modelin katman/shape/cache bilgisini motora veren küçük sözleşme

### 11. Kaynaklar
- MLSys 2026
  - Speculative Decoding: Performance or Illusion?
  - Hızlanma denklemi, alfa=1 simülasyonu, 4.9x adaptif tavan
- ESP
  - [arXiv 2603.17942](https://arxiv.org/abs/2603.17942)
  - Training-free mask token probing
- Lyanna
  - [arXiv 2602.21224](https://arxiv.org/abs/2602.21224)
  - Hidden-state draft + token-info + reuse
- DFlash
  - [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
  - Hidden bağlam + blok difüzyon, 6.1x
- EAGLE / EAGLE-3
  - [arXiv 2401.15077](https://arxiv.org/abs/2401.15077) / 2503.01840
- Hidden Transfer
  - [arXiv 2404.12022](https://arxiv.org/abs/2404.12022)
- FIRP
  - [arXiv 2410.20488](https://arxiv.org/abs/2410.20488)
- PLD
  - [PLD notu](https://ryanorban.com/notes/prompt-lookup-decoding/)
- REST
  - [arXiv 2311.08252](https://arxiv.org/abs/2311.08252)
- ANPD
  - [arXiv 2404.08698](https://arxiv.org/abs/2404.08698)
- Lookahead
  - [arXiv 2402.02057](https://arxiv.org/abs/2402.02057)
- llguidance fast-forward
  - [llguidance](https://github.com/guidance-ai/llguidance)
- CopyLoop
  - [arXiv 2610.38822](https://arxiv.org/abs/2610.38822)
- Hybrid Spark
  - [arXiv 2611.05611](https://arxiv.org/abs/2611.05611)
- MetaSD
  - [arXiv 2604.05417](https://arxiv.org/abs/2604.05417)
- PaSS
  - [arXiv 2311.13581](https://arxiv.org/abs/2311.13581)
- CLaSp
  - ACL 2025
- Token Recycling
  - ACL 2025
- SVIP
  - [arXiv 2411.18462](https://arxiv.org/abs/2411.18462)
- AdaEDL
  - [arXiv 2410.18351](https://arxiv.org/abs/2410.18351)
- PEARL
  - [arXiv 2408.11850](https://arxiv.org/abs/2408.11850)
- LK Losses
  - [arXiv 2602.23881](https://arxiv.org/abs/2602.23881)
- TransformerLens
  - [TransformerLens](https://transformerlensorg.github.io/)
- IBM vLLM Hook
  - [IBM vLLM Hook](https://github.com/IBM/vLLM-Hook)

## ⚠️ Açık riskler
- **Risk tablosu**
  | Risk | Etki | Durum |
  |---|---|---|
  | IVF logit vermiyor | örnekleme yok | açık |
  | Yaklaşık çıktı sapıyor | tam yolla aynı değil | açık |
  | Görev doğruluğu | ölçülmedi | açık |
  | 100 için bütçe | 10 ms | açık |
- **Not:** Bu riskler lossless iddiasını sınırlar.

## 📅 6 Kas sonrası — eksiksiz kayıt
- **Kurallar (kullanıcı)**
  - [x] Her hız **tok/s** ile verilir; ms kullanılmaz
  - [x] Ana harita bu dosya; başka harita yok
  - [x] Her iş bu haritaya eksiksiz eklenir
  - [x] Yalnız 1 çekirdek, 1 thread (`-ç 1`)
  - [x] Yalnız Qwen3.5-0.8B-Q8_0; mimari, quantizasyon, katman sayısı değişmez
  - [x] Test dosyası yok
- **5 Kas — IVF varsayılan** (`a2ce939`, 6 çekirdek)
  | Yol | tg (tok/s) |
  |---|---:|
  | Tam LM head | 39,5 |
  | **IVF 4096/512** | **55,0** |
- **5 Kas — B (aynı anda işlenen token) taraması** (`--gen-batch`, 6 çekirdek)
  | B | tok/s |
  |---:|---:|
  | 1 | 54,6 |
  | 10 | 239,9 |
  | 128 | 290,1 |
  | **256** | **~300 (karar)** |
  | 512 | 295,7 |
  - [x] 10 Kas: üretim yoluna taşındı — `qwen35.zig` Context.init `n_ubatch` 512 → **256** (prefill ve doğrulama aynı yoldan geçer)
  - [x] 10 Kas: 1 çekirdekte derleme + tok/s ölçümü (ajan) — `taskset -c 0`, `-t 1`, `ECHO_SKIP=` boş, r=5, Qwen3.5-0.8B-Q8_0
    | n_ubatch | pp128 | pp512 | tg32 | tg128 | Kat (pp128 / pp512 / tg32 / tg128) |
    |---|---:|---:|---:|---:|---|
    | 512 (önce) | 144,9 | 148,9 | 57,4 | 43,6 | 1,00x |
    | 256 (sonra) | 151,8 | 146,4 | 57,9 | 46,8 | 1,05x / 0,98x / 1,01x / 1,07x |
    - gerçek prompt ("Türkiye'nin başkenti neresidir?"): "Türkiye'nin başkenti **Ankara**'dır." — bozuk/döngü yok, tg 37,6 tok/s (16 token, kısa koşu)
    - sonuç: fark ölçüm gürültüsü içinde; 256 zararsız, kalır
- **5–6 Kas — Sezaryen Modu planı** → `docs/sezaryen-modu/` (00–04)
- **7 Kas — H01 + H06** (BENCHMARK-CPU-GPU-2026-11-04.md, 6 çekirdek)
  | Mod | tg128 (tok/s) | Kat |
  |---|---:|---:|
  | Hilesiz | 53,64 | 1,00x |
  | H06 grammar | 74,20 | 1,38x |
  | H01 n-gram | 120,67 | 2,25x |
  | H01 + H06 | 121,85 | 2,27x |
  - ⚠️ `--llama-bench` TG boş prompt ile başlar, model döngüye girer, n-gram döngüyü kopyalar (kabul 31/31); gerçek metin sayısı değil
- **7–8 Kas gece — hız barajları** (H13–H70)
  | Baraj | Sayı | Nasıl | Gerçek mi |
  |---|---:|---|---|
  | TG 200 | 270–277 | R8 asm çekirdek | ⚠️ bench döngüsü + n-gram |
  | TG 500 | 507 | GDN, iş çalma, n-gram 128 | ⚠️ aynı |
  | TG 750 | 762 | faz birleştirme, RT | ⚠️ aynı |
  | TG 1000 | 1157–1270 | **8 katman atlama** | ❌ çıktı bozuk |
  | Prefill 1000/1500/2000 | 1150/1613/2109 | **katman atlama + FFN budama** | ❌ çıktı bozuk |
  - ⚠️ `qwen35.zig` Model.load içinde gizli varsayılan: ECHO_SKIP yoksa 8 katman atlanır (hâlâ kodda)
- **8 Kas — kullanılabilirlik planı** (`KULLANILABILIRLIK-PLANI.md`)
  - [x] Tokenizer resmi ile aynı (6903/6903, NFC)
  - [x] Sohbet CLI, sunucu (OpenAI uyumlu), önek yeniden kullanımı
  - [x] Araç çağırma (`tools`, `tool_calls`), şablon resmi ile 12/12 aynı
- **8–9 Kas — hatalar**
  - `--fast` modu sorulmadan yaratıldı
  - 18 katman atlama + FFN 1920 "1180 tok/s" diye sunuldu → çıktı `0 0 0`
  - Sorulmadan taslak + doğrulama düzeni kuruldu → 1 çekirdekte 18,9 tok/s
- **10 Kas — bozan parça** (1 çekirdek, "hi whats going on bro")
  | Atlama | FFN | Çıktı | tg (tok/s) |
  |---:|---|---|---:|
  | 18 | 1920 | `1 3 0 0 0 0…` | 285,5 |
  | 18 | tam | `A A B 1 1 1 2…` | 101,2 |
  | 12 | tam | `长龙龙龙…` | 167,6 |
  | 8 | 1920 | `The text goes on.` döngü | 143,9 |
  | 8 | tam | `the origin of the origin…` | 65,7 |
  | 0 | 1920 | rol karışık, döngü | 48,1 |
  | **0** | **tam** | **doğru** | **33,7** |
  - Katman atlayan her ayar bozuk; FFN budama tek başına da bozuk; IVF sebep değil
- **10 Kas — tek düzen** (commit `3d368d2`, `a7d086f`, `c818334`)
  - Düzen: atlama yok, FFN tam, IVF LM head, n-gram 2/3/128, greedy
  - `--fast`, `--exact`, `--prefill-fast` kaldırıldı; testler silindi
  - `echo-core` her klasörden açılır; `-ç N` = N çekirdek
  - **Gerçek tg: ~35–38 tok/s, 1 çekirdek**, çıktı doğru (EN + TR)
  - Açık sorun: Türkçede 2. mesajdan sonra döngü
- **10 Kas — hile ajanları** (Sonnet, ayrı worktree, 1 çekirdek) — ana dala birleştirildi `73c934c`
  - [x] H02 retrieval/REST — çalışıyor, kazanç yok (`--rest DİZİN`, varsayılan kapalı)
  - [x] H03 ESP — çalışıyor, yavaş (`--esp K`, varsayılan kapalı)
  - [-] H04 Lyanna — atlandı: eğitim gerekiyor, eğitim istenmedi
  - [x] H10 task modu — sunucuda araç çağrısı geçerli, ~2,9x (`--task`, sohbet CLI'de etkisiz)
  - [x] 10 Kas: **H10 varsayılan** — `--task` bayrağı silindi; sunucuda `tools` varsa iskelet her zaman açık (1 çekirdek, aynı araç isteği, 3 tekrar)
    | Mod | tok/s | Kat | Çağrı |
    |---|---:|---:|---|
    | H10 kapalı (eski) | 35,2 | 1,00x | 0/3, düz metin |
    | **H10 varsayılan** | **93,4** | **2,65x** | 3/3 geçerli |
    - Araçsız istek: H10 devreye girmiyor (0 blok), "Ankara" cevabı aynı, 39,9 tok/s
- **10 Kas — Sezaryen hileleri, tarihli ölçüm** (1 çekirdek, 1 thread: `taskset -c 0`, `ECHO_THREADS=1`, `ECHO_SKIP=` boş, greedy, 128 token, Qwen3.5-0.8B-Q8_0)
  | Satır | tg32 (tok/s) ⚠️ | tg128 (tok/s) ⚠️ | gerçek prompt tg, genel (tok/s) | gerçek prompt tg, depo (tok/s) | Kat (genel / depo) | çıktı doğru mu |
  |---|---:|---:|---:|---:|---|---|
  | Baseline (D düzeni) | 117,1 | 114,9 | 60,3 | 54,7 | 1,00x / 1,00x | ilk 2 satır doğru; sonda döngü (baseline'da da var) |
  | H02 retrieval (`--rest`) | 31,5 | 39,1 | 56,3 | 52,7 | 0,93x / 0,96x | genel: baseline ile bit aynı; depo: ilk satırlar aynı, sonra sayısal ayrışma |
  | H03 ESP (`--esp 1`) | 23,8 | 30,6 | 27,4 | 27,3 | 0,45x / 0,50x | genel: baseline ile bit aynı; depo: sayısal ayrışma |
  | H04 Lyanna | atlandı | atlandı | atlandı | atlandı | — | eğitim istenmedi |
  | H10 görev modu (CLI, `--task`) | — | — | 59,8 | — | 0,99x | baseline ile bit aynı (CLI'de etkisiz) |
  | Baseline sunucu, araç isteği | — | — | 35,1 | — | 1,00x | 3 istekte 0/3 geçerli çağrı (düz metin) |
  | H10 sunucu, araç isteği (`--task`) | — | — | 101,6 | — | 2,90x | 3 istekte 3/3 geçerli çağrı (`hava_durumu`) |
  - ⚠️ tg32/tg128 `--llama-bench` boş prompt ile ölçüldü; baseline'da n-gram döngüyü kopyalıyor (kabul 89/92), sahte. Gerçek sonuç için gerçek prompt sütunu geçerli.
  - Genel: 60,3 ve depo: 54,7, 5 tekrarın medyanı (baseline ile dönüşümlü). H03 ayrı koşuldu (3 tekrar); aradaki fark 2 kattan büyük.
  - Araç isteğinde H10 43 token'ın 40'ını şemadan zorlayıp yalnız 3 token üretti; baseline düz metin üretti, yani kıyas aynı işi yapmıyor.
  - Özet:
    - H02: depoda eşleşme var (kabul 23/24), ama her taslakta ~1 token kazanılıyor; depo dışı metinde eşleşme yok. Net kayıp -4% ile -7%. REST_MIN 2 ve 1 denendi: 35 ve 31 tok/s.
    - H03: maske tokenlar her turda ek yük, kabul %16 (10/63). K=1..4, kayma 0,1 ve 0,3 denendi (8 koşu); en iyisi K=1 (27 tok/s), baseline'ın yarısı.
    - H10: araç çağrısında hızlandırıyor ve çağrıyı geçerli kılıyor. Sohbet ve araçsız istek değişmiyor. Kayıpsız değil, yalnız çağrı iskeleti zorlanıyor. Açık: tek şema ile ölçüldü.
    - `--no-ngram` artık n-gram'ı kapatıyor; önceden `--fast` bunu geçersiz kılıyordu.
- **10 Kas — B tek geçiş bedeli** (1 çekirdek, 1 thread: `taskset -c 0`, `ECHO_THREADS=1`, `ECHO_SKIP=` boş, `-t 1`; IVF ve n-gram/grammar kapalı; 2 ısınma, 7 tekrarın medyanı; Qwen3.5-0.8B-Q8_0; derleme HEAD `776c9a2`)
  | B | işlenen tok/s | geçiş/s | bedel |
  |---:|---:|---:|---:|
  | 1 | 23,98 | 23,98 | 1,00 |
  | 2 | 43,70 | 21,85 | 1,10 |
  | 4 | 69,29 | 17,32 | 1,38 |
  | 8 | 97,31 | 12,16 | 1,97 |
  | 16 | 122,58 | 7,66 | 3,13 |
  | 32 | 142,01 | 4,44 | 5,40 |
  - Bedel = B tokenlık geçişin süresi / 1 tokenlık geçişin süresi (1,00 = bedava). Hesap: B × tok/s(1) / tok/s(B).
  - Taban (aynı koşul, tek token üretimi): tg32 = 24,26 tok/s, tg128 = 24,17 tok/s. Haritadaki tg32 = 57,9 (1 çekirdek) IVF ve n-gram açıkken ölçülmüştü, bu tabanla karşılaştırılamaz.
  - Geçişin ~%86'sı sabit yük (ağırlık okuma, LM head), token başına ~%14. B=16'dan 32'ye 16 ek token tok/s'yi yalnız %16 artırıyor.
  - Durum katmanları (`qwen35.zig`): 24 ana katmanın 18'i durum katmanı, 6'sı dikkat katmanı (satır 114–115, 161). Durum katmanı başına 1.122.304 bayt (conv 73.728 + SSM 1.048.576, f32); 18 katman ~19,3 MiB, token sayısıyla büyümez. Dikkat katmanlarının KV'si büyür.
  - Geri sarma var: `restore` (satır 712–721, batch içi j. token), `snapshot` / `rollback` (725–749), `kaydet` / `yukle` (767–787), `clear` (789).
  - ⚠️ `echo-core-zig/src/llama/sampler.zig:36` derlenmiyor (çalışma ağacında, commit edilmemiş, dosya sonu hatası). Çalışma ağacı `zig build` ile derlenmiyor; ölçüm temiz HEAD derlemesiyle yapıldı.
  - Komut: `env ECHO_THREADS=1 ECHO_SKIP= taskset -c 0 <bin> --llama-bench -m <model> -t 1 --lm-ivf 0,0 --ngram 0 --grammar 0 --p6 0 --snap 0 --memory 0 -p B -n 0 -r 7`
- **10 Kas — Sezaryen fikir alışverişi**
  - [ ] **Denenecek — Hızlı ağırlık yazma** (Claude'un kozu; kullanıcı reddetti: tarif değil, ama ayrıca denenecek)
    - Qwen3.5'in 18 DeltaNet katmanında her token ile yeniden yazılan durum matrisi var: 16×128×128 (rs_state, `qwen35.zig:478`, `:487`; `gatedDeltaNet` `:1107`). `ssm_alpha` ve `ssm_beta` bu matris değil, 1024×16 ağırlıklar (`ArastirmaModelleri/Qwen3.5-0.8B-Q8_0_tensor_analysis.txt:56-66`).
    - Fikir: partikülü token olarak değil, bu matrise doğrudan anahtar→değer yazımı olarak emdirmek.
    - Kaynak: Schlag/Irie/Schmidhuber, ICML 2021; Yang/Kautz/Hatamizadeh, ICLR 2025 (Gated Delta Networks).
    - Ölçülmedi.
  - **Kullanıcı tarifi — ultra birleşik oda**: prefill ve üretim tek odada; a→b hattı, köprü, transfer yok; partikül yaratıldığı anda mıknatısla çekilir.
    - Sorun tanımı: her token için CPU → önbellekler → RAM → geri döngüsü (7 babayiğit: yazmaç, L1, L2, L3, bellek denetleyici, bellek yolu, RAM).
    - Geçişin ~%86'sı sabit yük (B tablosu).
  - **Doğrulama / lossless şartı** (Leviathan 2023, DeepMind Chen 2023): kullanıcı tarafından geçersiz sayıldı.
  - **Kaynak taraması 10 Kas**
    - (a) Kullanıcı terimleriyle ~110 kaynak, 31 tam metin: tarifin bütünü literatürde yok.
    - (b) Sektör terimleriyle 45 kaynak, 33 tam metin: eğitimsiz ve katman atlamasız en yakınlar ESP, Jacobi/Lookahead, PTD. Hiçbirinde 1 çekirdek ölçümü yok.
    - Yan bulgu: GGUF'ta motorun kullanmadığı eğitilmiş MTP bloğu var. `blk.24` tam katman (attn, ffn) artı `nextn.eh_proj / enorm / hnorm / shared_head_norm`; `nextn_predict_layers=1`. Motor 0–23'ü yükler (`qwen35.zig:113–114`).
- **Kayıtlar**
  - `docs/oturum-dokumu/` → 8–10 Kas döküm + analiz

## 🌀 11 Kas — Tarif (kullanıcının kendi terimleriyle)
- **1. İşlemci bir tarif makinesi**
  - Her hücre, her nokta %100 kullanılabilir; çekirdek bizi seçmez, biz onu seçeriz
  - İşlemci tahmin edemez; tarif uzayında senin istediğini yapar, o tahmin ettiğini sanır
  - Black box hacking gibi: istek at, gelen cevaba bak, ona göre işini yap; "mümkün değil" ayakları geç
  - Hiçbir şeyi tasarlandığı gibi kullanmam, her zaman kendi çıkarım için; "ya öyle değilse?"
- **2. Fırtına**
  - Komutan (Core 1): 3 emir, ortaya konuşur, koordine eder
  - Ordu (Core 2–3): kendi kimlikleri; kaos cep evrenlerini ortadaki emirden kendine çeker
  - Yeşil bulut / mikro tarif: atlı, takım taklavatı kontrol eder, buluta gönderir; kararana kadar içine alır, süblimleşir
  - Kara delik = fırtınanın gözü = dinginlik; mor = çevre, rüzgâr hızlanır, hava soğur, tam kaos
  - Bitince yerçekimiyle çakılma → düşüş **Infinity Fabric** → **IO die** (CCD→cIOD, şemayla doğrulandı)
- **3. Verim yalan, efficiency gerçek**
  - Ortak yok; devir daim, takım taklavat var; uyursan ölürsün; fırıncı gibi sürekli üretim
  - "L1 L2 L3'te çalışmazsak hızlı olamayız" yanlış; her nokta tasarım harikası
- **4. CPU'yu depolama birimi olarak kullanma**
  - Veri her daim CPU'da dolaşır, kan hücreleri / alyuvar gibi; sabit yerde durmaz, duramaz
  - İhtiyaç olunca ters kara delik açar, içine patlama yapar, bir anda veri gelir (bum)
  - RAM brute force; GPU daha zalim brute force; tembellerin işi
  - Ne reuse ne recompute: "önceden hesaplananı saklamak, her token için baştan hesaplamaktan verimli" varsayımına 180° karşı; tarifim buna bin basar
  - İlk ham hâli (Gemini'ye giden birleşik prompt): veri dondurulup saklanmaz, işlem hâlinde tutulur; sürekli matematik işlemden geçer; gönderim başarılı → sonlandıran sinyal; başarısız → ters işlem, orijinal veri geri
- **5. Veri hiçbir hücreye ait değil**
  - İşlemci bizzat Satürn, veriler halkası; işlemci kara delik, veriler içinde kara delik
  - CRT monitör gibi: içe vakumlayıp bomba etkisiyle tükürür; sonsuza kadar işlem
- **6. Dalga = kolektif bilinç = paralellik**
  - Dalga sistematikleştikçe paralel performans parabolik artar; 6–8 haneli kazanım
  - Sıradan paralelliğe karşı: çeyrek çekirdek, hatta çeyreğin çeyreği çekirdekle bu iş yapılır; paralellik gerçekten 100000 kat getirmiyorsa paralellik yapma
- **7. Zero security**
  - Bellek zaten güvenli değil; güvenli yapma takıntısı finansal korkudan; bende yok
  - Her şey ortada; patlatmak isteyen gelsin; gerekirse sıfırdan tuğla tuğla kurarız
  - "Mümkün mü" sormayız; "nasıl hile yaparım, hileli zarı kendi hileli zarımla oynarım" sorarız
- **8. Sinyal zayıflamaz; zayıflarsa kendi kendini şarj**
  - CPU küçük, mesafe kısa; uzun yol katetmiyoruz
  - Redstone yenileyici gibi, CPU'nun kendi parçalarından; 0.85 ise risk-getiri yapıp en yakın birime gider, charge olur
  - Reverse'e gerek yok: birimler 7/24 elektrik alır, geçerken çalar
  - Parçacık hızlandırıcı: veri kaos, her saniye hızlanır, kendi enerji koridorunu yaratır, speedforce, sonsuz döngü
  - Racetrack / manyetik bellek iş yaş: parazit var (kuantum düzeyinde bile hattın temiz olduğunu bilemezsin), uzun teller CPU'ya sığmaz, yan yana manyetizma birbirini etkiler; biz sıfırdan tasarlamayız, var olan CPU'da yaparız
  - "Bayrak eskiyince" diye kitabın ortasından konuşulmaz: önce ortam (pist güneş alıyor mu, ne kadar süre, bayrağın boyu, nasıl taşınıyor); değiştirme pit stop değil, cebinde yedek taşır (hakemin yedek kartı gibi)
- **9. Her daim operasyonel**
  - Kusursuzluk istemiyorum; her ne olursa olsun operasyonel kalmalı
  - Gerekirse 12 milyon tarifi içinde dondursun ama kararı kendi versin; özel durum seti yazmam (yazarsam forensic gerekir)
  - Bugünkü motorların derdi: sorun çıkınca PR/issue/tek tek destek ekleme
- **10. Determinizmden nefret — sezai yöntemi**
  - Çalışıyorsa formülüze etme, bırak çalışsın; sezgisel = aralıksız kaynak tüketip sezgiyi yakalamak
  - "Sat koy amına koy" = sezai yöntemi
  - X katmanı: israf et (keşif olabil); Y katmanı: israf etme (throughput düşmesin)
- **11. Ne statik ne dinamik — RDSM**
  - a/b/c/d değiliz; yeri gelir doğrusal, yeri gelir parabolik, yeri gelir asenkron (biri 10k biri 10)
  - Sonsuz senaryo odaklı; promptu gireni görev neyi gerektiriyorsa ona hizmet; boşa brute force yok
- **12. Hacker, araştırmacı değil**
  - Zero security çıkarım motoru; gerekirse zero day patlatır, 20 zafiyeti zincirler
  - Mikrocode/BIOS zafiyetleri dahil; kendinde + nday/rapor/exploit sömürür; amaç başarmak, paramız yok
- **13. Sabit başarı kuralı yok**
  - "Başarılı şu sayılır" diyemezsin; iterasyon lazım; doğru/yanlış yok; israf da değerli; bilinmeze gidiyoruz
- **14. Prefill / decode**
  - Prefill = kas gücü; decode = bant genişliği / iletişim hızı / veriye ulaşılabilirlik
  - Decode'u başaracağından emin; prefill için ayrı hileler var
- **15. Ölçek**
  - 800M'de çalışıyorum, sonra 3T'ye ölçekleyeceğim; model agnostik, parametre sayısı fark etmez
  - 3T modelin %1'ini kullanıp tam performans; işlem gücünün de %1'i
  - Quantization'a da karşıyım; o ayrı proje, şimdilik quantize modelle ilerliyorum
- **16. Çalışma biçimi**
  - Bana 10 yaşındakine anlatır gibi anlat; ben oradan karşı argüman üretirim, kendi tarifimi yaratırım
  - Hızlandırıcıda dönen şeyler yalnız veriden oluşmaz; kendini kısıtlama, "bu mu şu mu" deme, "neden bu olmaz" diye sor; herkesin en saçma bulduğu şey asıl çözüm olabilir
  - Konu basit, overengineering yok
  - Checkpoint al, kazanımları md'ye yaz, bir alan seç, dalı genişlet, bitir, sonra diğerine geç
  - Mindmap bir dizi değil; skill ve MCP
  - Kafamda 1000'den fazla tarif, 12 zincir (chain exploit gibi birbirini tetikler); fırtına, listemin 2. katının 12. maddesi
- **Not**
  - ChatGPT tarifin hiçbir parçasını 1:1 tekniğe çeviremedi; sürekli bildiği şeye (ring buffer, thread pool, akademi) çevirip yüzey değiştirdi
  - Kaynaklar baştan sona okundu: 1905 (329 s.; 1904 aynı konuşmanın dikey baskısı), 1207 (299 s.), `ryzen-5-3600-cpu-only-execution-checkpoint-2026-11-03.md` (ChatGPT Work çıktısı: tarifi ring buffer + checksum'a ve "kullanıcı alanından mümkün değil"e indirdi, kendi "Deney A"sını önerdi; tarif değil)
  - Bu dal tarifin kaydı; teknik karşılık ve ölçüm ayrı aşama

## 🌩 11 Kas — Fırtına keşfi ve token üreten yer
- **Tarif kaydı ve görsel**
  - Çizimin doğru okunuşu, anlamlandıramadıklarımız ve reddedilen eşleştirmeler: `docs/firtina/FIRTINA.md`
  - Hareketli görsel: `docs/firtina/firtina-teknik.html` (oda = CPU; iki simetrik fırtına hücresi; enerji; ters üçgen → kasırga → vakum → dalış; ters yön)
- **Odanın ölçülen haritası (belgesiz, bu işlemci)**
  - İki simetrik yarı: {cpu0,1,2,6,7,8} ve {cpu3,4,5,9,10,11}; a ile a+6 aynı yer
  - Ayrıntı: `echo-core-zig/benchmarks/kesif/c2c-2026-11-11.md`, `firtina-2026-11-11.md`
- **Eski yolun duvarı: modelin tamamının bir kez düşme tavanı** (`benchmarks/kesif/dusus-2026-11-11.md`)
  - | Düzen | tok/s tavanı | Kat Artışı |
  - | 1 çekirdek, motorun bugünkü yerleşimi (4 KB sayfa) | 28,2 | 1,00x |
  - | 1 çekirdek, 2 MB sayfa, vmovntdqa | 43,1 | 1,53x |
  - | Sol yarı (3 çekirdek) | 49,2 | 1,74x |
  - | Bütün oda (12 thread) | 44,7 | 1,59x |
  - Sonuç: her token bütün modeli taşıdıkça duvar ~49 tok/s; 4 hane taşımayı yok etmekle gelir
- **Token üreten yer: token_embd'den TR/EN/kod satırları CPU tarafında tek blok** (`ECHO_ODA_CIKIS=1`, varsayılan kapalı; `src/llama/oda.zig`; `benchmarks/kesif/oda-cikis-2026-11-11.md`)
  - Seçilen: 248.320 satırdan 140.330; 270,2 MB → 152,7 MB
  - | Çekirdek | Düzen | tg32 (tok/s) | tg128 (tok/s) | Kat Artışı |
  - | 1 | ham kapalı | 21,7 | 22,2 | 1,00x |
  - | 1 | ham açık | 29,6 | 28,1 | 1,36x / 1,27x |
  - | 1 | D kapalı | 37,1 | 39,4 | 1,00x |
  - | 1 | D açık | 37,1 | 39,1 | 1,00x / 0,99x |
  - | 2 | ham kapalı | 32,4 | 32,4 | 1,00x |
  - | 2 | ham açık | 40,2 | 38,7 | 1,24x / 1,19x |
  - | 2 | D kapalı | 52,2 | 50,4 | 1,00x |
  - | 2 | D açık | 53,0 | 54,4 | 1,02x / 1,08x (oynama içinde) |
  - Normal kullanımda (D) kazanç yok: IVF zaten çıkış tablosunun küçük bir kısmını okuyor
  - Doğruluk: D'de çıktı birebir; ham yolda 6 prompttan 1'i emojide ayrıştı; kalite ppl 17,72 → 17,63
  - Blokta emoji, €, ₺, oklar yok
- **Sıradaki**: duvar artık katmanlarda (~560 MB/token); 3. yol (model odada yaşar) için tarifin tekniği
