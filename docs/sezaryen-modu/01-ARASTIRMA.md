# Sezaryen Modu — Araştırma Haritası

Bu dosya, "A tarafı / B tarafı" tartışmasını ve literatürdeki gerçek sayıları sabitler.
Etiketler: **LİTERATÜR** (yayınlanmış), **ÖLÇÜLDÜ** (bu repoda), **HIPOTEZ** (henüz ölçülmedi).

---

## 1. Önce ekseni düzeltelim

Senin sorduğun ayrım:

- **A tarafı:** eğitimsiz, embedding uzayı.
- **B tarafı:** eğitimli, hidden state.

Bu ayrım pratikte temiz değil. Çünkü:

- ESP embedding'den girer ama hidden state'lerin dönüşümünden yararlanır.
- EAGLE hidden state kullanır ama token embedding'ini de girdiye ekler.
- PaSS embedding'i eğitir; yani "embedding = eğitimsiz" değildir.
- CLaSp hidden state'i eğitimsiz kullanır; yani "hidden state = eğitimli" değildir.
- Lyanna hidden state ile draft eder, token bilgisini örnekleme aşamasında enjekte eder; ikisini birleştirir.

Doğru eksen şu:

**Hangi belirsizliği ortadan kaldırıyorsun, hangi seri işi paralelleştiriyorsun, hangi işi hiç yaptırmıyorsun ve bunun karşılığında hangi garantiden vazgeçiyorsun?**

---

## 2. Yöntem tablosu

| Yöntem | Eksen | Eğitim | alfa | c | k | Bildirilen hız | Not |
|---|---|---|---|---|---|---|---|
| ESP | embedding mask | yok | orta | düşük | 1-2 | 1.22-1.48x | 2603.17942 |
| Lookahead | n-gram + Jacobi | yok | değişken | düşük | 3-5 | 1.8x; 4x çoklu GPU | 2402.02057 |
| PLD | n-gram | yok | değişken | ~0 | 10 | 2.4x | özetleme/QA |
| REST | retrieval | yok | değişken | düşük | 4-8 | 1.62-2.36x | 2311.08252 |
| ANPD | n-gram | yok | değişken | düşük | 8 | 3.67x | 2404.08698 |
| SoT | orkestrasyon | yok | N/A | N/A | segment | ~2x | token seviyesi değil |
| FastCoT | paralel bağlam | yok | N/A | N/A | - | ~%20 | 2311.08263 |
| Hidden Transfer | hidden state | projeksiyon | yüksek | orta | 3 | 1.4-2.3x | 2404.12022 |
| FIRP | hidden state | lineer | yüksek | orta | 3 | 1.9-3x | 2410.20488 |
| EAGLE / EAGLE-3 | hidden + token | head | 0.6-0.8 | orta | 3-5 | 2-3x | 2401.15077 / 2503.01840 |
| Medusa | hidden | head'ler | 0.5-0.7 | orta | 3-5 | 2-3x | 2401.10774 |
| Lyanna | hidden + token-info | 1 katman | yüksek | orta | 3 | 3.3x (SD'ye karşı) | 2602.21224 |
| DFlash | hidden bağlam + blok difüzyon | eğitimli | yüksek | orta | blok | 6.1x | 2602.06036 |
| Falcon | yarı-AR drafter | eğitimli | yüksek | orta | 3-5 | 2.91-3.51x | 2412.12639 |
| ParallelSpec | paralel drafter | eğitimli | yüksek | orta | 3-5 | 2.84x | 2410.05589 |
| Hybrid Spark | model + model-free | karışık | adaptif | adaptif | adaptif | model-based'e göre 1.10-1.62x | 2611.05611 |
| MetaSD | çoklu drafter | karışık | adaptif | adaptif | adaptif | tek drafter'ı geçer | 2604.05417 |
| PaSS | embedding | eğitimli | yüksek | düşük | 1-3 | - | 2311.13581 |
| CLaSp | hidden state | yok | - | düşük | adaptif | - | ACL 2025 |
| Token Recycling | aday geri dönüşümü | yok | - | düşük | - | - | ACL 2025 |
| CopyLoop | looped model | özel | yüksek | düşük | uzun | 2.98-10.70x | 2610.38822 |
| llguidance fast-forward | grammar | yok | 1 (kesin bölge) | ~0 | blok | kısıtlı işlerde yüksek | LLM değil, kısıt motoru |
| CLLMs | Jacobi + eğitim | eğitimli | yüksek | düşük | - | 2.4-3.4x | 2403.00835 |
| Diffusion LLM | mimari | eğitimli | - | - | - | değişken | genel değil |

---

## 3. Neden hepsi 1-2x bandında?

MLSys 2026 denklemi:

E = (1 - alfa^(k+1)) / ((1 - alfa) * (k * c + 1))

Bu denklem iki şeyi söylüyor:

1. **Taslak kalitesi (alfa) ve taslak maliyeti (c) birbirine bağlı.** Daha iyi taslak daha pahalı; daha ucuz taslak daha zayıf.
2. **Hedef doğrulama maliyeti baskın.** Taslak ne kadar iyi olursa olsun, hedef modelin forward'ı ödeniyor.

Bu yüzden:

- Eğitimsiz n-gram/embedding yöntemleri c'yi küçültür, alfa'yı düşürür.
- Eğitimli hidden-state yöntemleri alfa'yı yükseltir, c'yi yükseltir.
- İkisi de aynı Pareto eğrisine oturur.
- Görev farkı büyüktür: kod düzenleme/özetleme gibi kopya ağırlıklı işlerde n-gram 3-4x'e çıkar; serbest sohbette 1.2-1.8x'e düşer.

**Ölçüldü (bu repoda):** Qwen3.5-0.8B Q8_0, CPU, t=6, batch B=1 -> 55 tok/s; B=256 -> ~300 tok/s. Yani hedef forward'ı batch ile hızlandırmak mümkün, ama 10x değil; ~5.4x.

---

## 4. "C yok" değil, C var

Senin "ikisinin iyi tarafını birleştiren C yok mu?" sorunun cevabı: var.

### Lyanna (2602.21224)
- Hidden state seviyesinde autoregressive draft.
- Token bilgisini örnekleme aşamasında enjekte eder.
- Reddedilen taslakların hidden state'lerini yeniden kullanır.
- Standart SD'ye göre 3.3x, en iyi baseline'a göre 1.4x.
- Eğitim: 1 transformer katmanı, 68K örnek, 20 epoch.
- Ana model ağırlığı/quantizasyonu değişmez.

### EAGLE / EAGLE-3
- Hidden state + token embedding birlikte.
- EAGLE-3, "gelecekteki hidden state'i birebir taklit etme" hedefini bırakıp doğrudan token tahminine yönelir.
- 2-3x bandı.

### DFlash (2602.06036)
- Hedef modelin çeşitli katmanlarından özellikleri alıp küçük bir blok-difüzyon drafter'a KV olarak verir.
- Bir bloğu paralel üretir, hedef model doğrular.
- Qwen3-8B'de 6.1x'e kadar.

### Hybrid Spark (2611.05611)
- Aynı anda model-based ve model-free (n-gram) drafter çalıştırır.
- Her istek için hangi zincirin daha iyi olduğunu fiyatlandırıp seçer.
- Model-based'e göre 1.10-1.62x ek kazanç.

### MetaSD (2604.05417)
- Birden fazla drafter'ı bandit problemi olarak yönetir.
- Tek drafter'ı geçer.

### PaSS / CLaSp / Token Recycling
- PaSS: embedding'i eğitir.
- CLaSp: hidden state'i eğitimsiz kullanır.
- Token Recycling: reddedilen adayları geri dönüştürür.

Yani "A ya da B" değil; **A + B + dış kesinlik** yönünde ilerliyor. C var, ama C'nin de tavanı aynı denklem.

---

## 5. "Doğru mühimmat" hipotezi

Senin ikinci yönün ikiye ayrılıyor:

### 5.1 Hedefin üreteceği tokenı baştan bilmek
Bu, alfa'nın 1'e gitmesi demek. Denklem:

E ~= (k + 1) / (k * c + 1)

- c ~= 0 ise E ~= k + 1.
- 10x için k ~= 9 ve c ~= 0 gerekir.
- Ama hedef modelin k+1 tokenı işleme maliyeti "bir token" kadar değil. GPU'da küçük batch'te neredeyse öyle; CPU'da değil.

MLSys 2026 simülatörü alfa = 1 ideal durumunu test ediyor: doğrulama maliyeti baskın; adaptif kombinasyonla üretimde 4.9x tavan. Yani "hepsi doğru gelse bile" 10x değil, ~5x.

### 5.2 Devamın dışarıdan kesin olduğu yerler
Burada "doğru mühimmat" gerçekten bedava:

- **Kod düzenleme / özetleme / retrieval:** n-gram, PLD, REST, ANPD. Prompt'ta/geçmişte aynı dizi varsa gelecek tokenlar zaten biliniyor. PLD 2.4x, REST 1.62-2.36x, ANPD 3.67x.
- **Grammar/format kısıtları:** JSON şeması, zorunlu anahtar, ayraç. llguidance'ın fast-forward mekanizması, zorunlu metni token token modele seçtirmeden blok halinde ekler. Doğruluk tahminden değil, kısıttan gelir.
- **CopyLoop:** looped-Transformer base modellerde bağlam kopyalama + sığ eleme + örtüşen yürütme ile 2.98-10.70x. Ama özel model ailesi; genel Transformer'a taşınmış değil.

Yani 10x, "her modelde her görevde" değil; **devamın zaten belirli olduğu bölgelerde** mümkün. Senin "agresif hacker" içgüdün tam buraya denk geliyor: modelin seçim yapmasına gerek olmayan yeri modelden çıkar.

---

## 6. Neden "eğitim" tek başına cevap değil?

Eğitim, alfa'yı yükseltir ama c'yi de yükseltir. Eğitimli bir drafter:

- Daha iyi tahmin eder (alfa yukarı).
- Daha fazla parametre, daha fazla forward, daha fazla bakım (c yukarı).
- Hedef modele bağımlıdır; model değişince yeniden eğitim gerekir.

Eğitimsiz bir drafter:

- Bedava ve taşınabilir (c düşük).
- Daha zayıf (alfa düşük).
- Model değişse de çalışır.

Senin "eğitme saçma" sezgin, c tarafından bakınca doğru; ama alfa tarafından bakınca eksik. En iyi nokta genellikle **küçük, hedefe özel bir adaptör + eğitimsiz hızlı bir yol** kombinasyonu. Lyanna tam bu.

---

## 7. 10x nerede mümkün?

Üç rejim var:

1. **Özel model ailesi:** CopyLoop gibi looped-Transformer'larda 2.98-10.70x. Genel modelde yok.
2. **Kısıtlı çıktı:** grammar/JSON/kod. llguidance fast-forward ile seçim işi kalkar; 5-10x görülebilir.
3. **Küçük batch + GPU + bellek-bağımlı:** hedef forward'ı k+1 tokenı neredeyse 1 token maliyetinde işler; DFlash 6.1x, MLSys adaptif 4.9x. CPU'da bu rejim yok.

Senin CPU'da 10x hedefi, bu üç rejimin hiçbirine tam oturmuyor. Ama "10x her yerde olmaz" demek, "10x hiç olmaz" demek değil. Doğru soru: **hangi rejimdeyiz, hangi garantiyi bırakıyoruz?**

---

## 8. Echo Core için çıkarım

- **Kol 1 (eşdeğerlik):** PLD/REST -> ESP -> Lyanna-benzeri hidden-state reuse -> DFlash-benzeri blok drafter. Hedef 2-4x, iyi senaryoda 6x.
- **Kol 2 (görev doğruluğu):** grammar fast-forward + retrieval/copy + kod bağlamı. Hedef kısıtlı alanlarda 5-10x.
- **Ortak:** seçici doğrulama, adaptif k, çoklu drafter scheduler, cache/position yönetimi.
- **Kırmızı çizgi:** model dosyası/quantizasyon değişmez; adaptör motorun içinde.

---

## 9. Kaynaklar

Tam liste: [04-KAYNAKLAR.md](04-KAYNAKLAR.md).
