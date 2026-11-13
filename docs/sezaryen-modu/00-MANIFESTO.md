# Sezaryen Modu — Manifesto ve Duvar

> Bu belge bir ders notu değil. Bir tez sabitleme aracı.
> Burada "doğru tasarım" iddiası yok. "Bize göre" var.
> Tez sahibi: kullanıcı. Kayıt: Echo Core.

---

## 1. Tez, tek paragraf

Autoregressive model tokenı üretir, tokenı geri yer, sonraki tokenı üretir. Zincir büyüdükçe darboğaz büyür. Kullanıcının tezi şu: **modelin kendi kurallarına uymak zorunda değiliz.** Modelin karnını sezaryenle açalım, üretim boyunca açık tutalım, dışarıda çok daha hızlı token/özellik üreten bir "takyon çarpıştırıcısı" çalışsın ve bu üretilenleri doğrudan iç katmanlara püskürtelim. Model dosyasına, mimarisine, quantizasyonuna dokunmadan. Bunu modele özel değil, **motor seviyesinde** yapalım; Echo Core bütün modellere aynı tarifi uygulasın.

Bu tez ikiye ayrılıyor:

1. **Eşdeğerlik korunacak:** Hedef modelin üreteceği tokenı bozmadan, sadece daha az forward ile aynı çıktıyı almak. Bu, spekülatif decoding'in lossless dünyası.
2. **Görev doğruluğu yeterli olacak:** Hedefin birebir dağılımı değil, işin doğru olması önemli. Kod testi geçsin, JSON şeması tutsun, hesap doğru olsun. Bu, "aynı çıktı" garantisinden vazgeçip daha agresif hızlanma arayan dünya.

Bu iki dünya aynı belgeye sığmaz; ayrı ölçülür, ayrı etiketlenir. Karıştırmak, kanıtı bulanıklaştırır.

---

## 2. Değiştirilemez kırmızı çizgiler

- **Model mimarisi değişmez.** Model dosyası, katman yapısı, attention düzeni, tokenizer aynı kalır.
- **Quantizasyon değişmez.** Q8_0 dosyası Q8_0 kalır; ağırlık yeniden yazılmaz.
- **Manuel model ameliyatı yok.** Kullanıcı modelin içine girip elle katman değiştirmez.
- **Motor modele özel olabilir, ama model dosyasına dokunmaz.** Gerekiyorsa uyarlama Echo Core'un içinde, modelin dışında durur. Bu, "model-agnostik motor" ile "tek tensor reçetesi" arasındaki farktır.
- **Kayıpsız mod ile görev-doğruluğu modu ayrı etiketlenir.** "Lossless" yazıyorsa hedef dağılım korunuyor demektir; "task" yazıyorsa hedef dağılımdan sapma serbesttir.

---

## 3. Duvar: MLSys hızlanma denklemi

MLSys 2026, *Speculative Decoding: Performance or Illusion?* (Liu, Yu, Park, Stoica, Cheung) şu denklemi veriyor:

E(speedup) = (1 - alfa^(k+1)) / ((1 - alfa) * (k * c + 1))

- **alfa** = taslak tokenların kabul oranı
- **k** = önerilen taslak uzunluğu
- **c** = taslak üretiminin hedef modelin tek forward'ına göre maliyet oranı

Bu denklem "kanun" değil; **varsayımları olan bir model.** Duvar dediğimiz şey denklemin kendisi değil, arkasındaki varsayımlar:

1. Doğrulamayı hedef model yapmak zorunda.
2. Hedef modelin k+1 tokenı işlemesi, bir tokenı işlemesiyle aynı maliyette.
3. Taslak maliyeti c, kabul oranından bağımsız.
4. Kabul oranı alfa, pozisyondan ve içerikten bağımsız, i.i.d.
5. Çıktının hedef dağılımla birebir aynı olması gerekiyor.

Bu beş varsayımdan herhangi birini kırmak, denklemin dışına çıkmak demektir. Kullanıcının "hile" dediği şey tam olarak bu: **denklemi daha iyi optimize etmek değil, denklemin kurulduğu zemini değiştirmek.**

---

## 4. Hacker duruşu

"Yapılamaz" cümlesi, genellikle "mevcut varsayımlar altında yapılamaz" demektir. Bu belge, "yapılamaz"ı duvar ilan etmiyor; **hangi varsayımın kırılabileceğini** arıyor.

- Eğer herkes "model-agnostik tensor injection olmaz" diyorsa, bu bir varsayımdır; adaptör motorun içine taşınarak kırılabilir.
- Eğer herkes "doğrulama şart" diyorsa, bu bir varsayımdır; devamın dışarıdan kesin olduğu bölgelerde doğrulama seçimi kaldırılabilir.
- Eğer herkes "alfa i.i.d." diyorsa, bu bir varsayımdır; kopya/retrieval/grammar bölgelerinde alfa yapısal olarak 1'e yaklaşır.
- Eğer herkes "hedef dağılım korunmalı" diyorsa, bu bir varsayımdır; görev doğruluğu modunda bu şart kaldırılabilir.

Bu, "her şey mümkün" demek değil. Bu, **duvarın nerede olduğunu yanlış yerde aramamak** demek.

---

## 5. Ne iddia ediyoruz, ne iddia etmiyoruz

**İddia ediyoruz:**
- A/B (embedding vs hidden state) ayrımı temel eksen değil; aynı hattın iki kesiti.
- Hibritler zaten var: Lyanna, EAGLE, DFlash, Hybrid Spark, MetaSD.
- 1-2x bandı, aynı Pareto eğrisinin sonucu; uygulamaların "leş" olması değil.
- "Doğru mühimmat" (alfa'yı yükseltmek) en az doğrulama kadar kritik.
- 4-6x, seçilmiş yöntemlerle bu repoda denenebilir.
- 10x, her modelde değil; devamın dışarıdan kesin olduğu bölgelerde veya özel model ailelerinde mümkün.

**İddia etmiyoruz:**
- "10x her modelde, her görevde, kayıpsız olarak alınır."
- "Model-agnostik tensor reçetesi tek satırda çalışır."
- "Eğitim tamamen gereksizdir." (Bazı yollar eğitimsiz, bazıları küçük bir adaptörle.)
- "Bu belge hakem onaylıdır." Bu bir tez sabitleme ve deney planı belgesidir.

---

## 6. İki kollu strateji

### Kol 1 — Eşdeğerlik korunacak
Hedef modelin çıktısı aynı kalacak. Araçlar: PLD/REST/ANPD, ESP, Lookahead, Lyanna-benzeri hidden-state reuse, DFlash-benzeri blok drafter, seçici doğrulama. Hedef: 2-4x; iyi senaryoda 6x.

### Kol 2 — Görev doğruluğu yeterli olacak
Hedef dağılım birebir korunmayacak; iş doğru olacak. Araçlar: grammar fast-forward, JSON şema zorlaması, kod bağlamından kopyalama, retrieval. Hedef: kısıtlı alanlarda 5-10x; serbest metinde değil.

Bu iki kol aynı denklemi kullanmaz. Kol 1'de alfa, c, k ölçülür. Kol 2'de görev başarısı, hata maliyeti ve gecikme ölçülür. Etiketler karışmaz.

---

## 7. Bu belge neden var?

Çünkü fikirler uçuyor. Bu dosya, uçan fikirlerin **sabitlenmiş hali**. Her yeni deney, her yeni iddia buradaki etiketlerden birine bağlanacak:

- **HIPOTEZ** — henüz ölçülmedi.
- **ÖLÇÜLDÜ** — bu repoda koşuldu, sayı var.
- **LİTERATÜR** — başka yerde yayınlandı, link var.
- **ÇÜRÜDÜ** — denendi, kazanç yok.
- **AÇIK** — denenmedi, sırada.

Bir cümle bu etiketlerden birini taşımıyorsa, kanıt değildir.

---

## 8. Yol haritası

- [01-ARASTIRMA.md](01-ARASTIRMA.md) — literatür, A/B, hibritler, sayılar.
- [02-FORMUL-VE-HACK.md](02-FORMUL-VE-HACK.md) — MLSys denklemi, varsayım kırma, hile kataloğu.
- [03-DENEY-VE-MIMARI.md](03-DENEY-VE-MIMARI.md) — Echo Core mimarisi, deney fazları, metrikler.
- [04-KAYNAKLAR.md](04-KAYNAKLAR.md) — linkler ve tek satırlık notlar.
- [../../q8-tg-yontem-haritasi.mindmap.html](../../q8-tg-yontem-haritasi.mindmap.html) — ana harita; Sezaryen Modu dalı içinde.
