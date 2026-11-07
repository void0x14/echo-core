# İşleyiş Araştırması — log

Kural: sorgu = kullanıcının kendi cümleleri, 1:1. Yorum yok, eşleme yok, doğru/yanlış etiketi yok. Her sorgu ve dönen her şey loglanır.

## Kaynak metin (kullanıcı, 2026-11-07)

P1. market llm'in kendisidir, markete gitmek llm'in kendisine gitmek demektir, nasıl gideceğin sana kalmış, ister arabayla git ister yürüyerek ister takla atarak sonsuz olasılık var.
P2. insanoğlu her ürün için markete ayrı ayrı gitmez; bir listeyle 8 ürünü tek seferde de almaz, işini tek seferde halleder. alacağı 20 ürün varsa alır pazar arabasını ve markete gider hepsini tek seferde alır pazar arabasına yükler taşır götürür. ne kadar ürün alacaksak hepsini taşıyabiliyoruz pazar arabamızla.
P3. naparsan yap aldığın ürünlerin fiyatını her zaman ödersin zaten.
P4. kasa olmayan market yok; ya fiziksel kasiyerli kasalar ya dijital self-kasalar vardır. kasa olmazsa aldıklarının parasını ödeyemezsin. fundamental bir şeyi atlayamazsın. llm dediğin şey saf fiziktir, saf gerçektir.
P5. ödeme esnasında kartla ödeyip kartın onaylanmasını, işlem süresini beklemekle uğraşmam; alacaklarımın ne olduğu bellidir, fiyatları da bellidir; nakit paramı çıkartırım, kasiyere tak diye veririm, ürünlerimi pazar arabama koyar tak diye yol alırım. zaman kaybetmem.
P6. önceden markete gitmişsindir, gördüğün fiyatları hatırlarsın. ürün en son 10 TL idi, şimdi de 10 TL'dir ama yanıma ekstra 200 TL alırım ne olur ne olmaz. gidersin, ürün 20 TL olmuş: "+10 TL fazla vereceğim" diye kafana atar hızlıca ilerlersin. kasada 20 TL'yi önceden hazır edersin, "AL KARDEŞ" dersin. hatırladığın fiyat tutmasa da ürünü alırsın; doğru hatırlayana kadar beklemezsin.
P7. alfabede 29 harf var; onlarla milyarlarca kelime üretebilirim; günlük hayatta yüzden fazla kelime bile kullanmıyorumdur. modelin dili bilip bilmediğini, hangi dili bildiğini bilemezsin.
P8. listeyi doğru yazarım. eksik/yanlış yazdıysam ürünü isteyeni telefonla ararım (garanti artar; açmayabilir, beklersin, gecikme olur) ya da umursamam (dırdırını çekersin, kulağın duyar, dayanırsın).

## Sorgu logu

### S1 — P1+P2 birebir (Türkçe), exa neural, 10 sonuç
Dönen: market alışverişi sayfaları (iFood alışveriş skill'i, memurlar.net "markete pazar arabası ile gitmek", Seedance video promptu, M. Serdar Kuzuloğlu "akıllı market alışverişinin sırrı", HBR Türkiye "Pazarda, markette, sokakta", İzmir'e Dair "bir market arabası bize ne anlatır", soturkish semt pazarları, market-price-finder-mcp-tr, Listen alışveriş uygulaması, Peky kasasız market).
Veri: teknik (LLM çalıştırma) sonucu 0/10. Kasasız market sonucu 1/10 (Peky: kamera ile ürün algılama, kasaya girmeden çıkış).

### S2 — P1+P2 birebir (İngilizce çeviri), exa neural, 10 sonuç
Dönen: LLM ile ticaret yazıları (Dan Hockenmaier "LLMs vs. Marketplaces", M11 OpenMarket, Soren Larson "Autonomous Commerce", "The End of Marketplaces", Ian Jindal "World View AI", Mesa LLM pazar ajanları, MIT Kasbah "Marketplace Experiences", Google Cloud "Architecting Agentic Commerce", "LLMs as Retail Cart Assistants", John Roa "The Last Interface").
Veri: LLM çalıştırma tekniği 0/10. Hepsi LLM'in alışveriş yapması üzerine.

### S3 — P6 birebir (İngilizce), 10 sonuç
Dönen: tüketici fiyat hafızası çalışmaları.
- Kellogg, Chen/Iyer/Pazgal "When Memory Serves" (2011): müşteriler düşük fiyatları daha iyi hatırlar; firmalar hafıza sınırına göre fiyat koyar.
- Monroe/Lee "Remembering vs. Knowing" (1999): alıcılar yeni aldıkları ürünlerin fiyatını çoğu zaman hatırlayamıyor.
- Wadi/Ma "Shopping by algorithm" (2026, arXiv 2609.28372): LLM ajanları maliyet altında birim fiyat için gereken bilgiyi atlıyor.
- Leng "Can LLMs Mimic Human-Like Mental Accounting".
- **Bordalo/Gennaioli/Shleifer "Memory, Attention, and Choice" (2020):** hatırlanan geçmiş deneyimler bir norm (çapa) oluşturur; norma göre fiyat/kalite sürprizi büyük düzeltmeye yol açar. P6'daki "hatırlanan fiyat + sapmayı yerinde düzeltme" yapısının birebir modeli.
- Anderson/Simester "Price Cues and Customer Price Knowledge" (2008), Ofir vd. "Memory-Based Store Price Judgments" (2008), Jensen/Grunert "Price Knowledge During Grocery Shopping" (2014): fiyat bilgisi mağaza öncesi/sırası/sonrası ölçülmüş.
- Danziger/Hadar/Morwitz "Retailer Pricing Strategy and Consumer Choice under Price Uncertainty".
- Chen/Iyer/Pazgal "Limited Memory, Categorization, and Competition" (2010).
Veri: LLM çalıştırma tekniği 0/10. İnsan davranış modeli 10/10.

### S4 — P4+P5 birebir (İngilizce), 10 sonuç
Dönen: ödeme/kasa yazıları (Stripe yüz yüze ödeme yöntemleri, AWS Just Walk Out, North "invisible payments", self-checkout rehberi, MIT Kasbah, "The LLM Narrates, the Core Decides: Anatomy of an AI Checkout", Amazon Go "How to Stop Waiting and Kill the Credit Card", SOLUM kasasız mağaza, nekuda ajan ödemeleri, GOAT "One Checkout for AI Agents and Humans").
Veri: LLM çalıştırma tekniği 0/10. "Kasa var ama bekleme yok" yapısı: Just Walk Out / Amazon Go / invisible payments (ödeme ayrı adım değil, çıkışta arka planda kesinleşir).

### S5 — P7 birebir (İngilizce), 10 sonuç
Dönen: LLM'lerin dil bilgisi ölçümleri.
- "Lost in the Tower of Babel" (2605.01224): LLM'ler çok dilli görünüyor ama bu, büyük veriyle gelen tesadüfi çok dillilik.
- Zhang vd. EMNLP 2023 "Don't Trust ChatGPT when your Question is not in English".
- **Wendler vd. "Do Llamas Work in English?" (2402.10588):** çok dilli modeller içeride İngilizceyi ara dil olarak kullanıyor mu sorusu.
- Marchisio vd. "Language Confusion in LLMs" (2406.20052): modeller istenen dilde tutarlı üretemiyor.
- "Multilingual LLMs Are Not (Yet) Code-Switchers" (2305.14235), "ChatGPT Beyond English" (2304.05613), Nature podcast.
- **Conde vd. "Open Conversational LLMs do not know most Spanish words" (2403.15491):** açık modeller sözlükteki İspanyolca kelimelerin önemli kısmını bilmiyor/kullanamıyor.
- Jiang vd. TACL 2020 "How Can We Know What Language Models Know?".
Veri: P7'deki "modelin hangi dili bildiğini bilemezsin" ölçülmüş ve ölçülebilir bir soru olarak literatürde var.

### S6 — P8 birebir (İngilizce), 10 sonuç
Dönen: eksik bilgi/takip sorusu ve ajan tasarımı yazıları (kikomono "eksik bilgiyi sağlayabilecek kişiye sor", promptlabster "LLM bir şeyleri atlıyor", Amisha Gupta "No Promises I Can't Take Back" telefon ajanları, gecikme kurtarma ajanı, sistem prompt şablonları).
Veri: LLM çalıştırma tekniği 0/10. "Sor ya da geç" seçimi ajan/prompt düzeyinde tartışılmış.

### S7 — P2+P3 birebir (İngilizce), 10 sonuç
Dönen: ChatGPT içinde Walmart alışverişi, Stanford "One-Stop Shopping" (2017), Pepesto "Agent to Cart", ComboShoppingBench (2608.09282), Queek, Gentika "The Product Is Not Always the Unit of Demand", Walmart Instant Checkout başarısızlığı, ajanlar arası pazarlık.
Veri: LLM çalıştırma tekniği 0/10. Stanford "tek durakta alışveriş" P2 yapısının ekonomi tarafı.

## Ara sayım (S1–S7)

- 70 sonuç; LLM çalıştırma (inference) tekniği döndüren: 0.
- Birebir sorgu, kelimelerin günlük anlamına gidiyor: alışveriş, fiyat hafızası, ödeme, çok dillilik.
- Yapısal olarak P6'ya en yakın dönen: Bordalo/Gennaioli/Shleifer (hafıza normu + sürpriz düzeltmesi). P4/P5'e en yakın: Just Walk Out / invisible payments.

## Aşama 3 — hipotezler (tur 1)

Şablon: AKIŞ / VARSAYIM / ÇÜRÜTME / SONUÇ. Çürütme ölçütü yalnız P1–P8 cümleleri. Mekanizma bilgisi kaynakların özetinden.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H1 | Spekülatif çözüm (Leviathan 2023), bizdeki H01 | P1–P8'in tekniği | Yanlış tahminde kuyruğu atıp yeniden hesaplıyor mu? Evet. P2 ("tek seferde halleder") ve P6 ("doğru hatırlayana kadar beklemezsin") ile çelişir. | öldü |
| H2 | Jacobi çözümü | aynı | Sabit noktaya kadar tekrar eden tur; özette "tur başına nadiren 1 tokenden fazla". P2 ile çelişir. | öldü |
| H3 | Lookahead (2402.02057) | aynı | Jacobi turları + n-gram havuzu + doğrulama; çok tur. P2 ile çelişir. | öldü |
| H4 | CLLM / Jacobi Forcing | aynı | Modeli yeniden eğitiyor; hâlâ çok tur. P2 ile çelişir; ayrıca model dosyası değişiyor. | öldü |
| H5 | Difüzyon LLM (Fast-dLLM, LLaDA) | aynı | Başka model; birden çok gürültü giderme turu. P2 ile çelişir. ParallelBench: paralel çözümde kalite düşüyor. | öldü |
| H6 | DEER / ParaRNN | aynı | Girdileri bilinen bir diziyi Newton turlarıyla paralel çözüyor; çok tur. P2 ile çelişir. Not: Lyapunov üssünün işareti yakınsama hızını belirliyor (veri). | öldü |
| H7 | Aktivasyon seyreltme (TEAL) | aynı | Küçük aktivasyonun işini hiç yapmıyor. P3 ("fiyatını her zaman ödersin") ile çelişir. | öldü |
| H8 | RFVP / Load Value Approximation | P6 | Tahmin edilen değer kullanılıyor, gerçek değer hiç alınmıyor (istek düşürülüyor). P6'da gerçek fiyat ödeniyor, ürün alınıyor. P6 ile çelişir. P8'in "umursama" koluyla örtüşür. | P6 için öldü; P8-umursama için yaşadı |
| H9 | Bordalo/Gennaioli/Shleifer hafıza normu | P6 | Hatırlanan fiyat çapa, sapma yerinde düzeltiliyor, alışveriş durmuyor. P6 ile çelişki yok. Ancak LLM çalıştırma tekniği değil, insan davranış modeli. | yapı olarak yaşadı; teknik değil |
| H10 | Just Walk Out / invisible payments | P4+P5 | Kasa var (dijital kesinleştirme), müşteri beklemiyor. P4 ve P5 ile çelişki yok. LLM tekniği değil. | yapı olarak yaşadı; teknik değil |

### Tur 1 sonucu
- (a) Kanıtlanan: Bulunan yedi LLM tekniğinin (H1–H7) hepsi P2'de (tek sefer) ya da P3'te (fiyat her zaman ödenir) ölüyor. Yaşayanlar (H9, H10) yalnız insan/market yapısı. H8 yalnız P8-umursama kolunda yaşıyor.
- (b) Çürütülen: "Bilinen paralel/spekülatif çözüm ailesi P1–P8'in tekniğidir" → öldü.
- (c) Sıradaki tek adım: P2'yi çürütmeden geçen bir teknik bulmak için P2 cümlesini birebir (Türkçe ve İngilizce) akademik kaynakta aramak.

## Tur 2 — P2'yi geçen teknik araması (P2 birebir)

### S8 — P2 birebir Türkçe, arXiv, 10 sonuç
Dönen: Alzheimer OCT taraması, TEK modeli (fizik), TEK gradient flow, "Focusing Neuron", CTA ICRC2013, programlama dersi öz-yeterlik, uyarlanır evrişim çekirdeği, kan filmi renk normalizasyonu, mKdV yüzeyleri, Kerr-Schild çift kopya.
Veri: arXiv araması kelime eşleşmesiyle çalışıyor; "tek" kelimesi yazar/model adı "Tek" ile eşleşti. Konuyla ilgili 0/10.

### S9 — P2 birebir İngilizce, arXiv, 10 sonuç
Dönen: CMS/LHCb B_s→μμ, ATLAS performans, IceCube+LIGO, GWTC-4.0 (2 adet), KAGRA karanlık madde, LIGO sürekli dalga aramaları (3 adet), GEO600 FRB.
Veri: konuyla ilgili 0/10; uzun cümle arXiv'de anlam değil kelime yığını olarak eşleşiyor.

### S10 — P2 + "The market is the LLM itself" birebir İngilizce, exa research paper, 10 sonuç
Dönen: O'Leary "Large Language Model Commerce" (2026), ShopX (2606.31693), LLM sepet asistanı, Wadi/Ma "Shopping by algorithm", **Arciuolo/Abuzneid "Simultaneously Shop, Bag, and Checkout (2SBC-Cart)" (CSCI 2019)**, Basu "Grocery Retail through LLMs", ShoppingComp (2511.22978), Agentic Commerce survey, **Cochoy "shopping cart arithmetic" (Marketing Theory 2008, 323 atıf)**, Shopping Companion (2603.14864).
Veri:
- 2SBC-Cart: alışveriş, poşetleme ve kasayı tek birleşik eyleme topluyor; ürün arabaya bir kez giriyor, tekrar elleme yok, kasa ayrı adım değil. P2+P4+P5 yapısı.
- Cochoy: pazar arabası bütçe kısıtını hacim kısıtına çeviriyor; liste + raf bilgisi ile hesap yapılıyor. P2+P6 yapısı.
- LLM çalıştırma tekniği 0/10.

### Tur 2 sonucu (K2: blocker + denenmiş 3 yol)
- (a) Kanıtlanan: P2 birebir sorgusu üç yoldan (arXiv TR, arXiv EN, exa araştırma EN) LLM çalıştırma tekniği döndürmüyor: 0/30. Yapı olarak P2'ye uyan iki kaynak bulundu: 2SBC-Cart ve Cochoy. İkisi de market/tüketici çalışması.
- (b) Çürütülen: "P2 cümlesi birebir arandığında akademik kaynakta onu karşılayan bir LLM tekniği çıkar" → öldü (0/30).
- Blocker: arama motorları birebir cümleyi günlük anlamıyla (alışveriş) eşliyor. Kaynak: S1–S10, 100 sonuç, LLM tekniği 0.
- (c) Sıradaki tek adım: kullanıcı kararı. Kural (1:1, çeviri/yorum yok) korunursa arama yolu tükendi; aynı cümlelerle başka kaynak (GitHub, tez veritabanı) denenebilir.

### S11 — P2 + "The market is the LLM itself" birebir İngilizce, exa GitHub, 10 sonuç
Dönen: browser-use buy_groceries.py ("You will buy all of the items at the same store"), balakreshnan genaishoppingcart, chat2cart, agentic-commerce, Google AP2 shopper ajanı, browser-use shopping.py (Migros), e2llm-sifr Costco 47 ürünlük sepet, CartAI (Llama ile el yazısı listeden sipariş), ShopEZ AI Agent, UARK_AutoFillCart.
Veri:
- 10/10 sonuç "LLM markette alışveriş yapan ajan". Market LLM değil, LLM müşteri. P1'in tersi.
- browser-use: "tüm ürünler aynı mağazadan" → P2'nin günlük anlamı, ama ürünler yine tek tek aranıp sepete ekleniyor.
- LLM çalıştırma tekniği 0/10.

### Tur 3 sonucu
- (a) Kanıtlanan: dört kaynak (arXiv TR, arXiv EN, exa araştırma, exa GitHub), 40 sonuç, LLM çalıştırma tekniği 0/40.
- (b) Çürütülen: "GitHub'da kod olarak aranınca P2'yi karşılayan teknik çıkar" → öldü. GitHub "market + LLM" cümlesini LLM'i müşteri yapan ajanlarla eşliyor.
- (c) Sıradaki tek adım: P6 cümlesini (hatırlanan fiyat + 200 TL, 20 TL çıkarsa +10 kafaya yaz, ilerle) birebir aynı GitHub kaynağında aramak.

### S12 — P6 birebir İngilizce, exa GitHub, 10 sonuç
Dönen: bounswe market fiyat karşılaştırma senaryosu (İzmir), CubeZero (deterministik bütçe çekirdeği + LLM gerekçe), AlisverisListesiVeButceYonetimi (C, bütçe), odemeyecezzzzz (Getir Çarşı sepet karşılaştırıcı, 2 kayıt), fiyat-izi, market-price-finder-mcp-tr, PriceGetter, pricebook (birim fiyat), MarketPricesMCP.
Veri:
- 10/10 sonuç gerçek market fiyatı takibi/karşılaştırması. "TL" kelimesi aramayı Türk market araçlarına çekti.
- Hiçbiri "hatırlanan değerle ilerle, farkı sonra öde" yapısında değil; hepsi önce doğru fiyatı çekip sonra karar veriyor (P6'nın tersi).
- LLM çalıştırma tekniği 0/10.

### Tur 4 sonucu
- (a) Kanıtlanan: P6 birebir, GitHub'da 0/10. Toplam 50 sonuç, LLM çalıştırma tekniği 0/50.
- (b) Çürütülen: "P6 birebir GitHub'da aranınca teknik çıkar" → öldü. Para birimi ve "market" kelimesi aramayı gerçek fiyat araçlarına bağlıyor.
- (c) Sıradaki tek adım: P6 birebir, tez veritabanında (OpenAlex) aramak — denenmemiş son kaynak türü.

### S13 — P6 birebir İngilizce, OpenAlex, 10 istendi
Dönen: 0 kayıt (boş liste).

### S14 — P6 birebir Türkçe (kullanıcının kendi cümlesi), OpenAlex, 10 istendi
Dönen: 0 kayıt (boş liste).
Veri: OpenAlex uzun cümlede tüm kelimelerin aynı kayıtta geçmesini istiyor; hiçbir kayıt bu kelime setinin tamamını taşımıyor.

### S15 — P6 birebir İngilizce, Crossref, 10 sonuç
Dönen: "Are You Now or Have You Ever Been?" (oyun, 4 farklı kayıt), "Teacher Librarian (TL)", "TL;DR" kitap bölümü, ASCO, "If You Have More Than Two Initiatives", "You can't de-race...", "How do you take away more than you have? (çıkarma problemleri, tez)".
Veri: Crossref başlık kelimesi eşliyor; "TL" Türk lirası değil "Teacher Librarian"/"TL;DR" ile eşleşti. Konuyla ilgili 0/10.

### Tur 5 sonucu (K2: blocker + denenmiş 3 yol)
- (a) Kanıtlanan: birebir kuralıyla 6 kaynak türü (arXiv, exa araştırma, exa GitHub, OpenAlex, Crossref; TR ve EN) tarandı. 15 sorgu, 130 sonuç (S1–S15). P1–P8'i karşılayan LLM çalıştırma tekniği: 0.
- (b) Çürütülen: "Birebir cümle, doğru kaynakta aranırsa hazır teknik bulunur" → öldü. Denenmiş 3 ucuz yol: OpenAlex EN (0 kayıt), OpenAlex TR (0 kayıt), Crossref EN (0/10).
- Blocker: hazır teknik yayımlanmamış ya da birebir cümleyle erişilemiyor. Kaynak: S1–S15.
- (c) Sıradaki tek adım: arama aşaması kapandı, Aşama 3'e geçiş. P1–P8'den kendi tekniğimizi kurmak. İlk hipotez H11: P6 ("hatırlanan değerle ilerle, fark çıkarsa kafaya yaz, durma") yapısını echo-core'da kurmak ve ölçmek.

## Önceki sorgular (kullanıcının "kendi kelimelerinle arama" itirazından ÖNCE, benim çevirimle yapılmış — veri olarak saklanıyor)

- Jacobi decoding / CLLM (2403.00835, 2.4–3.4x, eğitim gerekir) / Lookahead decoding (2402.02057, 1.8x MT-bench, eğitimsiz) / Jacobi Forcing (2512.14681, 3.8–4.0x, eğitim gerekir) / Parallel Jacobi Decoding görüntü (2606.05703) / Speculative Jacobi Decoding (2410.01699) / FastCoT (2311.08263, %20).
- DEER (2309.12252), ELK quasi-Newton (2407.19115), ParaRNN (2510.21450), Gonzalez tez 2026 (Lyapunov üssünün işareti paralelleşmenin hızlı yakınsayıp yakınsamayacağını belirliyor), "Predictability Enables Parallelization" NeurIPS 2025.
- Fast-dLLM (2505.22618, 27.6x throughput LLaDA/Dream), ParallelBench (2510.04767, paralel çözümde kalite düşüşü), "How Efficient Are DLMs" (2510.18480, AR genelde daha yüksek throughput), Mercury, LLaDA, RePara (2610.18200, 16 token/adımda matematik/kod %15 altı), Flash-dLLM (2609.26796).
- TEAL (2408.14690, 1.53–1.8x), ReLU Strikes Back (2310.04564), DeltaLLM (2507.19608), Δ Attention, SpAx (2610.02598), Double Sparsity (2408.07092), DARE (2605.08134), KVPR.
- **Rollback-free value prediction (Thwaites vd. PACT 2014; RFVP TACO 2016):** önbellekte olmayan değeri tahmin et, yanlış tahmini takip etme ve geri alma; ort. %8 hız, %0.8 kalite kaybı. **Load Value Approximation (San Miguel vd. MICRO 2014):** ort. %8.5, en çok %28.6 hız. GPU için adres-adım destekli yaklaşık değer tahmini (2019). Seçici yaklaşık hesapla spekülatif yürütme (Nongpoh vd. 2019).

## Aşama 3 — kendi tekniğimiz (tur 6, ölçümlü)

Ölçüm: Ryzen 5 3600, t=6, `--llama-bench -p 0 -n 32,128 -r 3`, Qwen3.5-0.8B-Q8_0, IVF LM head. Kod: `src/llama/bench.zig` (`--p6`, `--snap`), `src/llama/qwen35.zig` (`no_trace`, `snapshot`, `rollback`).

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H11 | P6: taslak zincirinin hepsi tek forward'da ödenir, model farklı derse fark sayılır (`miss`), geri alma yok | Bekleme (geri alma) zamanı yiyor | `miss` say; tg ölç | Benchmark'ta `miss=0`, `alpha=1.000`: geri alma hiç olmuyor, kurtarılacak bekleme yok. Hız aynı (128.6 vs 127.8). **Öldü** (bu ölçümde P6'nın kurtaracağı şey yok). |
| H12 | P2: araba (batch 8) tek tur maliyetinde mi? | batch 8 RAM sınırlı | t=6 → t=3: RAM sınırlıysa hız düşmez | batch 1: 50.99→49.38 (0.97), batch 4: 165.87→110.20 (0.66), batch 8: 213.62→133.31 (0.62). batch 8 **hesaplama** sınırlı. **Yaşadı.** Araba 4 üründe dolup taşıyor. |
| — | Tur zaman dökümü (H01, tg128) | — | 128 token = ~15 tur×8 + 8 tek | Tur ≈ 56 ms, saf batch 8 = 37.5 ms → tur başına ~19 ms doğrulama ek yükü. |
| H11b | P5: "kartın onayını bekleme": GDN durum izi (18 katman × 8 pozisyon × 1 MB ≈ 150 MB yazma/tur) onay maliyeti | iz yazımı ~%15+ | izi kapat (`no_trace`), ölç | tg128 136.9/139.5 → 164.7/162.1 (+%18). **Yaşadı.** |
| H13 | P6 "yanına ekstra 200 TL al": pozisyon başı iz yerine tur başı tek snapshot (~19 MB); tutmazsa snapshot'a dön + kabul edileni yeniden işle (P8 "telefonla ara" bedeli yalnız tutmayan turda) | kayıpsız ve iz kadar hızlı | 4 istemde `--ngram 0` ile bayt karşılaştırma; tg ölç | 4/4 bayt bayt aynı (fibonacci, Paris, masal, sohbet). tg128: 136.6/135.7 → 162.9/165.8 (+%20); tg32: 99.3/88.3 → 119.8/118.9 (+%27). **Yaşadı.** |

### Tablo (CPU, tg, tok/s)

| Satır | tg32 | tg128 | Kat (tg128) |
|---|---:|---:|---:|
| Hilesiz (`--ngram 0`) | 52.38 | 53.64 | 1.00x |
| H01 (iz ile geri alma) | 93.8 | 136.2 | 2.54x |
| H01 + H13 (snapshot) | 119.3 | 164.3 | 3.06x |
| Saf batch 8 tavanı (doğrulamasız) | — | 213.6 | 3.98x |

### Tur 6 sonucu
- (a) Kanıtlanan: H13 kayıpsız (4/4 bayt aynı) ve tg128'de +%20. Batch 8 hesaplama sınırlı (thread yarıya inince 0.62x).
- (b) Çürütülen: H11 (P6 = geri almayı kaldır) bu benchmark'ta kazanç vermez, çünkü geri alma hiç olmuyor. "Batch RAM sınırlı" varsayımı 4 üstünde öldü.
- Kalan tur maliyeti: tur ≈ 42 ms, saf batch 8 = 37.5 ms. Fark (~4.5 ms) LM head'in 8 pozisyonu sırayla 8 ayrı aramayla yapması (P2'ye aykırı: 8 ayrı tur).
- 300 tok/s için: 8 token/tur ile tur ≤ 26.7 ms olmalı. Saf batch 8 bugün 37.5 ms; Q8×Q8 4×2 çekirdeği hesaplamada ~1.4x hızlanmalı.
- (c) Sıradaki tek adım: H14 — LM head'i 8 pozisyon için tek geçişte yapmak (P2: tek araba).
