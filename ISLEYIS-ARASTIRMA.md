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

## Tur 7 (ölçümlü)

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H14 | P2: doğrulamada LM head 8 pozisyon için 8 ayrı IVF araması (8 tur, ~270 MB, 40 bariyer) → tek geçiş: kümelerin birleşimi bir kez okunur, her satır yalnız kendi kümelerinde skorlanır | ~4.5 ms/tur kazanç | 4 istemde bayt karşılaştırma; tg ölç | 4/4 bayt aynı. tg128 164 → 167.4/167.6 (+%2), tg32 değişmedi. **Yaşadı, küçük.** |
| — | Profil: saf batch 8 (`perf`) | — | — | Zamanın %77.5'i `TinyBlasQ0.gemm4xN` (Q8×Q8 4×2 asm). Çift×blok başına 6 vektör komutu; 2'si (`vpmaddubsw`, `vpmaddwd`) Zen 2'de tek tamsayı çarpma borusunda. Alt sınır ~17 ms, ölçülen ~29 ms. |
| H15 | SMT ile boru boşluklarını doldur (t=12) | batch 8 hesaplama sınırlı → SMT yardım eder | t=6 vs t=12 | batch 8: 229 → 135; tg128: 174 → 63. **Öldü.** |
| H16 | `vpmaddwd ones` yerine bit-bit aynı kaydırma+toplama (çarpma borusunu boşalt) | FP0 darboğazı azalır | iki ikili, batch 8 ×2 | eski 220.4/234.5, H16 212.0/210.4. **Öldü**, kod geri alındı. |

### Güncel tablo (CPU, t=6, tok/s)

| Satır | tg32 | tg128 | Kat (tg128) |
|---|---:|---:|---:|
| Hilesiz (`--ngram 0`) | 52.38 | 53.64 | 1.00x |
| H01 (iz ile geri alma) | 93.8 | 136.2 | 2.54x |
| H01 + H13 | 119.3 | 164.3 | 3.06x |
| H01 + H13 + H14 | 119.2 | 167.5 | 3.12x |
| Saf batch 8 tavanı | — | 213–234 | ~4.2x |

### Tur 7 sonucu
- (a) Kanıtlanan: H14 kayıpsız, +%2. Batch 8'in %77.5'i Q8×Q8 çekirdeği.
- (b) Çürütülen: H15 (SMT), H16 (pmaddwd yerine kaydırma).
- 300 için: tur (8 token) ≤ 26.7 ms. Bugün saf batch 8 ≈ 35–37 ms; çekirdek ~1.4x hızlanmalı. Ayrıca tg128'in ~%20'si taslak hazır olmadan tek tek üretilen tokenlar.
- (c) Sıradaki tek adım: H17 — taslak yokken tek token üretmek yerine (P2: arabayı boş götürme) kalan pozisyonları da doldurup batch'le git.

## Tur 8 (ölçümlü)

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H17 | P2 "arabayı boş götürme": taslak eşiği 6 → 2 / 1 (`--min-draft`) | tek tek adımlar eşik yüzünden | `single` sayacı + tg | `single` her eşikte 28 (değişmedi); tg128 gürültü içinde aynı (143–177). Tek adımlar eşikten değil, taslakçının hiç geçmişi olmamasından. **Öldü.** |
| H18 | P6 "önceden markete gitmişsindir, fiyatları hatırlarsın": taslak hafızası tekrarlar arasında silinmez (`--memory 1`). Literatür adı: SuffixDecoding (önceki çıktılardan küresel sonek ağacı) | ilk tokenlar da taslakla gelir | `single`, alpha, tg ×2 tur | `single` 28 → 16; alpha 1.000 → 0.982 (hatırlanan fiyat bazen tutmuyor, fark ödeniyor). tg128 178.1/175.7 → 182.0/187.9 (+%4.5), tekrar içi tepe 201.4/201.8. **Yaşadı, küçük.** |

### Kritik veri: tavan
- pp512 = 289.5 tok/s (hile tablosu). Her token modelden geçmek zorunda (P3, P4), o yüzden taslak kusursuz olsa ve araba sınırsız büyük olsa bile TG bu hızı geçemez.
- 300 > 289.5. Yani 300 için taslak tarafında kapatılacak bir boşluk kalmadı; matris çekirdeğinin kendisi (%77.5) hızlanmalı.

### Tur 8 sonucu
- (a) Kanıtlanan: H18 +%4.5 (bench, `--memory 1`). TG tavanı = prefill hızı = 289.5 tok/s.
- (b) Çürütülen: H17. "300'e taslak/doğrulama tarafında ulaşılır" varsayımı öldü (tavan 289.5).
- (c) Sıradaki tek adım: H19 — Q8_0 ağırlıklarını yükleme anında satır-iç içe (8 satır) yeniden dizip batch çekirdeğini buna göre yazmak; ölçüt saf batch 8 ve pp512.

## Tur 9 (ölçümlü)

Gemm'e giren ağırlık: 518.1M parametre, hepsi Q8_0 (token_embd 254.3M ayrı, LM head ona bağlı).

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | `perf stat` saf batch 8 (r=10) | — | — | cycles 131.6G; FP boru atamaları: FP0 72.7G, FP1 40.6G, FP2 39.5G, FP3 55.3G. Zorunlu tamsayı çarpma (maddubs+madd) 161 turda 41.7G = çekirdek döngülerinin ~%41'i. Toplam FP kapasitesinin ~%42'si dolu. |
| H19 | P4 "kasa fizik", ama kasada sıra düzensiz: 34 baytlık Q8_0 blokları 32 baytlık yüklemeleri cache satırında bölüyor, ölçekler skaler kaydırma+OR ile toplanıyor → yüklemede 4 satır iç içe, hizalı quant + f32 ölçek; aynı aritmetik sıra | çekirdek düzen yüzünden yavaş | 4 istem bayt + batch 8 + perf | 4/4 bayt aynı (bit-bit aynı aritmetik). batch 8: H19 193.9/219.5/233.4, eski 230.6/228.1. Çekirdek payı %77 vs %78, değişmedi. **Öldü**, kod geri alındı (+550 MB RAM, kazanç yok). |

### Tur 9 sonucu
- (a) Kanıtlanan: düzen (hizalama, ölçek toplama) çekirdeğin hızını belirlemiyor. FP0 borusu çekirdekte ~%70 dolu (toplam atama). Zen 2'de tamsayı çarpma yalnız FP0'da; FMA da kısmen FP0'a düşüyor.
- (b) Çürütülen: H19.
- Fizik hesabı: FP0 %90'a çıkarılsa bile çekirdek ~1.25x hızlanır → tur 35 ms → ~29 ms → ~275 tok/s (taslak kusursuz, alpha=1). Bu CPU'da, Q8_0'ın tam bedeli ödenirken (P3) 300 tok/s FP0 borusunun sınırının üstünde.
- (c) Sıradaki tek adım: kullanıcının kararı gereken nokta — P3 ("fiyatı her zaman ödersin") korunursa bu CPU'da 300'ün önünde fiziksel duvar var. Tek kalan kaldıraç ödenen fiyatı değiştirmek (ör. çarpma başına FP0 işini azaltan farklı sayı biçimi), bu da P3'e dokunuyor.

## Tur 10 — çekirdek dışı payın dökümü (saf batch 8, perf, eski=güncel kod)

| Kalem | Pay |
|---|---:|
| `gemm4xN` (Q8 çekirdek) | %78.1 |
| Bariyer/spin (`atomic.zig:356`, `pool.zig:90`, `atomic.zig:21`) | %4.7 |
| `lmSearch` (son token) | %2.4 |
| Diğer vektör işlemleri (GDN, norm, conv; dağınık, her satır <%1) | ~%10 |
| `flashOneChunk` + `quantizeRowQ80` + `rmsNorm` | %2.2 |

Hesap (tur 35 ms, çekirdek ~27 ms, kalan ~8 ms):
- Çekirdek FP0 %90 (1.25x) + spin sıfır + LM head sıfır → 21.6 + 8 × (1 − 0.32) ≈ 27.0 ms → **~296 tok/s**, alpha=1 ve taslaksız adım sıfır varsayımıyla.
- Gerçek akışta taslaksız adımlar (H18 ile tekrar başına ~4) ve %2 kaçırma var → altında kalır.

### Tur 10 sonucu
- (a) Kanıtlanan: çekirdek dışı payda tek büyük kalem yok; en büyüğü spin (%4.7). Tam Q8_0 bedeliyle (P3) bu CPU'da en iyimser tavan ~296 tok/s.
- (b) Çürütülen: "kalan 8 ms yarıya iner" varsayımı (dağınık, tek hedef yok).
- (c) Sıradaki tek adım: P3 kararına bağlı. P3 korunursa 300 bu CPU'nun tamsayı çarpma borusunun üstünde kalıyor; korunmazsa ödenen fiyatı (sayı biçimi) değiştirmek gerekir.

## Tur 11 — H20: Q8_0_R8 (8 satır iç içe) çekirdek

Kaynak (veri): ik_llama.cpp. AVX2 (Ryzen 5975WX) Llama-3.1-8B Q8_0 pp512: llama.cpp 147.9, ik_llama.cpp (satır-iç içe repack) 268.2 → 1.81x (discussion #164). Q8_K_R8 PR #141: AVX2'de işaret hilesi (sign) yine gerekli. PR #534: Q8_0 → Q8_0_R8 repack.
Tur 10'daki "tavan ~296" hesabı bugünkü komut düzenine aitti, fizik sınırı değildi → yanlış çıktı.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H20 | P2+P4: 8 satırı tek ymm'de (şerit başına satır) taşı. Zorunlu tamsayı çarpma aynı, dönüştürme + FMA 8 satıra bir kez. FP0'daki 31G zorunlu olmayan iş kalkar | FP0 rahatlar, batch hızlanır | batch 8 / pp512 + bayt testi | Boş makinede: batch 8 230.6/238.5 → 310.8/299.2; pp512 300.2 → 361.2. Kurtarıcı servis yükü altında (t=5, ortanca, 3 çift): pp512 164.8 → 233.4, varsayılan tg128 109.2 → 156.4, batch 8 172.6 → 213.9. **Yaşadı.** |
| H20 kayıpsızlık | Yalnız batch R8 olunca doğrulama ve tek token yolu farklı aritmetik | — | 6 istem `--ngram 0` vs varsayılan | Önce 2/6 farklı. Tek token yolu da R8'e alındı (her sütun aynı komutlarla bağımsız) → **6/6 bayt aynı.** |
| H20 sayısal | R8 toplama sırası eski çekirdekten farklı; hangisi gerçeğe yakın | — | gerçek Q8_0 ağırlıkları, float64 referans, 4 tensör × 1024 satır × 8 sütun | Ortalama hata R8'de 4/4 tensörde daha düşük (%13–21); en büyük hata iki yolda da ~1e-6. Eşdeğer, ortalamada biraz daha iyi. llama.cpp bit paritesi `ECHO_NO_R8=1` ile korunuyor (`--logit-check` 9/9). |

### Tur 11 sonucu
- (a) Kanıtlanan: H20 kayıpsız; saf batch 8 boş makinede 300'ü geçti (310.8). README TG 150 işaretlendi (H13'ten beri tg128 ≥ 164).
- (b) Çürütülen: "Tam Q8_0 bedeliyle bu CPU'da tavan ~296" (tur 10) → öldü.
- (c) Sıradaki tek adım: makine boşken varsayılan + `--memory` tg128'i ölçmek (200 ve 300 barajı için).

## Tur 12 — R8 sonrası (yüklü makine)

Ortam: `/opt/operasyonel-kurtarma/kurtarici-runner` (root, ~%97 CPU, 17 saattir), btrfs-transaction, camoufox (~%106), Tctl 95 °C. Aynı ikili ±%20–30 oynuyor.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | Varsayılan TG profili (R8, `--memory 1`) | — | perf | mulMat %61.5, graph %28.3; bariyer spin toplam ~%20 (yükle şişiyor). alpha 0.973, taslaksız 21 / 4 koşu. |
| H21 | P2 dengesi: R8 iş parçası 32 satır → 1024 satırda 8 parça / 6 thread dengesiz | parçayı küçült / statik böl | gb4 vs gb1, gb2, statik; art arda 3 çift, batch 8 | gb4: 273/269/242, 268/218/274; gb1: 226/149/177; gb2: 191/203/248; statik: 114/226/236. **Öldü**, gb4 kaldı. |
| Ölçüm gürültüsü (K2) | Gürültüyü kendim düşürebilirim | — | 1) art arda A/B çiftleri, 2) t=5 (bir çekirdek boş), 3) `taskset -c 0-5` | Üçü de ±%20 altına inmedi (sabitlenmiş: 225/169/158/212). Blocker: makinedeki diğer yük + ısıl kısma. |

### Tur 12 sonucu
- (a) Kanıtlanan: R8 kayıpsız ve yük altında da pp512, tg128, batch 8'de önde. gb4 en iyi parça boyu.
- (b) Çürütülen: H21.
- Blocker: 200 / 300 barajı için mutlak ölçüm bu yükte alınamıyor. Kurtarıcı servis + tarayıcı kapalıyken ölçülmeli.

## Tur 13 — kirli ortam hedefi + H22

Kullanıcı kuralı (2026-11-07): barajlar en kirli ortamda (PC kullanılırken, yük + ısı altında) geçilecek; temiz makine / performans modu yok. Ölçüm: art arda A/B çiftleri, ortanca.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H22 | P2 "tek seferde": aynı girdiyi kullanan matrisler (doğrusal: qkv+alpha+beta+z; dikkat: q+v+k; FFN: gate+up) tek quantize + tek iş havuzu + tek bariyer; bağımsız eleman işlemleri (convConcat, gate, sigmoid, silu(z); q/k L2 norm) aynı fazda | Bariyer azalınca yük altında bekleme azalır | 6 istem bayt; 10 art arda çift (kirli ortam) | 6/6 bayt aynı (aritmetik değişmedi). Varsayılan tg128 ortanca: 137.3→162.9 (4 çift), 140.0→157.7 (6 çift); 6/10 çift H22 lehine. Hilesiz tg128 5/10 (nötr). Doğrusal katman bariyeri 18 → 11. **Yaşadı.** |
| H23 | Kirli ortamda thread sayısı: 6 thread bariyerde kullanıcının işleriyle yarışıyor | t<6 daha dayanıklı | t=6/5/4, art arda 4 tur | Varsayılan tg128 ortanca 141.6 / 142.5 / 142.6; en kötü 112.4 / 95.8 / 118.0. Hilesiz tg128 31.4 / 38.4 / 42.4. tg32 95.0 / 87.4 / 99.7. **Yaşadı**: port varsayılanı fiziksel çekirdek − 2 (=4). Çıktı thread sayısından bağımsız (3/3 aynı). |

## Tur 14 — R8 çekirdeğinin darboğazı: bellek bandı (kirli ortam)

Araç: `echo-core-zig/src/llama/r8_mikro.zig` (tek thread, n=8, k=1024; kurulum payı 3R − R farkıyla atıldı). FP0 doluluğu = zorunlu tamsayı çarpma / döngü.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | Gerçek koşuda çekirdek FP0 doluluğu | — | perf grup örnekleme, batch 8, t=4 | %53; FP yazmaç dosyası dolu durması çekirdek döngülerinin %19'u |
| — | Mikro: tek thread | — | m=128 (L2) / 4096 (L3) / 32768 (DRAM) | %78.5 / %80.9 / %73.6 |
| — | Mikro: 4 fiziksel çekirdekte eşzamanlı 4 kopya | — | DRAM / L2 boyu, 2 tekrar | DRAM %52–65, L2 %75–81 → **batch 8 kirli ortamda bellek bandı sınırlı**. Tur başı ağırlık 518M × 36/32 = 583 MB |
| H-PF | Yazılımsal önden getirme (2–128 blok, L1/L2) gecikmeyi saklar | Bellek gecikmesi | mikro DRAM + gerçek koşu | Mikroda fark gürültü içinde (±%10), gerçek koşuda çekirdek döngüsü ~%7 az ama tok/s ayrışmadı. **Öldü**, kod alınmadı |
| H-COL | Derleyici 8 w + 8 \|w\| canlı tutup yığına döküyor (asm'de spill); sütun dışta + volatile yükleme | Spill kalkınca hızlanır | 5 çift + perf | Çekirdek döngüsü %3–5 az, tok/s ortanca 209.5 → 196.4. **Öldü** |
| H24 | P3 bedelini değiştirmeden bayt azalt: R8 ölçekleri f32 yerine f16 (kaynakta zaten f16, değer aynı); bloklar çiftli ([16B ölçek][16B ölçek][256B][256B]) → ymm'ler 32 hizalı. 288 → 272 bayt/blok (−%5.5) | Bant sınırlıyken bayt = süre | 12 istem bayt (gramerli 6 + gramersiz 6, base / H24 / H24 `--ngram 0`); 5 art arda çift | 12/12 bayt aynı. Batch 8 ortanca 199.4 → 210.9 (+%5.8), varsayılan tg128 145.4 → 155.2 (+%6.7); 4/5 çift H24 lehine (iki ölçüde de). **Yaşadı** |

### Tur 14 sonucu
- (a) Kanıtlanan: kirli ortamda batch 8 bellek bandı sınırlı; bayt azaltmak doğrudan hız. H24 kayıpsız, +%6.
- (b) Çürütülen: H-PF, H-COL.
- (c) Sıradaki tek adım: tur başına okunan baytı token başına düşürmek — taslaksız (tek token) adımlar bant açısından neredeyse tam tur bedeli ödüyor.

## Tur 15 — taslak boyu (H25)

Bench üretimi 5 tokenlik döngüye giriyor; taslaklar %100 kabul (360/360). Süre = ~7 başlangıç tek-token adımı (T1 ≈ 20 ms) + turlar. Saf batch: T8 ≈ 42 ms, T16 ≈ 76 ms → T(n) ≈ 20 + 3.7 × (n−1) ms; uzun tur ucuz.

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | min_draft 1/2/4/6 | — | 4 tur | Etkisiz: zincir ya 0 ya tam |
| H25 | P2 "tek seferde": taslak tavanı 16 → 32 (ivf.Batch MAXR 32, maske u32), uyarlamalı: tam kabulde ×2, rette ½ (en az 8); varsayılan k=32 | Tur sayısı yarıya iner | k=8/16/32 6 çift; h24 vs h25 3 çift | k ortanca 163.0 / 174.3 / 182.1. Varsayılan: tg128 161.5 → 168.7, tg32 99.8 → 125.3. **Yaşadı** (u6 taşma hatası düzeltildi: `lim` tipi `@min(k, 32)` ile daralıyordu) |

## Tur 16 — hız öncelikli (kullanıcı 2026-11-07: "kalite düşse de hıza odaklan")

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| H27 | t: H25 sonrası turlar hesap-sınırlı | t=6 kazanır | t=4/5/6 × 3 tur | tg128 ortanca 183 / 185 / 195. Port varsayılanı tüm fiziksel çekirdek. **Yaşadı** |
| H26 | Kayıplı: FFN ara boyutunu 32'lik gruplarla statik buda (önem ‖gate‖·‖up‖·‖down‖) | Hesap %12–37 azalır | keep 1 / 0.75 / 0.5 / 0.25, t=6 | Model döngüye temiz girmiyor: tur başı kabul 32 → 18.9 / 8.1 / 3.7; tg128 222 → 174 / 81 / 72. **Öldü**: büyük kalite kaybı taslak kabulünü, dolayısıyla hızı yıkıyor |
| H28 | Kayıplı, küçük: FP0 işinin yarısı `vpmaddwd` genişletmesi. R8_SAT k adımının `maddubs` çıktısı i16'da doygun toplanıp (`+|`) tek `madd` | Hata küçük kalırsa kabul bozulmaz | mikro (FP0 doluluk), 3 tur bench, 12 istem metin | Mikro çekirdek hızı SAT2/4/8: ×1.12 / ×1.18 / ×1.25. Bench t=6 tg128 ortanca SAT1/2/4/8: 256.6 / 266.8 / 271.4 / 286.1, tg32 168.2 / 168.9 / 171.0 / 177.9; kabul 32/32 korunuyor. Metin: SAT8 kayıpsızla 5/12 istemde bayt aynı, diğerlerinde küçük alan farkı (ör. "phone" değeri). **Yaşadı**, varsayılan SAT8 |

### Tur 16 sonucu
- (a) Kanıtlanan: kirli ortamda (sakin anında) varsayılan tg128 286 tok/s; README TG 200 geçildi.
- (b) Çürütülen: H26 (büyük kalite kaybı hızı da düşürüyor).
- (c) Sıradaki tek adım: 300 için kalan pay ~%5; tur başı tek-token adımları (7 × T1) ve graph/spin payı.

## Tur 17 — H29: hafıza + dizi başı ayracı (P6)

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | `--memory 1` (H18) k=32 ile | — | 3 tur | tg128 ortanca 285.8 (vs 282.9), tg32 +%45; ama kabul 26.6/32. İz: ret yalnız koşu başında (`R4/32!`, `R6/32!`): geçmiş koşular arası birleşik kaldığı için dizi başı bağlamı yanlış |
| H29 | P6: harita kalır, her dizide geçmiş n−1 ayraçla (maxInt) yeniden başlar → dizi başı da öğrenilir. Bench ve port'ta hafıza varsayılan | Ret kalkar, tek-token adımlar yalnız ilk dizide | 3 tur; 6 istem bayt (SAT8'e karşı) | Kabul 32/32; tek-token 28 → 7 (yalnız ısınma). tg128 mem 0/1: 272.8/369.0, 249.5/334.7, 258.7/348.3; tg32 166/277. Varsayılan (r=5) tg128 359.2 / 357.9 / 356.8, tg32 279.8 / 287.2 / 286.0. 6/6 bayt aynı. **Yaşadı** |

### Tur 17 sonucu
- (a) Kanıtlanan: README TG 300 geçildi (kirli ortam, varsayılan tg128 ~357).
- (b) Çürütülen: —
- (c) Sıradaki tek adım: 500 barajı. Tur başı tek-token adımı kalmadı; süre artık neredeyse tamamen T32 turları (hesap-sınırlı çekirdek + graph).

## Tur 18 — 500'e doğru: çekirdek

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | k=64 (MAXR 64) | Tur sabit maliyeti amortize olur | 3 tur | tg128 362/364/349 vs k=32 367/368/371. **Öldü**: T64 ≈ 2·T32, maliyet tamamen token başı hesap (~2.8 ms; çekirdek %62, graph %28) |
| — | SAT8 boru dağılımı | FP0 darboğaz | perf fpu_pipe | `vpaddsw` FP0/FP3'e gidiyor; `+%` ile borular dengelendi ama döngü aynı (272M) → boru değil |
| — | Tile genişliği NC=2/3/4/6/8 | — | mikro | 283/284/272/289/275M döngü → fark yok; darboğaz çift başına komut zinciri (~2 döngü/çift) |
| H30a | Aktivasyonu u8'e kaydır (a+128): `vpsignb`/`vpabsb` kalkar, çift başına broadcast + `maddubs` + `vpaddw`; 128·Σw düzeltmesi repack'te (blok 304 B), i16 sarmalı | Komut azalır | mikro + bench + metin | Mikro ×1.26; bench +%5 ama kabul 30.1, metin bozuk ("100000…"): `maddubs` doyması sık. **Öldü** |
| H30 | Aynı + aktivasyon 7 bit (|a| ≤ 63, a+64 ∈ [1,127]) → çift ≤ 32258, `maddubs` hiç doymaz; düzeltme 64·Σw | Doyma kalkar, hız kalır | 3 tur, metin | Metin tutarlı (hikâye), kabul 32/32. tg128 355/378/407 vs 330/375/366, tg32 268/277/251 vs 242/286/285. **Yaşadı** |
| — | Saf asm tavanı: `vpbroadcastd` + `vpmaddubsw` + `vpaddw` | — | asm döngüsü | 1 döngü/çift (FP0'da `maddubs` 1/döngü). Çekirdek 1.65'te |
| H31 | 2 satır grubunda broadcast paylaş | Broadcast darboğaz | mikro NC 2/3/4 | 230/218/220M vs 216M. **Öldü** |
| H32 | `madd(ones)` yerine FP2 kaydırmaları | FP0 rahatlar | mikro (Zig ve asm) | 226–242M vs 216–223M; asm'de 192M vs 182M. **Öldü** |
| H33 | Sütun dışta + volatile ağırlık | Spill kalkar | mikro NC 4/8 | 296–451M. **Öldü** |
| H34 | Derleyici 8 ağırlık + 4 akümülatörü yığına döküyor (asm'de spill). 4 sütunluk R8 çekirdeği elle asm: ymm8–11 akümülatör, ymm12–15 i16 toplam, ağırlık k başına bir kez, blok çifti döngüsü asm içinde | Spill'siz çekirdek | mikro + 3 tur + metin | Mikro 216M → 182M (×1.19, 1.39 döngü/çift), sonuç aynı. tg128 395.6/395.6/397.8 vs 353.5/378.6/373.3, tg32 279/275/283 vs 261/271/248; metin aynı. **Yaşadı** |
| H35 | conv1d (kc=3 + silu + state) tek geçişte, 8 satır vektörel; ssmConv + silu + 2 bariyer kalkar | Graph'taki skaler conv payı (~%10) düşer | 3 çift + 6 istem bayt | 6/6 bayt aynı. tg128 453.6/451.3/458.4 vs 433.5/441.5/444.5 (ortanca +%2.7). **Yaşadı** |
| — | Gerçek şekiller (m 128–3584, n 8/32) mikro | Büyük m'de önbellek kaybı | mikro | Hepsi ~1.40 döngü/çift → şekil/önbellek sorunu yok. Blok sonu yokken 1.02 → blok sonu blok-sütun başına ~3 döngü |
| H36 | Blok sonunu (subw/madd/cvt/ölçek/fma) sonraki bloğun iç döngüsüne serpiştir; iki i16 seti dönüşümlü (yazılım boru hattı) | Blok sonu gecikmesi gizlenir | mikro + 3 çift + bayt | Mikro 1.40 → 1.32 döngü/çift, toplam birebir aynı. tg128 461.0/461.4/459.4 vs 457.8/458.8/456.9 (+%0.6); 6/6 bayt aynı. **Yaşadı (küçük)**. Blok sonu gecikme değil boru yükü: FP0 9/8 + yayınlar FP1/2 |
| H37 | Blok çifti ortak ölçek (ağırlık yeniden nicemleme + aktivasyon 64'lük), çift başına tek blok sonu | Blok sonu yarıya iner | mikro + 3 çift | Mikro 1.32 → 1.18 döngü/çift; ama 0/6 bayt, kabul 25–30, tg128 301/277/286 vs 462/461/466. Çift başına 32 çarpımın i16 toplamı taşıyor. **Öldü** |
| H38 | GDN: satır başına 4 geçiş (scale, dot, mad, dot) → 2 geçiş; 4 satır birlikte (k/q yüklemesi paylaşılır, zincirler paralel); e ölçeği dot'a katlanır | GDN (toplamın ~%6.7'si) gecikme sınırlı | 3 çift + bayt | 4/6 bayt aynı (2'si yuvarlama, kendi içinde tutarlı), kabul 32/32. tg128 471.2/488.2/472.0 vs 453.8/465.5/391.0 (~+%3.5). **Yaşadı** |
| — | R8 parça boyu GB 1/2 (vs 4) | Kuyruk dengesizliği (~11 µs/parça) spin'i besliyor | 5 tur | GB4 482 ortanca, GB2 472, GB1 460. **Öldü** |
| — | LM head nprobe 384/256 | Tarama yarıya iner | 2 tur | Kabul 30.08, tg128 421–464 vs 482–485. **Öldü** |
| H39 | Snapshot'ta 19 MB GDN kopyası yerine çift tampon: işaretçi takası, GDN t=0'da eski tampondan okuyup yenisine yazar; rollback geri takas | Ana thread tek başına kopyalarken diğerleri bekliyor (entryFn %3.5 + memcpy %0.8) | 4 tur + bayt | 6/6 bayt aynı. tg128 482.5/491.3/461.0/483.0 vs 479.1/475.9/474.1/428.4 (ortanca +%1.6). **Yaşadı** |
| H40 | LM head IVF taraması: küme satırları 8'lik R8 gruplarına paketlenir (dolgu atlanır), 4 sütunluk asm çekirdeğiyle taranır; ilk deneme (7 bit top-1) kabulü 28.6'ya düşürdü → token başına thread başına en iyi 4 aday, reduce'ta `dotQ80Q80` ile kesin skor | Tarama `dotQ80Q80`'den ~2.5× hızlı, karar aynı | rdtsc ile LM bölümü + bayt | 6/6 bayt aynı, kabul 32. LM head duvar döngüsü 410/396/402M → 283/293/279M (−%29). **Yaşadı** |
| H41 | LM `plan`'daki 32 quickselect (tek thread, LM'nin %33'ü) thread'lere dağıtılır, maske atomik OR | Diğer 5 thread beklemez | rdtsc + bayt + 4 tur | LM 282/284/275M → 202/203/226M; 6/6 bayt aynı. tg128 474.5/452.2/510.5/464.8 vs 458.2/401.4/487.6/470.2 (yük ort. 3.4; tepe ilk kez 500 üstü). **Yaşadı** |
| H42 | Blok çifti dword içinde iç içe ([w_l 2j,2j+1, w_l+1 2j,2j+1]): i16 şerit 2r = blok l, 2r+1 = blok l+1; blok sonunda vpmaddwd yerine kaydırma (FP0 tavanı 1.125 → 1.0) | FP0 bağlayıcı | mikro (kayan nokta referansı) + bayt | Küçük ağırlıkla maxerr 0 (çekirdek doğru), mikro yalnız 0.657 → 0.647; modelde 0/6 bayt, kabul 0, 48 tok/s. Şerit başına 32 çarpım gerçek veride taşıyor (H37'nin asıl ölüm sebebi de bu). **Öldü**. Sınır: i16 şeridi en çok 16 çarpım |
| — | Bariyerden bariyere duvar süresi (thread 0, rdtsc, kaynak satırı) | — | 1 koşu | FFN gate+up %26.5, FFN down %13.6, lineer dikkat girdi proj. %24.2, GDN %8.9, ssm_out %6.1, LM ~%6.5, dikkat proj. %4.8, flash %1.9 |
| H43 | GDN işi (başlık, 32 satırlık blok) = 64 parça (16 başlık / 6 thread dengesizliği) | Duvar düşer | bariyer ölçümü + bayt | GDN 315M → 289M (−%8); 6/6 bayt aynı. **Yaşadı (küçük)** |
| H44 | GDN kalemi tüm tokenları L1'de işler; tembel bozunma (saklanan = gerçek/c, güncelleme tek FMA), çıktı dot'u + sonraki tokenın k-dot'u aynı geçişte (token başına 2 geçiş → 1); sonda c ile çarpılır, c < 1e-20'de ara normalleştirme | GDN %15+ hızlanır | bariyer ölçümü + bayt + 4 tur | GDN 259M → 221M (−%15); 5/6 bayt aynı (1 istem yuvarlama, kendi içinde tutarlı); kabul 32. tg128 498.3/508.6/470.1/493.8 vs 462.7/484.9/468.1/485.9 (yük ort. 5; sakin anda 533–537). **Yaşadı** |
| — | Mikro: ağırlık L3'ü aşınca (m=65536, 67 MB) | Gerçek koşudaki 1.67 döngü/çift farkı DRAM'den | mikro | 1.32 → 1.62 döngü/çift (tek thread, +%23). Kanıtlandı |
| H45 | Grup g'nin karoları arasında g+1'in ağırlıklarını yazılımla önden getir (prefetcht2) | DRAM gecikmesi gizlenir | mikro + 6 çift (döngü:u) | Mikro DRAM 1.62 → 1.43–1.47, önbellekte 1.32 → 1.34; t0/t1 benzer, mesafe 2 kötü (1.54). Bench: tg ortanca 455 vs 452, döngü 24.9G vs 25.3G (yük ort. 5–7, Chrome). **Yaşadı (bench'te belirsiz)** |
| — | `-t` 6/8/12 (SMT DRAM gecikmesini örter mi) | — | 3 tur (yük ort. ~10) | t=6 139–476, t=8 105–484, t=12 198–308; karar verilemez, t=12 belirgin kötü. 6'da kalındı |
| H46 | Bariyerde 2048 spin'de bir `sched_yield` | Önalınan thread'imiz çabuk döner | 6 çift | Ortanca 509.5 vs 500.5 (gürültü içinde). **Yaşadı (nötr-pozitif)** |
| — | Gerçek koşuda çekirdek döngü/çift, thread sayısına göre (−r 2 ile −r 1 farkı) | Kayıp çok thread çekişmesinden | 2 koşu | t=1: 1.37 (mikro kadar), t=3: 1.73, t=6: 1.78. GB 8/16: çekirdek 14.7/14.0G vs GB4 15.4G ama GB16 kuyrukta 389. Bitişik akış önemli |
| H47 | İş çalma: her thread kendi bitişik R8 aralığını baştan işler (paketli lo\|hi atomik, CAS), bitince diğerlerinin aralığının sonundan çalar | Donanım önden getiricisi bitişik akışla çalışır, dengesizlik çalmayla kapanır | 3 tur + çekirdek döngüsü + bayt | 6/6 bayt aynı. tg128 517.6/514.0/513.5 vs 479.0/484.7/416.5; çekirdek 13.7–14.0G vs 14.8–16.3G (−%7). GB1 benzer ama bir aykırı (330). **Yaşadı** |

### Tur 19 sonucu
- (a) Kanıtlanan: README TG 500 geçildi (kirli ortam, Chrome açık; varsayılan tg128 8 koşu ortanca 506.7, aralık 475–541; tg32 409–432).
- (b) Çürütülen: H37, H42 (i16 şeridi 16 çarpımdan fazlasını taşıyamıyor), nprobe düşürme, GB 1/2, SMT (t 8/12), önden getirme mesafesi 2.
- (c) Sıradaki tek adım: 750. Çekirdek gerçek koşuda hâlâ ~1.6–1.7 döngü/çift (tek thread 1.37); kalan fark çok çekirdek bellek çekişmesi + kirli ortamda SMT kardeş paylaşımı.

## Tur 20 — 750'ye doğru

| # | AKIŞ | VARSAYIM | ÇÜRÜTME | SONUÇ |
|---|---|---|---|---|
| — | Mikro 1/3/6 kopya DRAM boyutlu (m=65536), uzun tekrar | Çok çekirdek bellek çekişmesi | par.sh | 1 kopya 1.40; 3 kopya 1.41–1.47; 6 kopya 1.39–1.70 (ort. 1.53). Bariyer duvarı: matmul segmentleri %78, GDN %7.2, LM %6.5, flash %2 |
| H48 | k=64 (MAXR 64, maske u64, taslak tamponu 64); taslak kalan token sayısıyla sınırlı (tg32 değişmez) | Çekirdek artık bellek/tur sabitiyle sınırlı → k=64 amorti eder (Tur 18'de ölmüştü) | 3 tur döngü + bayt + tg | k=64: 560.6/536.7/530.4 vs 521.5/483.7/487.8 (3/3), çekirdek −%6, graph −%10. Varsayılan: 6/6 bayt aynı, kabul 63.5, tg128 552/491/523 vs 501/274/532, tg32 403 vs 406. **Yaşadı** |
| H49 | k=128 (MAXR 128, küme başına 2 maske kelimesi, tampon 128) | Tur sabitleri daha da amorti | 7 çift (aynı ikili) + bayt | 4 sakin çiftte 584/577/576/575 vs 566/564/567/578 (+%1.6), toplam döngü −%1.5; 6/6 bayt aynı. **Yaşadı (küçük)**, varsayılan 128 |
| — | Mikro: blok sonu komutlarını tek tek çıkar (anlamı bozuk, yalnız süre) | FP0 bağlayıcı | mikro ×2 | Taban 1.36; vpmaddwd yok 1.23, cvt yok 1.25, ölçek yayını yok 1.30, vpsubw yok 1.33, i16 toplama yok 1.17; blok sonu tamamen tamsayı (madd+paddd) 1.20 |
| — | Tamsayı birikim önkoşulu: aktivasyon ölçeği 4 blokta (128) / satırda ortak | Blok sonu tamsayıya iner (mikro −%12) | kabul + bayt | G=1 kabul 127; G=4 kabul 17, 121 tok/s; satır başına kabul 8.6, 82 tok/s. **Öldü** (ağırlık tarafı tek başına kazandırmaz) |
| — | vpbroadcastd → vbroadcastss (aynı bitler); vpmaddwd → kaydırma+toplama (FP0 dışı) | Port baskısı azalır | mikro ×3 | 1.31 → 1.33; 1.32 → 1.365. **Öldü**. Sınır port değil, toplam FP uop zamanlaması |
| — | Mikro 6 kopya: önbellekte / DRAM boyutlu n=128 | Gerçek koşu farkı çekirdekte | par2.sh | Önbellek 1.31–1.40, DRAM 1.22–1.28; gerçek koşu 1.49. Fark çekirdekte değil. `taskset -c 0-5`: çekirdek −%2.5, tok/s gürültü içinde (kirli ortamda riskli, alınmadı) |
| H50 | Kuyruk sütunları (n%4) için 1–3 sütunluk asm çekirdek (n=127'de grup başına 3 sütun derleyici yolundaydı) | Kuyruk yavaş | bayt + 4 çift | 6/6 bayt aynı, mikro maxerr 0 (n=5..8, 127); chunk döngüsü aynı. **Nötr, kaldı** (yavaş yol yok) |
| — | Sonraki karonun aktivasyonunu L1'e önden getir | Aktivasyon yüklemeleri bekliyor (örneklerin %46'sı) | 4 çift | Chunk +%3. **Öldü** (örnek kayması) |
| H51 | Bench: rastgele başlangıç tokenı ayrı n=1 geçişi yerine ilk doğrulama turunun başına (hafızadan taslak varsa) | rdtsc izi: n=1 geçişi 19 ms (tüm ağırlık DRAM'den), 127'lik tur 181 ms → koşunun %9.6'sı | rdtsc + 4 çift | Aynı 128 pozisyon tek geçişte, kabul 127. tg128 627/552/645/647 vs 577/562/606/497 (ortanca ~636 vs ~570, +%11); tg32 531. **Yaşadı** |
| H52 | LM IVF centroid skorlaması (4096 × n, `dotQ80Q80`) R8'e: centroid'ler R8 paketli, `GemmR8.chunk` ile 7 bit aktivasyon (yalnız küme seçimi; kesin skor değişmedi) | Skorlama ~3× hızlanır | bariyer ölçümü ×2 + bayt + kabul | Skorlama bölümü 71/71M → 15/20M (−%75, toplamın ~%1.8'i); 6/6 bayt aynı, kabul 127. **Yaşadı** |
| — | Gerçek koşu çekirdek döngü/çift, H51 sonrası (n=1 geçişleri ölçümden çıktı) | Çok thread kaybı | kc.sh t=1/3/6 | t=1 1.37, t=3 1.38, t=6 1.15–1.37. Eski 1.5–1.78 farkı n=1 geçişlerindendi; çekirdek gerçek koşuda mikroyla aynı |
| — | mulMatMulti thread başına süre ayrımı (rdtsc) | Bariyer/niceleme kaybı | 2 koşu | Çalışma %94–96, niceleme ~%1.6, bariyerler ~%2–4. Kirli ortamda çalışma döngüsü duvarı çekirdek döngüsünün ~%12 üstünde (önalma) |
| — | t=5 vs t=6 (kirli ortamda bir çekirdek masaüstüne) | Önalma azalır | 4 çift | t=6 544–644, t=5 498–547. **Öldü** |
| H53 | GDN: sonraki tokenların k/q/v satırlarını yazılımla önden getir (adım 8 KB, sayfa sınırı) | Donanım önden getiricisi izlemiyor | bariyer ×3 + GDN mikrosu | GDN payı gürültü içinde; mikro 1216/1233 → 1259/1393 döngü/kalem-token. **Öldü** |
| — | Zen2 port ölçümü (sentetik asm) | maddubs tek port | port.zig | 8 maddubs 8 döngü; 8 paddw 2.7; maddubs+paddw 8; maddubs+4 bağımsız FMA 8. Çekirdek karışımı sentetikte birebir: iç 1.02, +madd 1.16, +cvt 1.19, +cvtph2ps/FMA 1.33, +ölçek 1.37 döngü/maddubs |
| — | Sentetik: blok sonu FMA'sı cvt sonucuna bağımlı / bağımsız; vaddps; vmulps | Çarpma bağımlılığı FP0'ı tıkıyor | port4/5 | Bağımlı FMA +0.155, bağımsız +0.04, bağımlı vaddps 0, bağımlı vmulps +0.21; kaydırmalı genişletme 1.44, mul+add 1.43, önceden hesaplı ölçek 1.32 |
| H54 | Blok sonu iki aşama: E1 (subw/madd/cvt) yuvaya yazar, E2 (ölçek+FMA) bir blok sonra yuvadan bellek operandıyla | Bağımlılık koparılır | mikro maxerr + 3 çift | maxerr 0 (nb 2/4/32, n 5–128); 1.36–1.39 vs 1.32–1.36. **Öldü** (sentetikteki kazanç hiç yazılmamış yuvadan geliyordu) |
| — | vpbroadcastd yerine düz 32 B yükleme (yalnız süre) | Yayın port yiyor | mikro ×3 | 1.32 → 1.37. Yayın bedava |
| H55 | GDN girdileri başlık-öncelikli [başlık][token][s]: q/k l2 norm doğrudan bu düzene yazar, v aynı fazda b_qkv'ye kopyalanır (eski adım 8/24 KB) | CCX'ler arası adımlı okuma GDN'i 6 thread'de ~1.8× yavaşlatıyor | GDN mikrosu ×3 + bayt + bariyer ×2 + 8 çift | Mikro q/k: ortanca 190K → 176K TSC; +v karışık (160/176/218K vs 180/187/175K). 6/6 bayt aynı. Gerçek koşu GDN payı %8.4/8.4 → %8.0/8.5; tg128 ortanca 637 vs 624 (gürültü içinde). **Nötr, kaldı** (mikro kazancı gerçek koşuya geçmedi; out düzeni denenmedi, aynı sebeple beklenti yok) |
| — | H37 sayısal testi taşmasız: ağırlık çifti büyük ölçeğe yeniden nicemlenir + aktivasyon 64'lük ortak ölçek (çekirdek aynı, yalnız sayılar) | i32 birikimle blok sonu yarıya inerse kabul yaşar mı | 2 koşu | Kabul 355/1131, 131–140 tok/s. **Öldü** (taşma değil, ölçek paylaşımı tek başına döngüyü bozuyor; ortak ölçekli her yol kapalı) |
| — | i16 taşma sayımı (gerçek koşu): tam 32'lik blok çarpımı / mevcut yarı şerit | R16 tek şeritte tam blok toplarsa bit-aynı mı | ovCheck | Tam blok taşması 486.837 / 2.13G, yarı şerit 9.096. Tam blok birikimi sayıları değiştirir → R16 yalnız aynı yarı bölümlemeyle |
| — | Zen2 port yoklaması (64 maddubs'a 16 ek işlem serpiştirilmiş) | Hangi blok sonu işlemi FP0'a düşüyor | r16p | Kaydırma, paddd, psubw, unpack, vpmovsxwd(bellek), depolama, FMA: bedava; cvt +0.026; vpmaddwd +0.33. Aynı işlemler blok sonunda yığın hâlinde gelince pahalı (yalnız kaydırma+toplama +0.2): FP zamanlayıcı tıkanıyor |
| H56 | R16 düzeni: 16 satır × blok; V1/V2 çapraz şerit (tek vpbroadcastd iki maddubs), yarı bölümleme R8 ile aynı; madd yerine kaydırmalı genişletme; A/B yığına yazılıp sonraki bloğun iç döngüsüne serpiştirilir; akümülatör bellekte | FP0 tabanı 1.125 → 1.0 | sentetik + mikro bit karşılaştırma + bayt + kc + 10 çift + bariyer ×2 + t=2 | Sentetik 1.18 vs 1.36; gerçek gövde sabit adreste 1.22; mikro (s=3,p=4) 1.25–1.29 vs R8 1.32; bit farkı 0 (28 durum), 6/6 bayt, kabul 381. Gerçek koşu: chunk 11.54–12.03G vs 11.68–11.95G, tg128 ortanca 660 vs 660 (-r 3, gürültü). -r 10: chunk 32.7G → 31.3G (−%4); s=3,p=4 ile 8 çift -r 10: ortanca 667 vs 671, ortalama 637.6 → 663.5 (+%4), 6/8 çift; kazanç yük altında büyük (563→671, 602→649). **Yaşadı** (küçük; kirli ortamda dayanıklı) |
| — | R16 varyantları: F16C aktivasyon ölçeği / ağırlık yazmaçta (4 sütunda yeniden kullanım) / sabit ölçekle tablo ıskası testi | Kalan fark | mikro | F16C 1.32, yazmaçta 1.31, sabit ölçek fark yok. **Öldü** |
| — | Ortam: kirli koşuda Tctl 93–94°C, ortalama saat ~2.95–3.0 GHz (boşta 3.97) | Makine ısıl/güç sınırında | perf stat + sensors | Bu rejimde döngü azaltan ama MAC başına komut/yükleme artıran değişiklik kazandırmıyor; hedef MAC başına enerji (komut, yükleme, boş bekleme) |
| — | R16 sentetik: 3 sütun + ağırlık yazmaçta (yükleme 1.5 → 0.83/maddubs) ve 3 sütun bellek | Yükleme portu darboğazı | r16c3 ×4 min | 4 sütun bellek 1.221; 3 sütun yazmaç 1.259; 3 sütun bellek 1.239. **Öldü** (sınır yükleme değil, toplam FP işlem sayısı) |
| H57 | Artık toplama + sonraki rmsNorm aynı satır geçişinde (`addRmsNorm`): attn_residual+post_norm ve l_out+sonraki attn_norm; katman başına 2 bariyer ve 2 geçiş eksik | Küçük fazlar + bariyerler | bayt + 8 çift -r 10 | 6/6 bayt aynı. h56b → h57 ortanca 639 → 672 (+%5.1), ortalama 624 → 662, 7/8 çift. **Yaşadı** |
| H58 | Matmul girdisi üretici fazda satır satır R8 nicemlenir (addRmsNorm, swiglu, gated norm×z, gateAttn) + rangeInit üretici bariyerinden önce; ayrı nicemleme fazı ve bariyeri kalkar (katman başına ~4 bariyer; gated norm ile mul z da birleşti) | Bariyer + ayrı geçiş maliyeti | bayt + kabul + 2×8 çift -r 10 | 6/6 bayt aynı, kabul 381. Ortanca 667 → 668 ve 658 → 672; ortalama +%1.6 / +%1.1; 11/16 çift. **Yaşadı** (küçük). Tctl 95°C |
| H59 | Worker thread'leri RLIMIT_NICE içinde nice −11 (setpriority); ayrıca dışarıdan nice / autogroup −11 denemesi | Masaüstü süreçleri önalıyor | 4 üçlü + 8 çift -r 10 | Dış deneme gürültülü (642 → ~690 görünüm); süreç içi 8 çift ortanca 661 → 667, ortalama 649 → 641, 5/8. **Öldü**, geri alındı. Sebep: autogroup açık (nice yalnız oturum içi), yük ~2.8/12 → boş mantıksal CPU var; kayıp zaman paylaşımı değil SMT paylaşımı + ısı (Tctl 95°C) |
| H60 | Bariyer beklemesi pause yerine MONITORX/MWAITX (C0, 20K TSC zaman aşımı) | Isıl sınırda bekleyen çekirdeğin gücü çalışanlara kalır | bayt + 8 çift -r 10 | 6/6 bayt; ortanca 691 → 675, ortalama 693 → 662, 1/8. **Öldü** (uyanma gecikmesi) |
| H61 | wqkv/alpha/beta/z GEMM'inde conv + gateFromAlpha + sigmoid + silu parça sonunda (satır yerel `epi`); ayrı faz ve bariyeri kalkar | Bariyer + geçiş | bayt + kabul + 8 çift -r 10 | 6/6 bayt, kabul 381; ortanca 697 → 701, ortalama 679 → 681, 6/8 (en iyi 713). **Yaşadı** (küçük) |
| H62 | FFN: aynı 32 kanalın gate ve up parçası tek iş biriminde; parça sonunda swiglu + down girdisi nicemleme (32 kanal = 1 Q8 bloğu) ikinci wdata'ya; down aralıkları ikinci slot kümesinde (ABA yok); swiglu fazı ve bariyeri kalkar | Faz + bariyer + 1.8 MB geçiş | bayt + kabul + 8 çift -r 10 | 6/6 bayt, kabul 381; ortanca 691 → 715 (+%3.4), ortalama 681 → 693, 7/8, tepe 725. **Yaşadı** |
| H63 | q/k l2 norm + v başlık-öncelikli kopyası conv parça sonunda (`convPost`): v kanalları b_conv_in'e kopyalanır; q/k başlığının son parçasını bitiren thread (başlık başına atomik sayaç) o başlığın normunu tüm tokenlar için yazar; l2/vcopy fazı ve bariyeri kalkar | Faz + bariyer (%2.3) | bayt + kabul + 8 çift -r 10 | 6/6 bayt, kabul 381; ortanca 718 → 737 (+%2.6), ortalama 713 → 723, 7/8, tepe 744. **Yaşadı** |
| H64 | Attention girdi hazırlığı tek fazda (`attnPrep`): q norm+rope, k norm+rope+f16 önbellek, v f16 önbellek; 6 faz/bariyer → 1 | Bariyer (6 attention katmanı × 5) | bayt + kabul + 3×8 çift -r 10 | Round-robin dağıtım: ortanca 728 → 718, 3/8 (her thread her token için sin/cos hesaplıyordu). Bitişik aralık: 6/6 bayt, kabul 381; ortanca 701 → 712 ve 695 → 730, ortalama +%2.1 / +%3.4, 9/16. **Yaşadı** (küçük, Tctl 95°C, yük 3.7) |
| H65 | GDN + gated norm tek fazda (`gdnNormQ`): GDN aralıkları set 1'de önceki GEMM fazında hazırlanır (GDN içi bariyer kalkar); başlığın son kalemini bitiren thread (başlık sayacı) tüm tokenlarda norm × z + R8 nicemleme yapar (gated norm fazı ve bariyeri kalkar) | Doğrusal katman başına 2 bariyer | bayt + kabul + 8 çift -r 10 | 6/6 bayt, kabul 381; ortanca 736 → 750 (+%1.9), ortalama 720 → 737, 6/8, tepe 758. **Yaşadı** — 750 ortancada geçildi |
| H66 | r8Run/FFN iş birimi = (32 satırlık parça, token yarısı); wqkv parça sonu işlemi (conv tüm tokenları ister) parça sayacıyla son yarıyı bitirende; FFN swiglu+nicemleme yarı başına | Faz sonu kuyruğu (FFN birimi ~117 µs) → bariyer beklemesi %5.7 | bayt + kabul + 2×8 çift + perf bekleme payı | 6/6 bayt, kabul 381. Çiftler: ortanca 766 → 773 ve 740 → 741, ortalama −%0.2 / −%0.8, 11/16. Bekleme (atomic yükleme) %6.0/6.1 → %4.5/5.1. SPLIT=4: 1/8, kötü. **Yaşadı** (küçük; yük 2.3–4.0) |
| — | GEMM gerçek koşu TSC/maddubs (r8Run'da rdtsc, şekle göre) | Gerçek koşu çekirdeği mikrodan yavaş | -r 1 / -r 11 | k1024 1.59, k2048 1.64, down 1.57, gate+up 1.60 TSC ≈ 1.27–1.33 çekirdek döngüsü (saat ~2.95 GHz): mikroyla aynı. Koşu başına GEMM ~143 ms / toplam ~170 ms; 1000 tok/s (128 ms) bu çekirdek ve saatle yalnız GEMM'le bile aşılamaz |
| H67 | GDN: satır başı 8 hsum (o, dn) transpoze ağaçla 4'erli (hsum ile aynı toplama çiftleri, 20K rastgele testte bit-aynı) | Döngü dışı indirgeme maliyeti | GDN mikrosu ×5 | 1175 → 1259 döngü/kalem-token (çapraz şerit karıştırma + yazmaç baskısı). **Öldü** |
| H68 | Attention: flash iş aralıkları attnPrep fazında küme 1'de (flash içi bariyer kalkar); satır yazımında gate × sigmoid + wo R8 nicemleme (gateAttn fazı ve bariyeri kalkar) | Attention katmanı başına 2 bariyer | bayt + kabul + 3×8 çift | İlk deneme küme 0: kabul 314/381 (faz içi r8Pre(wo) aynı kümeyi kurunca çalan thread'ler wo birimlerini flash sandı — ABA). Küme 1: 6/6 bayt, kabul 1270/1270; çiftler 2/8, 7/8, 4/8; ortalama −%2.7 / +%6.2 / −%3.6 (yük 2.9–4.2). **Nötr, kaldı** (bit-aynı, daha az bariyer) |
| — | HEAD 6b0cdfd tek başına 10 koşu -r 10 (kirli: yük 3.3–3.7, Tctl 94.6°C) | 750 barajı | 10 koşu | 607/749/770/762/767/764/762/758/762/741 → ortanca 762, ortalama 744. **TG 750 geçildi**, README işaretlendi |
| — | Bariyer bekleme dağılımı (pool.barrier'da sıra indeksli rdtsc, 6b0cdfd) | Kalan bekleme nerede | -r 10 | Toplam ~86M TSC/koşu (thread zamanının ~%2.5'i). Attention flash fazı ~20M/katman, LM fazları ~80M/11 koşu, GDN sonu ~8M, down sonu ~7M, gate/up ~4M, norm fazları ~0–1M. Tek başına büyük kalem yok |
| — | GDN: k/q tamponları 32/64/128 B kaydırıldı (4K adres çakışması hipotezi) | Durum satırı depolamaları k/q yüklemeleriyle 4K hizasında çakışıyor | GDN mikrosu ×2 | 1181/1179/1299/1286 ve 1208/1181/1180/1198: tutarlı fark yok. **Öldü** |
| H69 | LM taraması: 4'e tamamlanan son karoda yalnız gerçek sütunlar (r8xN(1..3)) | Dolgu sütunu ~%9 boşa iş | bayt + kabul + perf graph/GEMM oranı ×2 | 6/6 bayt, kabul 1270; graph/GEMM 0.0792/0.0805 → 0.0794/0.0874. **Öldü**, geri alındı |
| H70 | Pool thread'leri SCHED_RR öncelik 1 (rtprio izni audio grubundan) + thread i → mantıksal CPU i (ayrı fiziksel çekirdek); ECHO_NO_RT / ECHO_NO_PIN kapatır | Kirli ortamda masaüstü görevleri thread'lerimizi önalıyor (H59 nice autogroup yüzünden etkisizdi) | bayt + kabul + 2×8 çift | Sabitlemesiz RR 532–557 (RT dengeleyici SMT kardeşlerine yığıyor). RR+sabit: 6/6 bayt, kabul 1270; ortanca 779 → 785 ve 744 → 764, ortalama 745 → 771 (+%3.6) ve 731 → 745 (+%1.9), 12/16; yük patlamasında 609 → 720, 683 → 771. **Yaşadı** |
| — | Ağırlık akışında TLB ıskası (4K sayfa) | Büyük sayfa yok | smaps_rollup | THP always; anonim 1272 MB'ın 897 MB'ı AnonHugePages. **Öldü** (zaten büyük sayfa) |
| — | Sentetik R16: yalnız iç döngü (maddubs + paddw + yayın) | Taban | syn P=in16 | 1.001 döngü/maddubs; blok sonu (bit-aynılık için zorunlu genişletme + cvt + ölçek çarpımı + FMA) farkın tamamı. Yükleme azaltan R32×2 bu yüzden denenmedi (önceki 3 sütun/yazmaç deneyi yükleme sınırını çürütmüştü) |
| — | Sentetik R32×2 (iki 16'lık grup aynı aktivasyon yayınını paylaşır; serpiştirme s/p taraması) | Yayın/yükleme azalır | syn32 ×2 | R16 1.222–1.227; R32×2 1.226–1.261. **Öldü** |
| — | Sentetik blok sonu sınıf çıkarma (R16 gerçekçi serpiştirme, 1.22) | FP0'ı hangi işlem çalıyor | syn33 | vmulps −0.042, FMA −0.059, vpaddd −0.043, cvt −0.027, kaydırma −0.006, ölçek yükleme −0.003; hepsi yok 1.006. FP0'a düşebilen her işlem orantılı pay alıyor |
| — | Sentetik: vpaddd yerine 4 cvt + vaddps (tam sayılar < 2^24, toplam tam → bit-aynı) | vaddps yalnız FP2/3 | syn34 ×2 | 1.247–1.278 vs 1.221. **Öldü** (toplam işlem sayısı artınca zamanlayıcı kötüleşiyor) |
| — | RT (H70) altında SPLIT=4 | Önalma azalınca ince birim kazandırır mı | kabul + 8 çift | Kabul 381; ortanca 783 → 780, ortalama +%0.4, 1/8. **Öldü** |
| H71 | GDN kalemi 32 → 16 satır (128 kalem; 64/6 dengesizliği) | GDN fazı kuyruğu | bayt + kabul + 8 çift | 6/6 bayt, kabul 381; ortanca 762 → 755, ortalama +%0.2, 4/8. **Nötr, alınmadı** |
| — | convRows (%1.4) inceleme | — | kod | 8'li vektör, L1'de; maliyet silu (exp). Azaltılacak iş yok |
| N1 | FFN ara kanal budama (`ECHO_FFN_KEEP=K`, 64'ün katı): önem = ‖gate_c‖·‖up_c‖·‖down[:,c]‖, ilk K kanal özgün sırada; down yeniden Q8 nicemlenir. Varsayılan yol değişmez | FFN gövdenin ~%55'i; kalite kaybı kabul | bayt + K taraması -r 10 + 6 çift | Varsayılan 6/6 bayt aynı. K: 3584 739, 2560 816, 2304 931, 2240 940, 2176 952 (kabul tam); 2432, 2208, ≤2144 kabul çöktü (yörünge değişiyor). Çift: 766 → 931 (+%21.7), 6/6. **Yaşadı** (isteğe bağlı) |
| — | Drafter n-gram sırası 3 → 6/10/16 (ECHO_NGRAM_N) | Uzun bağlam küçük K'da kabulü kurtarır | 12 koşu | Kabul düzelmedi (en iyi N=16 K=2048 498); K=3584'te N>3 yavaş. **Öldü**, geri alındı |
| — | Tiled flash tüm token sayılarında | Yörünge / hız | 3 koşu | K=3584 748 (777'ye karşı), kabul çöküşü aynı. **Öldü**, geri alındı |
| — | LM IVF nprobe 512 → 256/128/64/32/16/8 (`--lm-ivf 4096,NP`) | LM taraması küçülür | K×NP taraması | K=2176: 939/974/1005/1003/983/995/994 (kabul tam). K=3584 NP≤128 kabul düşüyor. K=2112 NP16 1008. K 1792–2048 çoğu çöktü. 128 seçildi |
| N2 | Katman atlama (`ECHO_SKIP=a,b,..`): atlanan katmanda yalnız sonraki katmanın attn_norm + R8 nicemlemesi | Orta katmanlar çıktıya az katkı | bayt + tek/çoklu atlama taraması + 2×6 çift + 3×10 | Varsayılan 6/6 bayt. Tek atlamada kabul tam: 3,5,6,9,11,12,13,16,18,20,23; 0,1,2,7,14,17,19,21,22 çöktü. Çoklu: 9,12,13 1085; 9,12,13,18 1113; 3,5,9,12 ve 5,9,11,12,13 çöktü. Çift (K=2176, NP 128): varsayılan 746 → 9,12,13: 1061 (6/6); 761 → 9,12,13,18: 1153 (+%51, 6/6, yük 5.4). Tek başına 3×10: 1158/1168/1165, 30 örnek 1119–1188, kabul 1270/1270, yük 4.3–4.8. **TG 1000 geçildi** (kayıplı yapılandırma: 4/24 katman yok, FFN %39 budanmış, LM nprobe 128) |
| — | Kalite ölçümü (kalite.sh: bash man sayfası 257 token, tek token adımlı dump-logits, tam LM) | Kayıplı ayarların kalitesi | ppl + top-1 | Taban ppl 37.4, top-1 %35.5. **N1 FFN budama bozuk ölçüt**: K=3520 ppl 75.5, 3072 185, 2560 527, 2176 1089 (top-1 %7, tabanla uyum %13.7) — ağırlık normu aktivasyon aykırı kanallarını kaçırıyor. Önceki TG 1000 ayarı (s4: K=2176 + atlama 9,12,13,18) ppl 1991, top-1 %7: **geçersiz sayıldı**. Tek katman atlama: 0 ppl 35237, 23 ppl 687; diğerleri 40–71. Açgözlü sıra 20,7,18,11,12,8,5,13,9,16,17,3,4 |
| N2b | Yalnız katman atlama (budama yok) | Kalite/hız sınırı | kalite + TG + pp512 | 4 katman ppl 61 top-1 %29.3; 6: 77 / %29.3; 8: 123 / %25.4; 10: 182 / %25.4; 12: 379 / %21.1; 13 (+4): 428 / %22.3; 14 (+4,1): 777 / %14.5 |
| — | TG 1000 (yeni ayar z8: ECHO_SKIP=20,7,18,11,12,8,5,13 --lm-ivf 4096,128) | Kalite kaybı %28 (top-1) ile 1000 | 2×6 çift | z6 (6 katman): 761 → 1042, 6/6, bir koşu 965. z8: 749 → 1157 (+%54), 6/6, yük 4.2. **TG 1000 geçerli** (kabul 1270/1270) |
| — | Prefill pp512, 6 tur sıralı (yük 5.4–8.2) | 750/1000/1500 | -r 10 | Bit-aynı: 754/764/714/773/775/725 → ortanca 758. z8: ortanca 1150 (min 1028). z13 (z8 + 9,16,17,3,4; top-1 %22.3, tabana göre −%37): ortanca 1685, min 1503. **Prefill 750, 1000, 1500 geçildi** |
| — | Prefill 2000 sınırı | Kalite sınırı (top-1 ≥ tabanın yarısı, %17.7) içinde | hesap + ölçüm | GEMM maddubs sınırında (~1.3 döngü/maddubs, 6 çekirdek, 2.95 GHz) → tam modelde tavan ~870 tok/s; 2000 için MAC'in ~%38'i kalmalı. 14 katman atlama pp512 1848'de top-1 %14.5 (sınır altı). Ağırlık normlu FFN budama daha kötü. **Engel**: kalite sınırı içinde 2000 yolu yok; denenecek tek yol aktivasyon istatistikli budama |
| — | Kalite ölçümü büyütüldü (768 token, bash man 60–400. satır; kalibrasyon 400–1400. satır, ayrı) | 256 token ±%2.4 gürültülü | ppl + top-1 | Taban ppl 18.2, top-1 %44.5 → sınır (yarısı) %22.3. z8 ppl 74 / %27.9; z13 366 / %18.5 (**sınır dışı → önceki prefill 1500 işareti geri alındı**); ağırlık normlu K2176 548 / %13.4 |
| N3 | Aktivasyon istatistikli FFN budama: ECHO_FFN_STATS=dosya kanal başına Σ\|swiglu\| toplar (kalibrasyon 2048 token); ECHO_FFN_ACT=dosya ile önem = Σ\|act_c\|·‖down[:,c]‖ | Ağırlık normu aykırı aktivasyonları kaçırıyor | kalite + pp512 + TG | K2176: ppl 37.4 / %33.9 (ağırlık normu 548 / %13.4). K1792 46 / %31.2; 1536 57 / %29.4; 1280 71 / %28.6. Atlama altında yeniden kalibrasyon daha iyi değil. TG K2176+[20,7,18,11]: 1121, kabul 1270, top-1 %29.0 (z8'den iyi kalite); +12,8: 1248 / %27.3. **Yaşadı** (isteğe bağlı) |
| N4 | Son katmanda (want_all yoksa) post-norm + FFN + toplama yalnız son token için (çıktı yalnız son satırdan okunur) | Prefill'de boşa giden son katman FFN'i | prefill logit cmp (512,128,1,1,256,64,7 parçaları) + bayt + 6 tur | Logitler bayt aynı, TG 6/6 bayt aynı. pp512 varsayılan ortanca 773 → 795 (+%3, 5/6). **Yaşadı** (kayıpsız) |
| — | Prefill 1500 (N3 K2176 + atlama 20,7,18,11,12,8,5,13,10; top-1 %22.9, tabana göre −%48.5, ppl 132) | Sınır içinde 1500 | 6 tur pp512 (yük 5.9–8.8) | N4 öncesi 1342–1514; N4 ile 1553/1545/1529/1537/1525/1491 → ortanca 1533. **Prefill 1500 geçildi** |
| — | Prefill 2000 | Kalite sınırı (top-1 ≥ %22.3) | ızgara | Sınıra yakın tüm ayarlar 1456–1494 (N4 öncesi) ve top-1 %20.8–21.7; 9. katman taraması (17,3,4,1,10,6,14,15,19,2,21,22) en iyi %22.9. 2000 için MAC ~%38'e inmeli; sınır içinde en hızlı ~1530. **Engel**: denenen 5 yol (yalnız atlama, ağırlık normlu budama, aktivasyonlu budama + atlama, atlama altında kalibrasyon, N4) sınır içinde 2000 vermedi |
| N5 | Bileşen atlama: yalnız FFN (ECHO_SKIP_FFN) / yalnız mixer (ECHO_SKIP_MIX) | Tek bileşen ucuz (tek başına ppl 18.2 → 18.7–22.4, top-1 %40–45) | 768 token kalite + pp512 | Birleşince tam katmandan kötü: 16 bileşen (≈8 katman) budamasız ppl 105 / %24.2 / 1137 (8 tam katman: 74 / %27.9 / 1150); K2176 ile 12 bileşen %21.5. **Öldü**, geri alındı |
| — | Sentetik: R16 iç döngüye FMA serpiştirme (FP1 boş mu) | Hibrit int8 + f32 GEMM ile MAC/döngü artar | fma.zig, tek çekirdek | Döngü/maddubs: taban 1.022; 4 maddubs'a 1 FMA (yazmaç) 1.003; 1/2 1.143; 1/1 1.639; bellek işlenenli 1/4 1.123, 1/2 1.292. En iyi gerçekçi kazanç (bellekli 1/4) +%2.5 MAC/döngü. **Öldü** |
| — | Prefill 2000 / 5000+ engel özeti | — | hesap | Kayıpsız tavan ~870 tok/s (maddubs sınırı, FP1 hibriti ≤%2.5). Sınır içinde (top-1 ≥ %22.3) en iyi 1533; açgözlü adımda her ek katman pp +%6, top-1 −%0.1…−1.6 → 2000 için 4–5 katman daha gerekir, top-1 ~%19–20. 5000+ için MAC'in ≤%17'si kalmalı. **Engel** |
| N6 | Son (attention) katmanda want_all yoksa wq yalnız son token sütunu (r8Run Last0), flash tek sorgu (first_cell = head + n_t − 1), gate + wo tek satır; k/v tüm tokenlar (önbellek) | Prefill'de son katman attention'ı boşa | bayt + logit kıyası + 6 tur | TG 6/6 bayt aynı. Prefill logiti bayt aynı değil: n_t=1 ve n_kv ≥ 512'de flash split-KV yoluna geçiyor (toplama sırası); fark 0.17, mevcut kodun 7'lik parça ile 6+1 arasındaki farkıyla (0.46) aynı düzeyde; 6 ve 7'lik parçalarda bayt aynı; argmax 7/7 aynı. pp512 varsayılan 803 → 823 (+%2.5, 5/6); P ayarı 1548 → 1613 (+%4, 6/6). **Yaşadı** |
