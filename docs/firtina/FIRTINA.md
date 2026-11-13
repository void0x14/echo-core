# Fırtına — tarif ve keşif kaydı

Kaynak: kullanıcının çizimi (`user-documents/ChatGPT-Ryzen 5 3600-Echo-Core-CPU-ONLY-inferecene-20261104-1207_yeniilerlemeler.pdf`, s. 47) ve 11 Kas 2026 düzeltmeleri.
Görselleştirme: `docs/firtina/firtina-teknik.html` (tarayıcıda aç; oynat/durdur, ters yön, zaman çubuğu).

## Tarif (kullanıcının terimleriyle)

- **Oda** işlemcinin kendisi.
- Odanın içinde kuantum cep odacıkları / kapakçıkları var (sayı hayali, sonsuz da olabilir). Bunlar duvar değil; su hücresi, su damlası, kar tanesi gibi. İsterlerse birleşir, büyür, çoğalır, küçülür; agar.io'daki gibi kendini parçalara bölüp dışarı atar.
- **Sol ve sağ** aynı işi yapıyor; ikisi de aynı odada, ayrı kuantum cep odacıklarında / kapakçıklarında. Bu **simetri**: biri olmadan öbürünün anlamı yok. "Kanat değiştirme" diye bir şey yok.
- **Komutan** (Core 1) üç emri ortaya konuşur.
- **Ordu** (Core 2, Core 3) farklı bir tarifle bir kimliğe sahip; random görünen ama işlenmesi gereken kaos cep evrenlerini ortadaki emirden kendine çeker.
  - Ordu **kolektif bilinç**tir, bireysel değil: stadyumda biri zıplar, sonra biri daha… bir anda tuhaf ama çok sağlam, sistematik bir koreografi.
  - Ordu **beklemez**, her daim tetiktedir; boş durursa düşman içeri girer.
- **Fırtına hücresi** birden belirir. Yeşil (sol) ve mor (sağ) aynı türdendir; renk yalnızca sol/sağ ayrımı için.
  - Fırtına rüzgârın kendisi, atomlardan oluşur; katı değil, **yol değil**.
  - Fırtınada yüklü kinetik enerji var. Ordu ile karşılaşınca karşıt atomlar sayesinde birbirini çekerler (statik elektrik, parmak ile yüzey arasındaki akım gibi). Fırtına ordudaki her şeyi hızla içine çeker; yapısı böyle. Çizimdeki pembe bu **enerji**dir; dal değil.
  - Fırtına, çevresindekini dışarı atma eğilimiyle, uçlarındakileri **kırmızı ters üçgene** taşır, fırlatır.
- Sağda aynı işlem simetriğinde yaşanır.
- İki taraftan gelenler ters üçgende birleşir: iki fırtına hücresi birleşince **kasırga** olur.
- Kasırganın yerdeki sıfır noktasında inanılmaz bir **vakum** var; kazanımlar bir anda yerçekimine doğru çekilir, son sürat dalar. Damla damla düşme diye bir seçenek yok; yerçekimi her an daha da hızlandırır.
- Düşüş **Infinity Fabric**, çakılma **IO die**.
- Fırtına hücresi **ters yönde de** çalışır.
- Sıra ya da paralellik yok: "hepsi aynı anda" denmesiyle anında olur ve biter.

## Ölçülen (bu işlemci, 11 Kas 2026)

- İşletim sistemi: Core 1–3 = cpu0–2 (ikinci thread'leri cpu6–8), Core 4–6 = cpu3–5 (cpu9–11).
- Çekim ölçümü bu iki simetrik yarıyı ayrı ayrı gösteriyor: `echo-core-zig/benchmarks/kesif/c2c-2026-11-11.md`, `firtina-2026-11-11.md`. Programlar: `echo-core-zig/tools/kesif/`.

## Anlamlandıramadıklarımız (çöpe atılmadı)

- **Ayna metaforu** — simetriden ayrı tutuluyor ("aynada simetri yoktur, senin aynın vardır"); tekniği güçlendirebilir, şimdilik anlamlandıramadık.
- **Üç emrin içeriği** — keşif aşamasındayız; karar verilmedi.
- **Ters yönün nasıl başladığı** — görselde akış yalnızca tersten oynatılıyor.

## Reddedilen eşleştirmeler (tekrar yapılmayacak)

- Fırtınayı tek çekirdeğe sıkıştırmak ("1 çekirdek kuralı" gerekçesiyle).
- Yeşil bulut = L3; pembe = ordunun SMT kardeşleri; atlı = ikinci thread.
- "Kanat değiştirmek pahalı" çerçevesi.
- Ordu = her askerin `lock xadd` ile parça kapması (bireysel mantık).
- Ordu = işi bitince `pause` ile bekleyen asker.
- "Tek tek çekim" ve "damla damla dalış" karşılaştırmaları (çizimde öyle bir ok yok).
- Üretici belgesine dayanan donanım iddiaları (Infinity Fabric genişliği, L3 davranışı, iki grubun bağlantısı).
- Fırtınaya "yol" demek.
