# Proje Tanımı
Proje: **echo-core** — Zig ile yazılmış, yalnız CPU'da çalışan çıkarım motoru.

---

# Her Oturuma Başlarken Okunması Zorunlu Belgeler
- MUST read: `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/q8-tg-yontem-haritasi.mindmap.html`
- MUST read: `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/q8-tg-yontem-haritasi.mindmap.md`
- SHOULD read (yalnızca eski bağlam/tarihçe gerektiğinde): `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/docs/eski-haritalar/q8-tg-yontem-haritasi-2026-11-06.mindmap.html`

---

# Kesin Kurallar (Must / Must Not / Should / Should Not)

## 1. Model ve Mimari Kısıtlamaları
- MUST use only: `/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf`. Kullanıcı açıkça override vermediği sürece başka model KULLANILAMAZ.
- MUST NOT change: Model mimarisi, quantizasyon türü ve katman sayısı değiştirilemez.
- MUST NOT: Hız kazanmak uğruna katman atlama (layer skipping), FFN budama (FFN pruning) veya bit düşürme KESİNLİKLE YAPILAMAZ.

## 2. Çalışma Ortamı ve Donanım Ölçümü
- MUST use: Yalnızca **Zig**, yalnızca **CPU**.
- MUST run benchmark: Ölçümler kesinlikle **1 çekirdek ve 1 thread** üzerinde yapılmalıdır:
  - Komut formatı: `taskset -c 0`
  - Ortam değişkeni: `ECHO_THREADS=1`
  - Argüman: `-t 1` (veya `-ç 1`)
  - Atlamalar: `ECHO_SKIP=` boş olmalıdır.
- MUST NOT: GPU veya çoklu çekirdek/thread benchmark'a dahil EDİLEMEZ.

## 3. Raporlama, Metrikler ve Tablo Standardı
- MUST use: Hız birimi olarak YALNIZCA `tok/s` kullanılmalıdır.
- MUST NOT use: Hız raporlamasında `ms` (milisaniye) KULLANILAMAZ.
- MUST format: Her performans sonucu net bir Markdown tablosu olarak sunulmalıdır:
  - Baseline ve yöntemler/hileler ayrı satırlarda yer almalıdır.
  - Kolonlarda `tg32 (tok/s)`, `tg128 (tok/s)` (veya ilgili `pp128`, `pp512`) ve **Kat Artışı** (Örn: `2.25x`) yer almalıdır.
- SHOULD provide: Tablo altında sadece 3-4 maddelik sade ve net bir özet (ne kadar hızlandırdı, ne işe yaradı) verilmelidir.
- MUST NOT: Axiom personası, edebiyat, felsefe, robotik rol yapma ("özümüz gereği...", "Axiom olarak...") KESİNLİKLE YASAKTIR.
- MUST NOT: Anlaşılmayan akademik boş laflar, gereksiz matematiksel terimler ("alfa 1.000", "entropi katsayısı" vb.) KULLANILAMAZ.

## 4. Doğrulama ve Kararlılık
- MUST NOT present: Derlenmemiş, 1 çekirdekte bizzat ölçülmemiş ve çıktısı doğrulanmamış hiçbir sonuç veya kod kullanıcıya SUNULAMAZ.
- MUST NOT regress: Bir kez geçilen hız, elde edilen başarı ve geçerli ayarlar geri alınamaz, regresyona uğratılamaz.

## 5. Kod Tabanı Mülkiyeti ve İşbirliği
- MUST respect: Kod tabanı tamamen kullanıcıya aittir.
- MUST NOT push code: Ortaklaşa mutabık kalınmayan hiçbir kod üretime gidemez ve kod tabanına dahil edilemez.
- SHOULD consult: Kod yazmadan veya değiştirmeden önce kullanıcı ile fikir alışverişi yapılmalı, ortak karar alınmalıdır.
- MUST remember: Kullanıcının son talimatı, önceki tüm talimatları geçersiz kılar.

## 6. Git ve Harita Kayıt Süreci
- MUST NOT: `git push` KESİNLİKLE YASAKTIR.
- SHOULD commit: Commit serbesttir (kullanıcı onayı ve ortak karar doğrultusunda).
- MUST update map: Yapılan her geçerli iş ana yöntemi haritasına tarihli ve elde edilen `tok/s` değeri ile birlikte eklenmelidir.

---

# Zig Upstream MCP Kullanımı
- MUST use: Zig ile ilgili herhangi bir soru, araştırma, dokümantasyon arama, issue/PR takibi için `zig-upstream` MCP araçları KULLANILMALIDIR.
- MUST try first: `zig_upstream__search` (doğal dilde sorgu).
- SHOULD use: Dokümantasyon için `zig_upstream__docs`, issue/PR için `zig_upstream__issues` ve `zig_upstream__prs`.
