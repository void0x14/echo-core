# Sezaryen Modu — Mimari ve Deney Planı

Bu dosya, "bana göre" önerilen mimariyi ve bu repoda koşulacak deney sırasını yazar.
Kırmızı çizgi: model dosyası ve quantizasyon değişmez; uyarlama motorun içinde kalır.

---

## 1. Echo Core'un üç modu

### MODE L — Lossless
- Hedef modelin dağılımı korunur.
- Her kabul edilen token, hedefin kendi doğrulamasından geçer.
- Etiket: lossless.
- Kullanım: genel metin, sohbet, güvenilirlik gereken işler.

### MODE T — Task
- Hedef dağılım birebir korunmaz; görev doğruluğu korunur.
- Örnek: kod testi, JSON şeması, matematik cevabı.
- Etiket: task.
- Kullanım: yapılandırılmış çıktı, kod, doğrulanabilir işler.

### MODE D — Deterministic
- Devamın dışarıdan kesin olduğu bölgeler.
- Grammar, JSON anahtarı, sabit şablon, retrieval copy.
- Model seçimi kaldırılır; blok halinde eklenir.
- Etiket: deterministic + task veya deterministic + lossless (duruma göre).
- Kullanım: JSON, DSL, kod şablonu, kaynak metin kopyası.

Üç mod aynı sonuç tablosuna karışmaz. Her deney hangi modda olduğunu yazar.

---

## 2. Runtime ABI

Echo Core, modelden bağımsız bir orkestratör olacak. Model başına küçük bir capability adapter olacak.

### 2.1 Çekirdek çağrılar

- **capabilities():** Modelin katman sayısı, hidden dim, tokenizer, RoPE/position düzeni, cache düzeni, izin verilen hook noktaları.
- **propose(prefix, k):** Aday token/özellik ağacı üret.
- **verify(prefix, tree):** Hedef modelle doğrula; kabul edilen uzunluğu döndür.
- **commit(accepted):** Kabul edilen tokenları cache'e yaz.
- **rollback(rejected):** Reddedilen tokenların cache/pozisyon etkisini geri al.
- **inject(layer, vector):** Seçili katmana dış vektör bas (modele göre shape kontrolü).
- **snapshot() / restore():** Deney ve hata ayıklama için cache durumu.

### 2.2 Capability adapter alanları

- model ailesi ve sürümü
- katman sayısı
- hidden dim
- head / KV head düzeni
- norm/residual sırası
- RoPE tipi, position id kuralı
- tokenizer ve vocabulary
- cache düzeni (KV, recurrent state)
- güvenli injection noktaları (input embedding, layer L residual, LM head)
- desteklenen modlar (L/T/D)

Adapter model ağırlığı değiştirmez. Sadece "nereden, hangi shape ile, hangi kurala göre" bilgisini verir.

---

## 3. Hook noktaları

### 3.1 Input embedding
- ESP tarzı mask token injection.
- Token ID yerine float vektör kabul eden bir yol gerekir.
- Mevcut motorda embedRow yalnız token ID alıyor; yeni bir "embedding injection" yolu açılmalı.

### 3.2 Layer L residual
- Hidden Transfer / FIRP / Lyanna tarzı hidden state injection.
- h_l = h_l + delta_h.
- delta_h, o katmanın hidden dim'ine uymalı.
- Eğitimsiz rastgele vektör OOD olur; küçük bir projeksiyon veya ölçek gerekir.

### 3.3 LM head
- Token-info injection (Lyanna).
- Logit bias veya token embedding bias.
- IVF LM head ile uyumlu olmalı.

### 3.4 KV cache
- Kabul edilen tokenların KV'si commit.
- Reddedilenlerin KV'si rollback.
- Ağaç doğrulamada her dalın pozisyonu ayrı.
- Mevcut Context.head yalnız ileri gidiyor; truncate/rollback gerekir.

---

## 4. Deney fazları

### Faz 0 — PLD / REST (hedef 2x)
- **Ne:** Prompt ve üretilen geçmişte n-gram ara; tek zincir aday üret; hedef model tek forward'da doğrula.
- **Neden ilk:** Eğitim yok, model değişmiyor, en hızlı prototip.
- **Gereken mühendislik:**
  - n-gram trie / lookup.
  - Tek zincir doğrulama (ağaç yok).
  - KV rollback/truncate.
  - Tokenizer sınır kontrolü.
- **Başarı ölçütü:** Özetleme/QA/kodda >=1.8x, çıktı kalitesi gözle görülür bozulmadan.
- **Durma koşulu:** 1.3x altındaysa n-gram kalitesi bu modelde yetersiz; Faz 1'e geç.

### Faz 1 — ESP (hedef 1.2-1.5x)
- **Ne:** Prompt embedding ortalamasını mask token olarak ekle; gelecek tokenları tek forward'da sor.
- **Neden:** Eğitimsiz, model-agnostik, input embedding seviyesinde "sezaryen".
- **Gereken mühendislik:**
  - Embedding injection yolu.
  - Mask token pozisyonları ve tree attention.
  - Ağaç budama.
- **Başarı ölçütü:** >=1.2x, lossless.
- **Durma koşulu:** 1.1x altındaysa ağaç yönetimi maliyeti kazancı yiyor.

### Faz 2 — Lyanna-benzeri hidden-state reuse (hedef 2-3x)
- **Ne:** Hidden state seviyesinde draft; token bilgisini örneklemede enjekte et; reddedilen hidden state'i yeniden kullan.
- **Neden:** C'yi düşürür, alfa'yı korur; C'nin en somut örneği.
- **Gereken mühendislik:**
  - 1 katmanlı draft model (eğitim).
  - Token-info embedding (düşük rank).
  - Hidden-state reuse ve re-sampling.
  - Doğrulama füzyonu.
- **Başarı ölçütü:** >=2.5x, lossless; eğitim maliyeti kabul edilebilir.
- **Durma koşulu:** Eğitim verisi/maliyeti bu repoda karşılanamıyorsa, Faz 2'yi dış kaynak/model ile sınırla.

### Faz 3 — DFlash-benzeri blok drafter (hedef 4-6x, GPU)
- **Ne:** Hedefin çok katmanlı özelliklerini küçük blok-difüzyon drafter'a KV ver; bloğu paralel üret.
- **Neden:** 6.1x'e kadar; "hepsi 1-2x" duvarını kıran örnek.
- **Gereken mühendislik:**
  - Çok katmanlı feature çıkarma.
  - Blok difüzyon drafter (eğitim).
  - Blok doğrulama.
- **Başarı ölçütü:** GPU'da >=4x; CPU'da hedef yok.
- **Durma koşulu:** CPU-only hedefiyle uyumsuzsa, ayrı "GPU araştırma" dalına taşı.

### Faz 4 — Grammar fast-forward (hedef 5-10x, kısıtlı)
- **Ne:** JSON/DSL/kod şablonunda zorunlu metni blok ekle; token seçimini kaldır.
- **Neden:** "Doğru mühimmat"ın en somut hali; alfa = 1, c ~= 0.
- **Gereken mühendislik:**
  - Grammar motoru (GBNF/llguidance benzeri).
  - Token sınırı doğrulaması.
  - Blok ekleme + bağlamı işleme.
- **Başarı ölçütü:** JSON/kodda >=3x, şema %100 geçerli.
- **Durma koşulu:** Token sınırı bozuluyorsa, yalnız güvenli sabit bloklarda kullan.

### Faz 5 — Hibrit scheduler (hedef 4x+ genel)
- **Ne:** n-gram + ESP + hidden-state + grammar drafter'larını içerik bazında seç.
- **Neden:** Hybrid Spark/MetaSD çizgisi; tek drafter'ın tavanını aşar.
- **Gereken mühendislik:**
  - Drafter fiyatlandırma/gecikme tahmini.
  - Bütçe dağıtımı.
  - Kabul istatistikleri.
- **Başarı ölçütü:** Karışık iş yükünde >=4x, mod başına regresyon yok.
- **Durma koşulu:** Scheduler overhead kazancı yiyorsa, sabit öncelikli basit kurala dön.

---

## 5. Metrikler

Her deneyde ayrı ayrı:

- **alfa:** kabul edilen / önerilen.
- **c:** taslak forward süresi / hedef forward süresi.
- **k:** önerilen uzunluk.
- **speedup:** baseline tok/s / spec tok/s.
- **Kabul uzunluğu dağılımı:** pozisyon, istek, veri kümesi.
- **Taslak süresi, doğrulama süresi, cache/pozisyon yönetimi süresi:** ayrı.
- **Kalite:** lossless modda birebir eşleşme; task modda görev başarısı.
- **Bellek:** ek KV, drafter, token-info tablosu.

Tek bir "tok/s" sayısı yeterli değil. Hangi terimin değiştiği görülmeli.

---

## 6. Bu repodaki ilk somut adım

1. **KV rollback/truncate ekle.** Context.head ileri gidiyor; kabul edilen prefix'e geri dönüş yok. Bu olmadan spekülatif doğrulama yapılamaz.
2. **Tek zincir doğrulama ekle.** Aday k tokenı tek decode çağrısında işle; her pozisyonun logitini hedef dağılımla karşılaştır; en uzun eşleşen prefix'i kabul et.
3. **n-gram drafter ekle.** Prompt + geçmişten lookup; Faz 0.
4. **Benchmark modu ekle.** Baseline vs spec; alfa, c, k, speedup; aynı prompt seti.
5. **Sonra ESP embedding injection.** Input embedding yolunu aç.
6. **Sonra hidden-state reuse.** Draft model ve token-info injection.

Bu sıra, en az mühendislikle en yüksek kanıtı üretir. Önce n-gram + tek zincir; sonra ağaç; sonra hidden state.

---

## 7. Riskler

- **KV rollback yanlışsa** model bağlamı bozulur; çıktı sessizce sapar.
- **Pozisyon/RoPE yanlışsa** kabul edilen tokenlar yanlış bağlama oturur.
- **Tokenizer sınırı** yüzünden "doğru metin" yanlış token dizisine dönüşür.
- **Ağaç doğrulama** maliyeti, kazancı yiyebilir.
- **Eğitimli drafter** hedef model değişince çöp olur; adapter bakımı gerekir.
- **Task modu** lossless değildir; etiket karışırsa bütün ölçümler güvenilirliğini kaybeder.

---

## 8. Kırmızı çizgi kontrolü

Her faz sonunda:

- Model dosyası hash'i değişti mi? Hayır.
- Quantizasyon tipi değişti mi? Hayır.
- Değişiklik yalnız motor/adaptör katmanında mı? Evet.
- Lossless iddiası varsa, hedef dağılımla eşleşme ölçüldü mü? Evet.
- Task iddiası varsa, görev başarısı ölçüldü mü? Evet.

Bu beş sorudan biri "hayır" ise, deney sonucu rapora girmez.
