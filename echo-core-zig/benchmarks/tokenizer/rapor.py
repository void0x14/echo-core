# sonuc_c.tsv + sonuc_py.tsv -> SONUC.md
import datetime, os, subprocess
D = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(D, "_veri")
KORP = ["tr", "en", "kod", "uzunkelime"]
PREFILL = [("kayıpsız pp512", 823), ("kayıplı pp512", 2109), ("metindeki varsayım", 3000)]
IDDIA = {"tiktoken (Rust)": "30–60", "HF tokenizers (Rust)": "10–25", "llama.cpp (libllama)": "15–40",
         "SentencePiece (uyarlanmis, on-bolme Python)": "20–50", "fastBPE cekirdegi (process_bpe)": "40–80", "kitoken (Rust)": "60–120",
         "LinMaxMatch (trie, en uzun eslesme)": "200+", "Aho-Corasick (pyahocorasick C, on-bolme Python)": "~500", "Qwen BPE: dizi tarama": "> heap", "Qwen BPE: heap (oncelik kuyrugu)": "—"}

satir, trie = {}, None
for f in ["sonuc_c.tsv", "sonuc_py.tsv"]:
    for l in open(os.path.join(V, f), encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if p[0] == "#trie_dugum": trie = int(p[1]); continue
        if len(p) < 7 or p[2] == "HATA" or p[1] == "HATA": satir.setdefault(p[0], {})["hata"] = p[-1]; continue
        satir.setdefault(p[0], {})[p[1]] = dict(b=int(p[2]), t=float(p[3]), n=int(p[4]), tam=int(p[5]), top=int(p[6]))

ref = satir["Qwen resmi tokenizer.json (HF, referans)"]
bpt = {k: ref[k]["b"] / ref[k]["n"] for k in KORP}
mbs = lambda r, k: r[k]["b"] / r[k]["t"] / 1e6 if k in r and r[k]["t"] > 0 else None
fmt = lambda x: "—" if x is None else (f"{x:.0f}" if x >= 10 else f"{x:.2f}" if x >= 0.1 else f"{x:.4f}")
cpu = subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", shell=True, capture_output=True, text=True).stdout.strip()

o = [f"# Tokenizer ölçümü — {datetime.date.today()}", "",
     f"Otomatik üretildi (`calistir.sh`). CPU: {cpu}, tek çekirdek (`taskset`). Model: Qwen3.5-0.8B-Q8_0 (vocab 248 320, merge 247 587, ön-bölme `qwen35`).",
     "Karşılaştırma ölçüsü: modelin kendi tokenizer'ı (Qwen resmi `tokenizer.json`, huggingface.co/Qwen/Qwen3.5-0.8B). Doğruluk = paragraf bazında token id dizisi resmi tokenizer ile birebir aynı mı.", "",
     "## 1. Hız (MB/s, tek çekirdek, medyan)", "",
     "| Yöntem | TR | EN | Kod | 16 KB tek kelime (en kötü) | Metindeki iddia | TR hızı, resmi tokenizer = 1.00x |",
     "|---|---|---|---|---|---|---|"]
lr = mbs(ref, "tr")
for ad, r in satir.items():
    if "hata" in r and "tr" not in r: o.append(f"| {ad} | HATA: {r['hata']} | | | | | |"); continue
    kat = f"{mbs(r, 'tr') / lr:.2f}x" if mbs(r, "tr") else "—"
    o.append(f"| {ad} | {fmt(mbs(r, 'tr'))} | {fmt(mbs(r, 'en'))} | {fmt(mbs(r, 'kod'))} | {fmt(mbs(r, 'uzunkelime'))} | {IDDIA.get(ad, '—')} | {kat} |")

o += ["", "## 2. Resmi tokenizer ile birebir aynı bölünen paragraf", "",
      "| Yöntem | TR | EN | Kod | Toplam | Token sayısı (TR) |", "|---|---|---|---|---|---|"]
for ad, r in satir.items():
    if "tr" not in r or r["tr"]["top"] == 0: continue
    h = lambda k: f"{r[k]['tam']}/{r[k]['top']}" if k in r else "—"
    tt = sum(r[k]["tam"] for k in KORP if k in r); tp = sum(r[k]["top"] for k in KORP if k in r)
    o.append(f"| {ad} | {h('tr')} | {h('en')} | {h('kod')} | %{100 * tt / tp:.2f} | {r['tr']['n']} (ref {ref['tr']['n']}) |")

o += ["", "## 3. 10 000 token prompt'ta tokenizer payı (TR metin)", "",
      f"TR'de ölçülen bayt/token: **{bpt['tr']:.2f}** (EN {bpt['en']:.2f}, kod {bpt['kod']:.2f}). 10 000 token = {10000 * bpt['tr'] / 1024:.0f} KB.", "",
      "| Yöntem | Tokenize (ms) | " + " | ".join(f"Pay @ {a} {s} tok/s" for a, s in PREFILL) + " |",
      "|---|---|" + "---|" * len(PREFILL)]
for ad, r in satir.items():
    m = mbs(r, "tr")
    if not m: continue
    ms = 10000 * bpt["tr"] / (m * 1e6) * 1e3
    o.append(f"| {ad} | {ms:.2f} | " + " | ".join(f"%{100 * ms / (ms + 10000 / s * 1e3):.3f}" for _, s in PREFILL) + " |")

o += ["", "## 4. Ham gerçekler", "",
      f"- Trie düğüm sayısı: {trie:,}. Yoğun Aho-Corasick tablosu (düğüm × 256 × 4 B) ≈ **{trie * 256 * 4 / 1e6:.0f} MB** → L2/L3'e sığmaz (Ryzen 5 3600 L3 = 32 MB).",
      f"- Bayt/token: TR {bpt['tr']:.2f}, EN {bpt['en']:.2f}, kod {bpt['kod']:.2f} (metindeki varsayım: 4).",
      "- `echo bugünkü` satırı: `tokenizer.zig:137` algoritmasının C kopyası, yalnız ilk 4 KB TR üzerinde (çok yavaş olduğu için).",
      "- Python ile ölçülenlerde (HF, tiktoken, kitoken, SPM) Python çağrı + id listesi oluşturma maliyeti dahil; tek çağrı / tüm dosya olduğu için pay küçük.",
      "- SentencePiece byte-level GPT-2 sözlüğünü doğrudan yükleyemiyor: puan = −merge sırası verilerek uyarlandı, ön-bölme Python `regex` ile yapıldı (süreye dahil).",
      "- Aho-Corasick: pyahocorasick (C) `iter_long` = soldan en uzun eşleşme; kelimeler `\\x00` ile ayrılır, ön-bölme Python `regex` (süreye dahil).",
      "- fastBPE: yalnız `process_bpe` birleştirme çekirdeği kullanıldı (kendi ön-bölmesi boşlukla böler, Qwen'e uymaz); ön-bölme bizimki.", ""]
open(os.path.join(D, "SONUC.md"), "w", encoding="utf-8").write("\n".join(o))
print("\n".join(o))
