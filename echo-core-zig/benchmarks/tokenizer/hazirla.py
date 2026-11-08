# GGUF'tan vocab/merge, unicode sınıf tablosu, HF tokenizer.json ve korpusları üretir.
import glob, json, os, random, struct, sys, unicodedata
from gguf import GGUFReader

KOK = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MODEL = os.path.join(KOK, "ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf")
VERI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_veri")
# llama.cpp src/llama-vocab.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 (tokenizer.json'daki orijinal)
DESEN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SINIR = 1 << 20  # korpus başına 1 MB
RESMI_URL = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/main/tokenizer.json"  # modelin kendi tokenizer'ı = karşılaştırma ölçüsü


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def alan(r, k):
    f = r.fields[k]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def main():
    os.makedirs(VERI, exist_ok=True)
    r = GGUFReader(MODEL)
    toks = alan(r, "tokenizer.ggml.tokens")
    f = r.fields["tokenizer.ggml.token_type"]
    tip = [int(f.parts[i][0]) for i in f.data]
    merges = alan(r, "tokenizer.ggml.merges")
    b2u = bytes_to_unicode(); u2b = {v: k for k, v in b2u.items()}

    # vocab.bin: yalnız normal (tip 1) tokenlar, ham bayt halinde
    normal = {}
    with open(os.path.join(VERI, "vocab.bin"), "wb") as o:
        kayit = []
        for i, t in enumerate(toks):
            if tip[i] != 1: continue
            try: b = bytes(u2b[c] for c in t)
            except KeyError: continue
            normal[t] = i; kayit.append((i, b))
        o.write(struct.pack("<I", len(kayit)))
        for i, b in kayit: o.write(struct.pack("<II", i, len(b)) + b)
    with open(os.path.join(VERI, "merges.txt"), "w", encoding="utf-8") as o:
        o.write("\n".join(merges) + "\n")
    # kullanıcı tanımlı (tip 4): llama.cpp parse_special=false iken bile metinden ayırır (<think>, <tool_call> ...)
    resmi = os.path.join(VERI, "orijinal", "tokenizer.json")
    if not os.path.exists(resmi):
        import urllib.request; os.makedirs(os.path.dirname(resmi), exist_ok=True); urllib.request.urlretrieve(RESMI_URL, resmi)
    # resmi tokenizer'ın metinden ayırdığı tokenlar (<|im_start|>, <think> ...)
    kullanici = [(a["id"], a["content"]) for a in json.load(open(resmi, encoding="utf-8"))["added_tokens"]]
    assert all(t.startswith("<") for _, t in kullanici)
    with open(os.path.join(VERI, "kullanici.tsv"), "w", encoding="utf-8") as o:
        o.write("".join(f"{i}\t{t}\n" for i, t in kullanici))
    with open(os.path.join(VERI, "ozel.json"), "w") as o:
        json.dump({t: i for i, t in enumerate(toks) if tip[i] != 1}, o)

    # unicode bayrakları: 1=L 2=M 4=N 8=boşluk (Unicode White_Space); iki kademeli tablo
    ws = set([*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000])
    bloklar, idx, s1 = [], {}, []
    for blk in range(0x110000 // 256):
        b = bytearray(256)
        for j in range(256):
            cp = blk * 256 + j; c = unicodedata.category(chr(cp))
            b[j] = (1 if c[0] == "L" else 0) | (2 if c[0] == "M" else 0) | (4 if c[0] == "N" else 0) | (8 if cp in ws else 0)
        b = bytes(b)
        if b not in idx: idx[b] = len(bloklar); bloklar.append(b)
        s1.append(idx[b])
    with open(os.path.join(VERI, "ucat.bin"), "wb") as o:
        o.write(struct.pack("<I", len(bloklar))); o.write(struct.pack(f"<{len(s1)}H", *s1)); o.write(b"".join(bloklar))

    # HF tokenizer.json (BPE + qwen35 regex + ByteLevel); NFC normalizer yok (llama.cpp de yapmıyor)
    from tokenizers import Tokenizer, Regex, models, pre_tokenizers, decoders
    tk = Tokenizer(models.BPE(vocab={**normal, **{t: i for i, t in kullanici}}, merges=[tuple(m.split(" ", 1)) for m in merges], ignore_merges=False))
    tk.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(DESEN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tk.decoder = decoders.ByteLevel()
    from tokenizers import AddedToken
    tk.add_tokens([AddedToken(t, special=False, normalized=False) for _, t in kullanici])
    tk.save(os.path.join(VERI, "tokenizer.json"))

    # korpuslar (sabit kaynaklar)
    def topla(desenler, ad):
        buf = []
        for d in desenler:
            for p in sorted(glob.glob(os.path.join(KOK, d), recursive=True)):
                try: buf.append(open(p, encoding="utf-8").read())
                except Exception: pass
        t = "\n\n".join(buf).encode("utf-8")[:SINIR].decode("utf-8", "ignore")
        open(os.path.join(VERI, f"korpus_{ad}.txt"), "w", encoding="utf-8").write(t)
    topla(["ISLEYIS-ARASTIRMA.md", "YONTEM-ARASTIRMA.md", "TAVAN-VE-VARSAYIMLAR.md", "HANDOFF.md", "TG-MAX.md", "README.md",
           "session-2fa515ab-0d46-4736-9d06-1e49b3b6489a-vasiyet.md"], "tr")
    topla(["../bench-build/llama.cpp/docs/**/*.md", "../bench-build/llama.cpp/README.md"], "en")
    topla(["echo-core-zig/src/**/*.zig", "../bench-build/llama.cpp/src/*.cpp"], "kod")
    random.seed(1)  # en kötü durum: 16 KB boşluksuz rastgele harf = tek dev kelime
    open(os.path.join(VERI, "korpus_uzunkelime.txt"), "w").write("".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(16384)))
    # referans: modelin resmi tokenizer'ı, paragraf paragraf
    resmi_tk = Tokenizer.from_file(resmi)
    for k in ["tr", "en", "kod", "uzunkelime"]:
        par = open(os.path.join(VERI, f"korpus_{k}.txt"), encoding="utf-8").read().split("\n\n")
        with open(os.path.join(VERI, f"ref_{k}.txt"), "w") as o:
            o.write("".join(" ".join(map(str, resmi_tk.encode(p, add_special_tokens=False).ids)) + "\n" for p in par))
    print("hazir:", VERI, "normal token:", len(normal), "merge:", len(merges))


if __name__ == "__main__":
    main()
