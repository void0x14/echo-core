# Hazır kütüphaneler: HF tokenizers, tiktoken, kitoken, SentencePiece (uyarlanmış). Tek çekirdek, aynı korpus, aynı referans.
import base64, os, statistics, sys, time
os.environ["RAYON_NUM_THREADS"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
import regex
VERI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_veri")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hazirla import DESEN, bytes_to_unicode

KORP = ["tr", "en", "kod", "uzunkelime"]
b2u = bytes_to_unicode(); u2b = {v: k for k, v in b2u.items()}


def vocab_bytes():
    import struct
    d = open(os.path.join(VERI, "vocab.bin"), "rb").read(); n = struct.unpack_from("<I", d)[0]; p = 4; v = {}
    for _ in range(n):
        i, l = struct.unpack_from("<II", d, p); p += 8; v[d[p:p + l]] = i; p += l
    return v


KULLANICI = [l.split("\t") for l in open(os.path.join(VERI, "kullanici.tsv"), encoding="utf-8").read().splitlines()]
KULLANICI = {t: int(i) for i, t in KULLANICI}


def yap_hf():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(VERI, "tokenizer.json"))
    return lambda s: t.encode(s, add_special_tokens=False).ids


def yap_resmi():
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(VERI, "orijinal", "tokenizer.json"))
    return lambda s: t.encode(s, add_special_tokens=False).ids


def yap_tiktoken():
    import tiktoken
    e = tiktoken.Encoding("qwen35", pat_str=DESEN, mergeable_ranks=vocab_bytes(), special_tokens=KULLANICI)
    return lambda s: e.encode(s, allowed_special="all", disallowed_special=())


def yap_kitoken():
    import kitoken
    k = kitoken.Kitoken.from_tokenizers_file(os.path.join(VERI, "tokenizer.json"))
    return lambda s: k.encode(s, True)


def yap_spm():
    # SentencePiece BPE byte-level GPT-2 sözlüğünü doğrudan okuyamaz; uyarlama:
    # parça puanı = -(onu üreten merge sırası), girdi = bizim ön-bölmemizle bölünüp boşlukla birleştirilmiş GPT-2 harfleri.
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    v = vocab_bytes(); id2u = {i: "".join(b2u[c] for c in b) for b, i in v.items()}
    rank = {}
    for r, m in enumerate(open(os.path.join(VERI, "merges.txt"), encoding="utf-8").read().splitlines()):
        a, b = m.split(" ", 1); rank.setdefault(a + b, r)
    N = max(id2u) + 1
    mp = pb.ModelProto(); mp.trainer_spec.model_type = pb.TrainerSpec.BPE
    mp.trainer_spec.split_by_whitespace = True; mp.trainer_spec.byte_fallback = False
    ns = mp.normalizer_spec; ns.name = "identity"; ns.add_dummy_prefix = False; ns.remove_extra_whitespaces = False; ns.escape_whitespaces = True
    for i in range(N):
        p = mp.pieces.add()
        if i in id2u:
            p.piece = id2u[i]; p.score = -float(rank.get(id2u[i], 0)) if len(id2u[i]) > 1 else 0.0; p.type = pb.ModelProto.SentencePiece.NORMAL
        else:
            p.piece = f"<bos_{i}>"; p.score = 0; p.type = pb.ModelProto.SentencePiece.UNUSED
    u = mp.pieces.add(); u.piece = "<unk>"; u.type = pb.ModelProto.SentencePiece.UNKNOWN; unk = N
    sp = spm.SentencePieceProcessor(model_proto=mp.SerializeToString())
    pat = regex.compile(DESEN)
    ozel = regex.compile("(" + "|".join(regex.escape(t) for t in sorted(KULLANICI, key=len, reverse=True)) + ")")

    def enc(s):
        out = []
        for j, parca in enumerate(ozel.split(s)):
            if j % 2: out.append(KULLANICI[parca]); continue
            if not parca: continue
            g = " ".join("".join(b2u[c] for c in w.encode()) for w in pat.findall(parca))
            out += [x for x in sp.encode(g) if x != unk]
        return out
    return enc


def yap_ahocorasick():
    # Gerçek Aho-Corasick otomatı (pyahocorasick, C). iter_long = soldan en uzun, örtüşmesiz eşleşme.
    # Kelimeler arasına hiçbir tokende olmayan \x00 konur; böylece eşleşme ön-bölme sınırını aşamaz.
    import ahocorasick
    A = ahocorasick.Automaton()
    for b, i in vocab_bytes().items(): A.add_word("".join(b2u[c] for c in b), i)
    A.make_automaton()
    pat = regex.compile(DESEN)
    ozel = regex.compile("(" + "|".join(regex.escape(t) for t in sorted(KULLANICI, key=len, reverse=True)) + ")")
    tab = str.maketrans({i: b2u[i] for i in range(256)})

    def enc(s):
        out = []
        for j, parca in enumerate(ozel.split(s)):
            if j % 2: out.append(KULLANICI[parca]); continue
            if not parca: continue
            g = "\x00".join(w.encode().decode("latin-1").translate(tab) for w in pat.findall(parca))
            out += [v for _, v in A.iter_long(g)]
        return out
    return enc


def olc(f, s, R):
    v = []
    for _ in range(R):
        t0 = time.perf_counter(); o = f(s); v.append(time.perf_counter() - t0)
    return statistics.median(v), len(o)


def main():
    yapicilar = [("Qwen resmi tokenizer.json (HF, referans)", yap_resmi), ("HF tokenizers (Rust)", yap_hf), ("tiktoken (Rust)", yap_tiktoken), ("kitoken (Rust)", yap_kitoken),
                 ("SentencePiece (uyarlanmis, on-bolme Python)", yap_spm),
                 ("Aho-Corasick (pyahocorasick C, on-bolme Python)", yap_ahocorasick)]
    for ad, y in yapicilar:
        try:
            f = y(); f("deneme")
        except Exception as e:
            print(f"{ad}\tHATA\t0\t0\t0\t0\t0\t{type(e).__name__}: {str(e)[:150]}", flush=True); continue
        for k in KORP:
            s = open(os.path.join(VERI, f"korpus_{k}.txt"), encoding="utf-8").read()
            ref = open(os.path.join(VERI, f"ref_{k}.txt")).read().split("\n")
            par = s.split("\n\n")
            try:
                t, nt = olc(f, s, 3)
                tam = sum(1 for p, r in zip(par, ref) if " ".join(map(str, f(p))) == r)
                print(f"{ad}\t{k}\t{len(s.encode())}\t{t:.6f}\t{nt}\t{tam}\t{len(par)}", flush=True)
            except Exception as e:
                print(f"{ad}\t{k}\tHATA\t0\t0\t0\t0\t{type(e).__name__}: {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
