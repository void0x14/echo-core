"""NFC sınaması: korpusların NFD hali + rastgele aksan yığınları; ref = resmi tokenizer.json (NFC normalizer içerir)."""
import os, random, sys, unicodedata as ud
from tokenizers import Tokenizer
VERI = sys.argv[1]
tok = Tokenizer.from_file(os.path.join(VERI, "orijinal", "tokenizer.json"))
def yaz(ad, pars):
    pars = [p for p in pars if p and "\n\n" not in p]
    open(os.path.join(VERI, f"korpus_{ad}.txt"), "w", encoding="utf-8").write("\n\n".join(pars))
    with open(os.path.join(VERI, f"ref_{ad}.txt"), "w") as o:
        for p in pars: o.write(" ".join(map(str, tok.encode(p, add_special_tokens=False).ids)) + "\n")
for k in ["tr", "en", "kod"]:
    pars = open(os.path.join(VERI, f"korpus_{k}.txt"), encoding="utf-8").read().split("\n\n")
    yaz(f"nfd_{k}", [ud.normalize("NFD", p) for p in pars])
r = random.Random(7)
isaret = [c for c in range(0x300, 0x370)] + [0x323, 0x327, 0x328, 0x31B, 0x0E48, 0x093C, 0x05B4, 0x0651, 0x302A, 0x1DC0]
taban = list("aeiouAEIOUsSgGcCnNyYzZ") + ["क", "ב", "ا", "ᄀ", "가", "Å", "Ω", "ཱི", "̈́"]
hangul = ["ᄀ", "ᅡ", "ᆨ", "가", "ᄒ", "ᅵ", "ᇂ"]
pars = []
for _ in range(2000):
    s = []
    for _ in range(r.randint(1, 12)):
        s.append(r.choice(taban + hangul))
        s += [chr(r.choice(isaret)) for _ in range(r.randint(0, 4))]
        if r.random() < 0.3: s.append(" ")
    pars.append("".join(s))
yaz("nfc_rastgele", pars)
print("tamam")
