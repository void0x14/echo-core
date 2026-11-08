"""NFC tabloları -> src/tokenizer/nfc.bin (Hangul hariç; o algoritmik).
Düzen (hepsi u32 little-endian):
  başlık: n_qc, n_dec, n_pool, n_ccc, n_comp
  qc aralıkları [bas, son]  : NFC'yi değiştirebilecek kod noktaları (ccc!=0, NFC(c)!=c, birleşmenin 2. öğesi)
  dec  [cp, ofs, uzunluk]    : tam kanonik ayrışma (özyinelemeli açılmış), havuz ofsetiyle
  havuz u32[]
  ccc  [cp, sınıf]
  comp [a, b, sonuç]         : (a,b) sıralı
"""
import struct, sys, unicodedata as ud
SB, LB, VB, TB = 0xAC00, 0x1100, 0x1161, 0x11A7
def hangul(c): return SB <= c < SB + 11172
qc, dec, ccc, comp = set(), {}, {}, {}
for c in range(0x110000):
    if 0xD800 <= c < 0xE000: continue
    ch = chr(c); k = ud.combining(ch)
    if k: ccc[c] = k; qc.add(c)
    if hangul(c): continue
    d = ud.normalize("NFD", ch)
    if d != ch: dec[c] = [ord(x) for x in d]
    if ud.normalize("NFC", ch) != ch: qc.add(c)
    raw = ud.decomposition(ch)
    if raw and not raw.startswith("<"):
        p = [int(x, 16) for x in raw.split()]
        if len(p) == 2 and ud.normalize("NFC", chr(p[0]) + chr(p[1])) == ch:
            comp[(p[0], p[1])] = c; qc.add(p[1])
qc |= set(range(VB, VB + 21)) | set(range(TB + 1, TB + 28))  # Hangul V, T birleşir
assert min(qc) >= 0x300, hex(min(qc))  # UTF-8 öncü bayt >= 0xCC hızlı yolu buna dayanır
ar = []
for c in sorted(qc):
    if ar and ar[-1][1] == c - 1: ar[-1][1] = c
    else: ar.append([c, c])
pool, decs = [], []
for c in sorted(dec): decs.append((c, len(pool), len(dec[c]))); pool += dec[c]
o = open(sys.argv[1], "wb")
o.write(struct.pack("<5I", len(ar), len(decs), len(pool), len(ccc), len(comp)))
for a, b in ar: o.write(struct.pack("<2I", a, b))
for t in decs: o.write(struct.pack("<3I", *t))
o.write(struct.pack(f"<{len(pool)}I", *pool))
for c in sorted(ccc): o.write(struct.pack("<2I", c, ccc[c]))
for (a, b) in sorted(comp): o.write(struct.pack("<3I", a, b, comp[(a, b)]))
print(f"unicode {ud.unidata_version}: qc {len(ar)} aralık, dec {len(decs)}, ccc {len(ccc)}, comp {len(comp)}")
