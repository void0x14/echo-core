#!/usr/bin/env python3
"""Tam (lossless) yolun logit'leri referansla bit bit aynı mı denetler.

Referans: llama.cpp ile sıfır farklı doğrulanmış a5e346a binary'sinin çıktısı.
  tools/logit_check.py --save [--bin eski_binary]  referansı yazar
  tools/logit_check.py [--bin binary] [-t 6]        karşılaştırır; fark varsa çıkış kodu 1
"""
import argparse, hashlib, json, os, random, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = f"{os.path.dirname(ROOT)}/ArastirmaModelleri"
REF = f"{ROOT}/benchmarks/logit-ref.json"
NAMES = ["Qwen3.5-0.8B-Q8_0", "Qwen3.5-0.8B-UD-Q8_K_XL", "unc-0.8b-q5km"]
SEQS = ["5,1,128,1,300,1", "512,1,1", "7,300,300,1,1"]

ap = argparse.ArgumentParser()
ap.add_argument("--save", action="store_true")
ap.add_argument("--bin", default=f"{ROOT}/zig-out/bin/echo-core-zig")
ap.add_argument("-t", type=int, default=6)
a = ap.parse_args()


def digest(name, steps):
    n = sum(map(int, steps.split(",")))
    rng = random.Random(1234)
    toks = ",".join(str(rng.randrange(1000, 48000)) for _ in range(n))
    with tempfile.NamedTemporaryFile() as f:
        subprocess.run([a.bin, "--llama-bench", "-m", f"{MODELS}/{name}.gguf", "-t", str(a.t), "--dump-logits", f.name,
                        "--tokens", toks, "--steps", steps, "--ctx", str(n + 1)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return hashlib.sha256(open(f.name, "rb").read()).hexdigest()


got = {f"{m}|{s}": digest(m, s) for m in NAMES for s in SEQS}
if a.save:
    json.dump(got, open(REF, "w"), indent=1)
    print(f"yazıldı: {REF}")
    sys.exit(0)
ref = json.load(open(REF))
bad = [k for k in ref if ref[k] != got.get(k)]
print(f"{len(ref) - len(bad)}/{len(ref)} eşit")
for k in bad:
    print("FARK", k)
sys.exit(1 if bad else 0)
