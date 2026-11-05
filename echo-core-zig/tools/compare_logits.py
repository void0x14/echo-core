#!/usr/bin/env python3
"""echo-core ve llama.cpp logit dökümlerini karşılaştırır: echo-core-zig ile ref_logits aynı token dizisinde."""
import argparse, os, random, struct, subprocess, sys, array, math

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--steps", default="5,1,1,1")
ap.add_argument("--ctx", type=int, default=1024); ap.add_argument("-t", type=int, default=6)
ap.add_argument("--out", default="/tmp/echo_logits"); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--vocab", type=int, default=40000)
a = ap.parse_args()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECHO = f"{ROOT}/zig-out/bin/echo-core-zig"
LB = os.environ["LLAMA_BIN"]
REF = os.environ.get("REF_LOGITS", f"{LB}/ref_logits")
steps = [int(s) for s in a.steps.split(",")]
rng = random.Random(a.seed)
tokens = ",".join(str(rng.randrange(a.vocab)) for _ in range(sum(steps)))
env = dict(os.environ, GGML_DISABLE_VULKAN="1", LD_LIBRARY_PATH=LB)
subprocess.run([REF, a.model, tokens, a.steps, str(a.t), str(a.ctx), a.out + ".ref"], check=True, env=env, capture_output=True)
subprocess.run([ECHO, "--llama-bench", "-m", a.model, "--dump-logits", a.out + ".echo", "--tokens", tokens, "--steps", a.steps, "--ctx", str(a.ctx), "-t", str(a.t)], check=True)
ra = array.array("f"); rb = array.array("f")
ra.frombytes(open(a.out + ".ref", "rb").read()); rb.frombytes(open(a.out + ".echo", "rb").read())
if len(ra) != len(rb):
    print("BOYUT FARKI", len(ra), len(rb)); sys.exit(1)
nv = len(ra) // len(steps)
ok = True
for i, s in enumerate(steps):
    x = ra[i * nv:(i + 1) * nv]; y = rb[i * nv:(i + 1) * nv]
    md = max(abs(p - q) for p, q in zip(x, y))
    amx = max(range(nv), key=lambda j: x[j]); amy = max(range(nv), key=lambda j: y[j])
    top = sorted(range(nv), key=lambda j: -x[j])[:5]
    topy = sorted(range(nv), key=lambda j: -y[j])[:5]
    print(f"adim {i} n={s}: max|fark|={md:.5f} argmax ref={amx} echo={amy} top5 ayni={top == topy}")
    ok &= (amx == amy) and md < 0.05 and not math.isnan(md)
print("SONUC:", "ESIT" if ok else "FARKLI")
sys.exit(0 if ok else 1)
