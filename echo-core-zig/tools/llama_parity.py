#!/usr/bin/env python3
"""llama-bench ile echo-core --llama-bench'i aynı testlerde, sırayla koşup karşılaştırır.

Kullanım: LLAMA_BIN=.../build-vk/bin tools/llama_parity.py [--reps 5] [--rounds 1] [--models a,b] [--out sonuc.json]
Her test ayrı süreçte koşar; termal kaymayı dengelemek için motorlar test başına yer değiştirir.
"""
import argparse, json, os, statistics, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--models", default="Qwen3.5-0.8B-Q8_0,Qwen3.5-0.8B-UD-Q8_K_XL,unc-0.8b-q5km")
ap.add_argument("--tests", default="pp128,pp512,tg32,tg128")
ap.add_argument("-t", type=int, default=6)
ap.add_argument("--out")
a = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ECHO = f"{ROOT}/zig-out/bin/echo-core-zig"
MODELS = f"{os.path.dirname(ROOT)}/ArastirmaModelleri"
LB = os.environ["LLAMA_BIN"]
ENV = dict(os.environ, GGML_DISABLE_VULKAN="1", LD_LIBRARY_PATH=LB)


def args_of(test):
    n = int(test[2:])
    return ("-p", str(n), "-n", "0") if test.startswith("pp") else ("-p", "0", "-n", str(n))


def llama(model, test):
    p = subprocess.run([f"{LB}/llama-bench", "-m", model, "-t", str(a.t), "-r", str(a.reps), *args_of(test), "-ngl", "0", "-o", "json"],
                       capture_output=True, text=True, env=ENV, check=True)
    return json.loads(p.stdout)[0]["samples_ts"]


def echo(model, test):
    p = subprocess.run([ECHO, "--llama-bench", "-m", model, "-t", str(a.t), "-r", str(a.reps), *args_of(test)],
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])["samples_ts"]


rows = []
for name in a.models.split(","):
    model = name if "/" in name else f"{MODELS}/{name}.gguf"
    name = os.path.basename(name).removesuffix(".gguf")
    for test in a.tests.split(","):
        ls, es = [], []
        for r in range(a.rounds):
            if r % 2 == 0:
                ls += llama(model, test); es += echo(model, test)
            else:
                es += echo(model, test); ls += llama(model, test)
        lm, em = statistics.median(ls), statistics.median(es)
        rows.append(dict(model=name, test=test, llama=lm, echo=em, ratio=em / lm, llama_samples=ls, echo_samples=es))
        print(f"{name:26s} {test:6s} llama={lm:8.2f} echo={em:8.2f} oran={em / lm * 100:6.1f}%", flush=True)
if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), threads=a.t, reps=a.reps, rows=rows), open(a.out, "w"), indent=1)
worst = min(r["ratio"] for r in rows)
print(f"en kötü oran: {worst * 100:.1f}%")
sys.exit(0 if worst >= 1.0 else 2)
