#!/usr/bin/env python3
"""echo-core'u BENCHMARK-CPU-GPU-2026-11-04.md'deki llama.cpp CPU değerleriyle hücre hücre karşılaştırır.

Kullanım: tools/doc_parity.py [--reps 5] [--rounds 1] [--out sonuc.json]
Belgedeki gibi 5 tekrarın medyanı raporlanır; her tur için tekrar edilir ve en iyi tur medyanı da yazılır.
"""
import argparse, json, os, re, statistics, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("-t", type=int, default=6)
ap.add_argument("--out")
a = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = f"{os.path.dirname(ROOT)}/BENCHMARK-CPU-GPU-2026-11-04.md"
ECHO = f"{ROOT}/zig-out/bin/echo-core-zig"
MODELS = f"{os.path.dirname(ROOT)}/ArastirmaModelleri"

section = open(DOC).read().split("## CPU\n", 1)[1].split("### CPU", 1)[0]
base = {}
for m, test, val in re.findall(r"^\| (\S+) \([^)]*\) \| (pp128|pp512|tg32|tg128) \| ([\d.]+) \|", section, re.M):
    base[(m, test)] = float(val)
assert len(base) == 12, base


def run(model, test):
    n = int(test[2:])
    args = ["-p", str(n), "-n", "0"] if test.startswith("pp") else ["-p", "0", "-n", str(n)]
    p = subprocess.run([ECHO, "--llama-bench", "-m", f"{MODELS}/{model}.gguf", "-t", str(a.t), "-r", str(a.reps), *args],
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])["samples_ts"]


rows = []
for (model, test), ref in base.items():
    medians = [statistics.median(run(model, test)) for _ in range(a.rounds)]
    med, best = statistics.median(medians), max(medians)
    rows.append(dict(model=model, test=test, doc=ref, echo_median=med, echo_best=best, round_medians=medians))
    print(f"{model:26s} {test:6s} belge={ref:8.2f} echo={med:8.2f} en_iyi={best:8.2f} oran={med / ref * 100:6.1f}%", flush=True)
if a.out:
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), load=os.getloadavg(), reps=a.reps, rounds=a.rounds, rows=rows), open(a.out, "w"), indent=1)
worst = min(r["echo_median"] / r["doc"] for r in rows)
print(f"en kötü oran: {worst * 100:.1f}%  load={os.getloadavg()[0]:.1f}")
sys.exit(0 if worst >= 1.0 else 2)
