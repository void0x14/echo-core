#!/usr/bin/env python3
"""Deney C: istasyonlu (retime/recheck) geçiş, istasyonsuz tek geçişe karşı.

ÖN-KAYIT (bu dosya koşudan önce commit edildi):
  Yazmalar: W2 (S1→S2), W3 (S2→S3), W4 (S3 çıktı). Her yazmada olasılık p ile hata.
  Tek geçiş: kontrol yok. İstasyonlu: her yazmadan sonra CRC; tutmazsa üst buffer'dan yeniden yaz (en fazla 3).
  Beklentiler (N blok, q = sabit blok oranı):
    tek geçiş yanlış       ≈ N·(q + (1−q)·(1−(1−p)³))
    istasyonlu kayıp       ≈ N·(q + (1−q)·(1−(1−p⁴)³))
  C-1  istasyonlu yanlış teslim = 0 (blok ve uçtan uca, her koşul)
  C-2  tek geçiş yanlış, beklentiden ±%20 (p=0 → 0)
  C-3  istasyonlu kayıp, |gözlem−beklenen| ≤ 3√beklenen + 3
  C-4  istasyonlu/tek geçiş süre oranı, p=0 blok düzeyi ≤ 2.0 (overhead kabul eşiği)
  E-1  uçtan uca istasyonlu doğrulama yanlış = 0
  E-2  p=0.1'de DOĞRU tok/s (= toplam tok/s × doğru blok oranı) istasyonlu > tek geçiş
       (çürütme: istasyon overhead'i doğru iş kazancını yiyorsa işe yaramaz)
Çalıştırma: tools/run_deney_c.py [blocks|e2e|all]
"""
import json
import math
import statistics as S
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # echo-core-zig (worktree)
MODEL = Path('/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf')
BIN_BLK = ROOT / 'zig-out' / 'bin' / 'deney-c'
BIN_E2E = ROOT / 'zig-out' / 'bin' / 'deney-c-e2e'
OUT_DIR = ROOT / 'benchmarks'
BLK = 4096
RETRY = 3

N = 200_000
CPU = 0
REPS = 5
SCENARIOS = [(0.0, 0.0), (0.01, 0.0), (0.1, 0.0), (0.3, 0.0), (0.1, 0.001)]
MODES = ['single', 'station']

E2E_TOKENS = 32
E2E_REPS = 5
E2E_ARMS = [('none', 0.0, 0.0), ('single', 0.0, 0.0), ('station', 0.0, 0.0),
            ('single', 0.1, 0.0), ('station', 0.1, 0.0)]

BENCH_TG32 = 24.48          # BENCHMARK-CPU-GPU-2026-11-04.md, Q8_0 tg32
WRITES = 3


def run_blk(mode, p, q, seed):
    cmd = [str(BIN_BLK), str(MODEL), mode, str(N), str(p), str(q), str(seed), str(CPU)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise SystemExit(f'{mode} p={p} q={q} seed={seed} başarısız:\n{proc.stderr}')
    return json.loads([ln for ln in proc.stderr.splitlines() if ln.startswith('{')][-1])


def run_e2e(cond, p, q, seed):
    cmd = [str(BIN_E2E), str(MODEL), cond, str(E2E_TOKENS), str(p), str(q), str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f'e2e {cond} p={p} başarısız:\n{proc.stderr}')
    return json.loads([ln for ln in proc.stderr.splitlines() if ln.startswith('{')][-1])


def med(rows, key):
    return S.median(r[key] for r in rows)


def exp_wrong_single(p, q):
    return N * (q + (1 - q) * (1 - (1 - p) ** WRITES))


def exp_lost_station(p, q):
    return N * (q + (1 - q) * (1 - (1 - p ** (RETRY + 1)) ** WRITES))


def block_part(md, raw):
    groups = defaultdict(list)
    for p, q in SCENARIOS:
        for mode in MODES:
            run_blk(mode, p, q, 0)                              # ısınma, atılır
            for r in range(1, REPS + 1):
                groups[(p, q, mode)].append(run_blk(mode, p, q, r))
    raw['blocks'] = {f'{p}|{q}|{m}': rows for (p, q, m), rows in groups.items()}

    md += ['## Blok düzeyi', '',
           f'N = {N} blok, B = {BLK} B, CPU {CPU} pinned, medyan / {REPS} tekrar. Girdi: gerçek ffn_down tensör baytları.', '',
           '| p | q | mod | faydalı MB/s | µs/blok | p50 ns | p99 ns | doğru | yanlış | kayıp | tespit | yeniden yazma | CRC kontrol |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (p, q, m) in sorted(groups):
        g = groups[(p, q, m)]
        us = 1e6 / med(g, 'blocks_per_s')
        lo = min(r['useful_mb_s'] for r in g)
        hi = max(r['useful_mb_s'] for r in g)
        md.append(
            f"| {p} | {q} | {m} | {med(g, 'useful_mb_s'):.1f} ({lo:.1f}–{hi:.1f}) | {us:.3f} | "
            f"{med(g, 'p50_ns'):.0f} | {med(g, 'p99_ns'):.0f} | {med(g, 'correct'):.0f} | {med(g, 'wrong'):.0f} | "
            f"{med(g, 'lost'):.0f} | {med(g, 'detected'):.0f} | {med(g, 'retries'):.0f} | {med(g, 'crc_checks'):.0f} |")

    checks = []
    bad = [k for k, g in groups.items() if k[2] == 'station' and any(r['wrong'] for r in g)]
    checks.append(('C-1', not bad, 'istasyonlu yanlış = 0' + (f'; ihlal {bad}' if bad else '')))

    c2 = []
    for (p, q, m), g in groups.items():
        if m != 'single':
            continue
        exp = exp_wrong_single(p, q)
        obs = med(g, 'wrong')
        c2.append((p, q, obs, exp, obs == 0 if exp == 0 else 0.8 * exp <= obs <= 1.2 * exp))
    checks.append(('C-2', all(x[4] for x in c2),
                   'tek geçiş yanlış: ' + '; '.join(f'p={p},q={q}: {o:.0f}/{e:.0f}' for p, q, o, e, _ in sorted(c2))))

    c3 = []
    for (p, q, m), g in groups.items():
        if m != 'station':
            continue
        exp = exp_lost_station(p, q)
        obs = med(g, 'lost')
        c3.append((p, q, obs, exp, abs(obs - exp) <= 3 * math.sqrt(exp) + 3))
    checks.append(('C-3', all(x[4] for x in c3),
                   'istasyonlu kayıp: ' + '; '.join(f'p={p},q={q}: {o:.0f}/{e:.1f}' for p, q, o, e, _ in sorted(c3))))

    ratio = (1e6 / med(groups[(0.0, 0.0, 'station')], 'blocks_per_s')) / \
            (1e6 / med(groups[(0.0, 0.0, 'single')], 'blocks_per_s'))
    checks.append(('C-4', ratio <= 2.0, f'p=0 istasyonlu/tek geçiş süre oranı = {ratio:.2f} (eşik 2.0)'))

    md += ['', '### Blok düzeyi kontrolleri', '']
    for code, ok, detail in checks:
        md.append(f"- **{code}** {'GEÇTİ' if ok else 'ÇÜRÜDÜ'} — {detail}")
    return ratio


def e2e_part(md, raw):
    arms = defaultdict(list)
    for cond, p, q in E2E_ARMS:                                  # ısınma, atılır
        run_e2e(cond, p, q, 0)
    for rep in range(1, E2E_REPS + 1):                           # koşullar tekrarlar arasında karışık
        for cond, p, q in E2E_ARMS:
            arms[(cond, p, q)].append(run_e2e(cond, p, q, rep))
    raw['e2e'] = {f'{c}|{p}|{q}': rows for (c, p, q), rows in arms.items()}

    base = arms[('none', 0.0, 0.0)]
    base_tps = med(base, 'tg_tps_engine')
    chunks = base[0]['chunks']
    md += ['## Uçtan uca (gerçek decode, Q8_0)', '',
           f'tg{E2E_TOKENS}, BOS-only bağlam, medyan / {E2E_REPS} tekrar. Her token sonrası ağırlık havuzu ({chunks} blok, '
           f'{base[0]["pool_bytes"] / 1e6:.0f} MB) boru hattından geçer.', '',
           '| koşul | motor tok/s | toplam tok/s | doğru blok oranı | doğru tok/s | pipe ms/token | pipe p99 ms | tespit | yeniden yazma | kayıp | CRC kontrol | doğrulama yanlış | doğrulama kayıp |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    doğru = {}
    for cond, p, q in E2E_ARMS:
        g = arms[(cond, p, q)]
        label = f'{cond} p={p}' if cond != 'none' else 'taban'
        frac = 1.0 if cond == 'none' else med(g, 'ver_correct') / chunks
        tot = med(g, 'tg_tps_total')
        good = tot * frac
        doğru[(cond, p)] = good
        md.append(
            f"| {label} | {med(g, 'tg_tps_engine'):.2f} | {tot:.2f} | {frac:.4f} | {good:.2f} | "
            f"{med(g, 'pipe_ms_per_tok'):.0f} | {med(g, 'pipe_p99_ms'):.0f} | {med(g, 'detected'):.0f} | "
            f"{med(g, 'retries'):.0f} | {med(g, 'lost'):.0f} | {med(g, 'crc_checks'):.0f} | "
            f"{max(r['ver_wrong'] for r in g):.0f} | {med(g, 'ver_lost'):.0f} |")

    checks = []
    bad = [k for k, g in arms.items() if k[0] == 'station' and any(r['ver_wrong'] for r in g)]
    checks.append(('E-1', not bad, 'istasyonlu doğrulama yanlış = 0' + (f'; ihlal {bad}' if bad else '')))
    s01 = doğru[('single', 0.1)]
    t01 = doğru[('station', 0.1)]
    checks.append(('E-2', t01 > s01,
                   f'p=0.1 doğru tok/s: istasyonlu {t01:.2f} vs tek geçiş {s01:.2f}'))

    md += ['', '### Uçtan uca kontroller', '']
    for code, ok, detail in checks:
        md.append(f"- **{code}** {'GEÇTİ' if ok else 'ÇÜRÜDÜ'} — {detail}")
    md += ['', '## Karşılaştırma', '',
           f'- Bu oturumda taban: {base_tps:.2f} tok/s (tg{E2E_TOKENS}); dosya tg32: {BENCH_TG32} tok/s → {100 * base_tps / BENCH_TG32:.0f}%.',
           '- "Doğru tok/s" = toplam tok/s × doğru blok oranı; decode metnini değil, ağırlık geçişinin bütünlüğünü ölçer.']


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if part not in ('blocks', 'e2e', 'all'):
        raise SystemExit('kullanım: run_deney_c.py [blocks|e2e|all]')
    subprocess.run(['zig', 'build', '-Doptimize=ReleaseFast'], cwd=ROOT, check=True)
    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y-%m-%d-%H%M%S')

    md = [f'# Deney C — istasyonlu geçiş ({stamp})', '',
          f'Model: `{MODEL.name}`. Ön-kayıt: C-1..C-4, E-1..E-2 (tools/run_deney_c.py başında, commit edildi).', '']
    raw = {}
    if part in ('blocks', 'all'):
        block_part(md, raw)
    if part in ('e2e', 'all'):
        e2e_part(md, raw)

    stem = f'deney-c-{stamp}-{part}'
    (OUT_DIR / f'{stem}.json').write_text(json.dumps(raw, indent=1))
    text = '\n'.join(md) + '\n'
    (OUT_DIR / f'{stem}.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
