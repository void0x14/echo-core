#!/usr/bin/env python3
"""Deney B: iş parçacığı yerleşimi ring gecikmesini değiştirir mi?

ÖN-KAYIT (bu dosya koşudan önce commit edildi):
  B-1  Yerleşim etkisi var: sınıflar (tek thread, SMT kardeşi, aynı CCX farklı çekirdek, farklı CCX)
       arasında ping-pong p50 değerlerinin en büyük / en küçük oranı ≥ 1.20.
  B-2  Tutarlılık: aynı sınıftaki iki farklı çift arasında p50 farkı < %20 (yapısal etki).
  B-3  Doğrulama: pinli koşullarda gözlenen CPU maskeleri istenen çifte eşit
       (üretici = cpuA, tüketici = cpuB; tek thread = cpuA).
  B-4  Doğruluk: her koşulda yanlış blok = 0.
Sonuç kuralı (checkpoint): B-1, B-2, B-3, B-4 geçerse yerleşim etkisi yapısal ve ölçülebilir;
yazılım analoğu vardır. B-1 geçip B-2 geçmezse etki çift-özel/gürültü olabilir; "yerel kaynak
seçimi" kanıtlanmış sayılmaz. B-1 geçmezse yerleşim gecikmeyi değiştirmiyor.

Topoloji sysfs'ten okunur (/sys/devices/system/cpu), CCX sınırı L3 paylaşım listesinden çıkar.
Çalıştırma: tools/run_deney_b.py   (çıktı: benchmarks/deney-b-<stamp>.{md,json})
"""
import json
import statistics as S
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # echo-core-zig
BIN = ROOT / 'zig-out' / 'bin' / 'deney-b'
OUT_DIR = ROOT / 'benchmarks'
SYS = Path('/sys/devices/system/cpu')
REPS = 5


def parse_list(text):
    out = []
    for part in text.strip().split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(out)


def topology():
    info = {}
    for p in sorted(SYS.glob('cpu[0-9]*'), key=lambda x: int(x.name[3:])):
        c = int(p.name[3:])
        core = int((p / 'topology/core_id').read_text())
        sib = parse_list((p / 'topology/thread_siblings_list').read_text())
        l3 = None
        for idx in sorted(p.glob('cache/index*')):
            if (idx / 'level').read_text().strip() == '3':
                l3 = tuple(parse_list((idx / 'shared_cpu_list').read_text()))
        info[c] = {'core': core, 'sib': sib, 'l3': l3}
    return info


def configs(info):
    cpus = sorted(info)
    c0 = cpus[0]
    groups = sorted({info[c]['l3'] for c in cpus})
    if len(groups) < 2:
        raise SystemExit('bu makinede iki L3 grubu yok; CCX sınıfları çıkarılamadı')
    g1, g2 = groups[0], groups[1]

    def diff_core_same_ccx(group, base):
        for x in group:
            if info[x]['core'] != info[base]['core']:
                return x
        raise SystemExit(f'{base} için aynı CCX içinde farklı çekirdek yok')

    smt = [x for x in info[c0]['sib'] if x != c0]
    if not smt:
        raise SystemExit('SMT kardeşi yok')
    ccx1 = (c0, diff_core_same_ccx(g1, c0))
    ccx2 = (g2[0], diff_core_same_ccx(g2, g2[0]))
    cross_a = (c0, g2[0])
    cross_b = (max(g1), max(g2))
    return [
        {'name': 'tek-thread', 'class': 'single', 'mode': 'single', 'args': [str(c0)], 'cpus': (c0,)},
        {'name': 'smt-kardesi', 'class': 'smt', 'mode': 'pair', 'args': [str(c0), str(smt[0])], 'cpus': (c0, smt[0])},
        {'name': 'ccx1-farkli-cekirdek', 'class': 'same_ccx', 'mode': 'pair', 'args': [str(ccx1[0]), str(ccx1[1])], 'cpus': ccx1},
        {'name': 'ccx2-farkli-cekirdek', 'class': 'same_ccx', 'mode': 'pair', 'args': [str(ccx2[0]), str(ccx2[1])], 'cpus': ccx2},
        {'name': 'farkli-ccx-a', 'class': 'cross', 'mode': 'pair', 'args': [str(cross_a[0]), str(cross_a[1])], 'cpus': cross_a},
        {'name': 'farkli-ccx-b', 'class': 'cross', 'mode': 'pair', 'args': [str(cross_b[0]), str(cross_b[1])], 'cpus': cross_b},
        {'name': 'serbest-os', 'class': 'free', 'mode': 'free', 'args': [], 'cpus': None},
    ]


def run(cfg):
    cmd = [str(BIN), cfg['mode']] + cfg['args']
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise SystemExit(f"{cfg['name']} başarısız:\n{proc.stderr}")
    return json.loads([ln for ln in proc.stderr.splitlines() if ln.startswith('{')][-1])


def med(rows, key):
    return S.median(r[key] for r in rows)


def main():
    subprocess.run(['zig', 'build', '-Doptimize=ReleaseFast'], cwd=ROOT, check=True)
    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y-%m-%d-%H%M%S')

    info = topology()
    cfgs = configs(info)
    raw = {}
    for cfg in cfgs:
        run(cfg)                                                # ısınma, atılır
        raw[cfg['name']] = {'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
                            'runs': [run(cfg) for _ in range(REPS)]}

    by = {k: v['runs'] for k, v in raw.items()}
    p50 = {k: med(v, 'pp_p50_ns') for k, v in by.items()}
    p99 = {k: med(v, 'pp_p99_ns') for k, v in by.items()}
    mbs = {k: med(v, 'pipe_mb_s') for k, v in by.items()}
    cls = {c['name']: c['class'] for c in cfgs}

    class_p50 = {}
    for name, c in cls.items():
        if c != 'free':
            class_p50.setdefault(c, []).append(p50[name])
    class_med = {c: S.median(v) for c, v in class_p50.items()}
    ratio = max(class_med.values()) / min(class_med.values())
    b1 = ratio >= 1.20

    ccx = ['ccx1-farkli-cekirdek', 'ccx2-farkli-cekirdek']
    crs = ['farkli-ccx-a', 'farkli-ccx-b']
    d_ccx = abs(p50[ccx[0]] - p50[ccx[1]]) / min(p50[ccx[0]], p50[ccx[1]])
    d_crs = abs(p50[crs[0]] - p50[crs[1]]) / min(p50[crs[0]], p50[crs[1]])
    b2 = d_ccx < 0.20 and d_crs < 0.20

    b3_rows = []
    b3 = True
    for c in cfgs:
        if c['class'] == 'free':
            continue
        r0 = by[c['name']][0]
        want_p = 1 << c['cpus'][0]
        want_c = 1 << c['cpus'][-1]
        ok = r0['mask_prod'] == want_p and r0['mask_cons'] == (want_c if c['class'] != 'single' else want_p)
        b3 &= ok
        b3_rows.append(f"{c['name']}: üretici {bin(r0['mask_prod'])} / tüketici {bin(r0['mask_cons'])} → {'eşleşti' if ok else 'EŞLEŞMEDİ'}")

    mism = sum(r['pp_mism'] + r['pipe_mism'] for v in by.values() for r in v)
    b4 = mism == 0

    md = [f'# Deney B — iş parçacığı yerleşimi ve ring gecikmesi ({stamp})', '',
          'Ön-kayıt: `tools/run_deney_b.py` başındaki B-1..B-4 (commit edildi, koşudan önce).', '',
          f'Topoloji (sysfs): ' + ', '.join(f"cpu{c}: çekirdek {info[c]['core']}, L3 {info[c]['l3']}, SMT {info[c]['sib']}" for c in (0, 1, 2, 3, 6)), '',
          f'Ölçüm: ping-pong (tek blok uçuşta, N={raw[cfgs[0]["name"]]["runs"][0]["n_pp"]}) ve pipeline (64 slot, N={raw[cfgs[0]["name"]]["runs"][0]["n_pipe"]}), '
          f'4096 B blok, medyan / {REPS} tekrar (ısınma atıldı).', '',
          '| koşul | sınıf | CPU | ping-pong p50 ns | p99 ns | pipeline MB/s | yanlış |',
          '|---|---|---|---:|---:|---:|---:|']
    for c in cfgs:
        cpus = '-' if c['cpus'] is None else ','.join(str(x) for x in c['cpus'])
        errs = sum(r['pp_mism'] + r['pipe_mism'] for r in by[c['name']])
        md.append(f"| {c['name']} | {c['class']} | {cpus} | {p50[c['name']]:.0f} | {p99[c['name']]:.0f} | "
                  f"{mbs[c['name']]:.0f} | {errs} |")

    md += ['', '## Kontroller', '',
           f"- **B-1** {'GEÇTİ' if b1 else 'ÇÜRÜDÜ'} — sınıf p50 medyanları: " +
           ', '.join(f'{k} {v:.0f} ns' for k, v in class_med.items()) + f'; en büyük/en küçük = {ratio:.2f} (eşik 1.20)',
           f"- **B-2** {'GEÇTİ' if b2 else 'ÇÜRÜDÜ'} — aynı CCX çiftleri farkı {100 * d_ccx:.1f}%, farklı CCX çiftleri farkı {100 * d_crs:.1f}% (eşik %20)",
           f"- **B-3** {'GEÇTİ' if b3 else 'ÇÜRÜDÜ'} — " + '; '.join(b3_rows),
           f"- **B-4** {'GEÇTİ' if b4 else 'ÇÜRÜDÜ'} — toplam yanlış blok = {mism}", '']
    text = '\n'.join(md) + '\n'
    (OUT_DIR / f'deney-b-{stamp}.md').write_text(text)
    (OUT_DIR / f'deney-b-{stamp}.json').write_text(json.dumps({'topology': {str(k): v for k, v in info.items()}, 'raw': raw}, indent=1))
    print(text)


if __name__ == '__main__':
    main()
