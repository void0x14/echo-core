#!/usr/bin/env python3
"""Deney D: sezai mod — keşif ile üretim katmanının ayrımı; ödül-güdümlü denetleyici.

ÖN-KAYIT (bu dosya koşudan önce commit edildi):
  Tasarım notu: adaptive (v2) tasarımı, 1 saniyelik smoke testinden sonra revize edildi
  (v1 değişim tespiti gürültüde tetiklenip keşfi aşırı artırıyordu). v1 ablasyon olarak
  `adaptive1` adıyla aynı koşularda ölçülür; kriterler revizyondan sonra sabitlendi.
  Yük fazları: [], [1,4], [1,2,4,5], [0..5] hog çekirdekleri; her biri 5 sn.
  Koşullar: static (k=6), explore (döngüsel k), fixed (k=6 + %12.5 rastgele keşif),
            adaptive (v2), adaptive1 (v1 ablasyon).
  D-1  Tüm koşullarda yanlış teslim = 0.
  D-2  Toplam (4 faz) doğru MB/s: adaptive ≥ 1.05 × static.
  D-3  En az 2 yük fazında (1..3) adaptive ≥ 1.10 × static (doğru MB/s).
  D-4  Toplam: adaptive ≥ 1.00 × fixed (sabit küçük keşif bütçesi).
  D-5  Toplam: explore < static (sürekli keşif bedeli).
  D-6  Ablasyon (bilgi): adaptive (v2) ≥ adaptive1 (v1) toplam.
  Sonuç kuralı (checkpoint): D-2, D-3, D-4 birlikte geçerse adaptif mod için kanıt vardır.
  Tek bir yük senaryosunda kazanç yetmez: dört faz ayrı değerlendirilir ve Pareto (MB/s, p99) raporlanır.
  Gürültü: %5'in altındaki farklar gürültü olarak yorumlanır.

Çalıştırma: tools/run_deney_d.py   (çıktı: benchmarks/deney-d-<stamp>.{md,json})
"""
import json
import statistics as S
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf')
BIN = ROOT / 'zig-out' / 'bin' / 'deney-d'
OUT_DIR = ROOT / 'benchmarks'
CONDS = ['static', 'explore', 'fixed', 'adaptive', 'adaptive1']
REPS = 5
PHASE_S = 5
PHASE_NAMES = ['yük yok', 'hafif (2 çekirdek)', 'orta (4 çekirdek)', 'tam (6 çekirdek)']


def run(cond, seed):
    cmd = [str(BIN), str(MODEL), cond, str(seed), str(PHASE_S)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise SystemExit(f'{cond} seed={seed} başarısız:\n{proc.stderr}')
    return json.loads([ln for ln in proc.stderr.splitlines() if ln.startswith('{')][-1])


def med(xs):
    return S.median(xs)


def main():
    subprocess.run(['zig', 'build', '-Doptimize=ReleaseFast'], cwd=ROOT, check=True)
    OUT_DIR.mkdir(exist_ok=True)
    stamp = time.strftime('%Y-%m-%d-%H%M%S')

    raw = {}
    for cond in CONDS:
        run(cond, 0)                                             # ısınma, atılır
        raw[cond] = [run(cond, r) for r in range(1, REPS + 1)]

    def thr(cond, ph):
        return med([r['fazlar'][ph]['thr_mb_s'] for r in raw[cond]])

    def tot(cond):
        return med([r['toplam_thr_mb_s'] for r in raw[cond]])

    md = [f'# Deney D — sezai mod: keşif ve üretim ayrımı ({stamp})', '',
          f'Model: `{MODEL.name}` (ffn_down bayt havuzu). 6 işçi, pencere = 2048 blok × 4 KB, '
          f'faz = {PHASE_S} sn, medyan / {REPS} tekrar (ısınma atıldı).', '',
          '## Faz bazında doğru MB/s (medyan)', '',
          '| koşul | ' + ' | '.join(PHASE_NAMES) + ' | toplam |',
          '|---|' + '---:|' * (len(PHASE_NAMES) + 1)]
    for cond in CONDS:
        md.append(f'| {cond} | ' + ' | '.join(f'{thr(cond, ph):.0f}' for ph in range(4)) + f' | {tot(cond):.0f} |')

    md += ['', '## Faz bazında p99 (pencere p99 medyanı, µs) ve keşif', '',
           '| koşul | faz | doğru MB/s | p99 µs | keşif pencere / toplam | k dağılımı (k=1..6, medyan) | yanlış |',
           '|---|---|---:|---:|---:|---|---:|']
    for cond in CONDS:
        for ph in range(4):
            rows = [r['fazlar'][ph] for r in raw[cond]]
            kh = [med([row['k'][i] for row in rows]) for i in range(6)]
            expl = med([row['kesif'] for row in rows])
            wins = med([row['pencere'] for row in rows])
            md.append(f"| {cond} | {PHASE_NAMES[ph]} | {thr(cond, ph):.0f} | "
                      f"{med([row['p99_us_med'] for row in rows]):.1f} | {expl:.0f} / {wins:.0f} | "
                      f"{', '.join(f'{x:.0f}' for x in kh)} | {sum(row['yanlis'] for row in rows)} |")

    checks = []
    wrong_total = {c: sum(r['toplam_yanlis'] for r in raw[c]) for c in CONDS}
    checks.append(('D-1', all(v == 0 for v in wrong_total.values()), f'toplam yanlış: {wrong_total}'))
    ratio_tot = tot('adaptive') / tot('static')
    checks.append(('D-2', ratio_tot >= 1.05, f'adaptive/static toplam = {ratio_tot:.3f} (eşik 1.05)'))
    good_phases = [ph for ph in (1, 2, 3) if thr('adaptive', ph) >= 1.10 * thr('static', ph)]
    checks.append(('D-3', len(good_phases) >= 2,
                   'adaptive ≥ 1.10 × static fazları: ' + (', '.join(PHASE_NAMES[p] for p in good_phases) or 'yok')))
    checks.append(('D-4', tot('adaptive') >= tot('fixed'),
                   f'adaptive {tot("adaptive"):.0f} vs fixed {tot("fixed"):.0f} MB/s'))
    checks.append(('D-5', tot('explore') < tot('static'),
                   f'explore {tot("explore"):.0f} vs static {tot("static"):.0f} MB/s'))
    checks.append(('D-6', tot('adaptive') >= tot('adaptive1'),
                   f'v2 {tot("adaptive"):.0f} vs v1 {tot("adaptive1"):.0f} MB/s (ablasyon, bilgi)'))

    md += ['', '## Kontroller', '']
    for code, ok, detail in checks:
        md.append(f"- **{code}** {'GEÇTİ' if ok else 'ÇÜRÜDÜ'} — {detail}")
    passes = all(ok for code, ok, _ in checks if code in ('D-2', 'D-3', 'D-4'))
    md += ['', f"**Sonuç kuralı (D-2 ∧ D-3 ∧ D-4):** {'geçti — adaptif mod için kanıt var' if passes else 'geçmedi — adaptif mod için kanıt yok'}."]

    text = '\n'.join(md) + '\n'
    (OUT_DIR / f'deney-d-{stamp}.md').write_text(text)
    (OUT_DIR / f'deney-d-{stamp}.json').write_text(json.dumps(raw, indent=1))
    print(text)


if __name__ == '__main__':
    main()
