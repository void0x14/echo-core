#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core')
ZIG_DIR = ROOT / 'echo-core-zig'
MODELS_DIR = Path('/home/void0x14/Desktop/models')
ECHO_BIN = ZIG_DIR / 'echo-core-zig'
LLAMA_BENCH = Path('/home/void0x14/llama.cpp/build/bin/llama-bench')
PROMPT = 'Hello'
MAX_TOKENS = '1'
ECHO_TIMEOUT = 180
LLAMA_TIMEOUT = 180
STAMP = time.strftime('%Y-%m-%d-%H%M%S')
JSON_OUT = ZIG_DIR / 'benchmarks' / f'bench-matrix-{STAMP}.json'
MD_OUT = ZIG_DIR / 'benchmarks' / f'bench-matrix-{STAMP}.md'


def run_command(cmd, cwd, timeout):
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            'status': 'ok' if proc.returncode == 0 else 'error',
            'returncode': proc.returncode,
            'stdout': proc.stdout,
            'stderr': proc.stderr,
            'elapsed_s': round(time.time() - started, 3),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            'status': 'timeout',
            'returncode': None,
            'stdout': exc.stdout or '',
            'stderr': exc.stderr or '',
            'elapsed_s': round(time.time() - started, 3),
        }


def parse_echo(stdout, stderr):
    for text in (stdout.strip(), stderr.strip()):
        if not text:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return None


def parse_llama(stdout):
    lines = [line for line in stdout.splitlines() if not line.startswith('ggml_vulkan:')]
    text = '\n'.join(lines).strip()
    if not text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    result = {}
    for item in data:
        if item.get('n_prompt', 0) > 0:
            result['pp_tps'] = item.get('avg_ts')
            result['pp_ns'] = item.get('avg_ns')
            result['model_type'] = item.get('model_type')
        if item.get('n_gen', 0) > 0:
            result['tg_tps'] = item.get('avg_ts')
            result['tg_ns'] = item.get('avg_ns')
            result['model_type'] = item.get('model_type')
    return result or None


def safe_float(v):
    if v is None:
        return ''
    return f'{v:.2f}'


def build_markdown(rows):
    out = []
    out.append('# Benchmark Matrix')
    out.append('')
    out.append(f'- prompt: `{PROMPT}`')
    out.append(f'- max tokens: `{MAX_TOKENS}`')
    out.append(f'- echo timeout: `{ECHO_TIMEOUT}s`')
    out.append(f'- llama timeout: `{LLAMA_TIMEOUT}s`')
    out.append('')
    out.append('| Model | Echo Status | Echo PP | Echo TG | Llama Status | Llama PP | Llama TG | Notes |')
    out.append('| --- | --- | ---: | ---: | --- | ---: | ---: | --- |')
    for row in rows:
        notes = []
        if row['echo']['status'] != 'ok':
            notes.append(f"echo:{row['echo']['status']}")
        if row['llama']['status'] != 'ok':
            notes.append(f"llama:{row['llama']['status']}")
        if row['echo'].get('parsed') is None and row['echo']['status'] == 'ok':
            notes.append('echo:unparsed')
        if row['llama'].get('parsed') is None and row['llama']['status'] == 'ok':
            notes.append('llama:unparsed')
        out.append(
            '| {model} | {echo_status} | {echo_pp} | {echo_tg} | {llama_status} | {llama_pp} | {llama_tg} | {notes} |'.format(
                model=row['model_name'],
                echo_status=row['echo']['status'],
                echo_pp=safe_float((row['echo'].get('parsed') or {}).get('pp_tps')),
                echo_tg=safe_float((row['echo'].get('parsed') or {}).get('tg_tps')),
                llama_status=row['llama']['status'],
                llama_pp=safe_float((row['llama'].get('parsed') or {}).get('pp_tps')),
                llama_tg=safe_float((row['llama'].get('parsed') or {}).get('tg_tps')),
                notes=', '.join(notes),
            )
        )
    out.append('')
    out.append('## Raw Status')
    out.append('')
    for row in rows:
        out.append(f"### {row['model_name']}")
        out.append('')
        out.append(f"- echo status: `{row['echo']['status']}` elapsed `{row['echo']['elapsed_s']}s`")
        out.append(f"- llama status: `{row['llama']['status']}` elapsed `{row['llama']['elapsed_s']}s`")
        if row['echo']['status'] != 'ok' or row['echo'].get('parsed') is None:
            stderr = (row['echo']['stderr'] or row['echo']['stdout']).strip()
            if stderr:
                out.append(f"- echo detail: `{stderr[:400].replace('`', '\\`')}`")
        if row['llama']['status'] != 'ok' or row['llama'].get('parsed') is None:
            stderr = (row['llama']['stderr'] or row['llama']['stdout']).strip()
            if stderr:
                out.append(f"- llama detail: `{stderr[:400].replace('`', '\\`')}`")
        out.append('')
    return '\n'.join(out)


def main():
    models = sorted(MODELS_DIR.glob('*.gguf'))
    rows = []
    for model in models:
        print(f'[{len(rows)+1}/{len(models)}] {model.name}', flush=True)
        echo_cmd = [
            str(ECHO_BIN),
            '--model', str(model),
            '--prompt', PROMPT,
            '--max-tokens', MAX_TOKENS,
            '--bench',
            '--json',
        ]
        llama_cmd = [
            str(LLAMA_BENCH),
            '-m', str(model),
            '-p', '2',
            '-n', '1',
            '-ngl', '0',
            '-r', '1',
            '--no-warmup',
            '-o', 'json',
        ]

        echo = run_command(echo_cmd, ZIG_DIR, ECHO_TIMEOUT)
        echo['parsed'] = parse_echo(echo['stdout'], echo['stderr'])

        llama = run_command(llama_cmd, ROOT, LLAMA_TIMEOUT)
        llama['parsed'] = parse_llama(llama['stdout'])

        rows.append({
            'model_name': model.name,
            'model_path': str(model),
            'echo': echo,
            'llama': llama,
        })

        JSON_OUT.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        MD_OUT.write_text(build_markdown(rows), encoding='utf-8')

    print(JSON_OUT)
    print(MD_OUT)


if __name__ == '__main__':
    main()
