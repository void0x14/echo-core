#!/usr/bin/env bash
# Tek komut: tokenizer karşılaştırması. Çıktı: SONUC.md
# Gereken: uv, g++, libllama (llama.h + libllama.so). Kütüphaneler .venv'e kurulur.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=../../../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf
CEKIRDEK=${CEKIRDEK:-3}   # ölçüm tek çekirdeğe sabitlenir
PY=.venv/bin/python

if [ ! -x "$PY" ]; then uv venv -q .venv; fi
VIRTUAL_ENV=$PWD/.venv uv pip install -q gguf numpy regex tokenizers tiktoken kitoken sentencepiece protobuf pyahocorasick

echo "[1/4] veri hazırlanıyor"; $PY hazirla.py
echo "[2/4] derleniyor";        g++ -O3 -march=native -std=c++20 -w tokbench.cpp -o _veri/tokbench -lllama
echo "[3/4] C/C++ ölçüm";        taskset -c "$CEKIRDEK" ./_veri/tokbench _veri "$MODEL" > _veri/sonuc_c.tsv
echo "[4/4] kütüphaneler";       taskset -c "$CEKIRDEK" $PY kutuphaneler.py > _veri/sonuc_py.tsv
$PY rapor.py > /dev/null
echo "bitti: $(pwd)/SONUC.md"
