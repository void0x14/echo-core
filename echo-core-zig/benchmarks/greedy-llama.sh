#!/bin/bash
# echo-core greedy çıktısını llama-simple ile karşılaştırır (128 token, ham prompt)
M=../ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf; S=${TMPDIR:-/tmp}/echo-greedy; mkdir -p $S
P=("The capital of France is" "Bir zamanlar uzak bir köyde" "def fibonacci(n):" "Explain why the sky is blue in simple terms." "İstanbul'un tarihi hakkında kısa bilgi:")
for i in "${!P[@]}"; do
  llama-simple -m $M -n 128 "${P[$i]}" 2>/dev/null > $S/l$i.txt
  ./zig-out/bin/echo-core $M --raw --temp 0 -n 128 --no-stats -p "${P[$i]}" 2>/dev/null > $S/e$i.txt
  ./zig-out/bin/echo-core $M --raw --exact --temp 0 -n 128 --no-stats -p "${P[$i]}" 2>/dev/null > $S/x$i.txt
  python3 - "$S/l$i.txt" "$S/e$i.txt" "$S/x$i.txt" "${P[$i]}" <<'PY'
import sys
l=open(sys.argv[1]).read(); p=sys.argv[4]
l=l[l.find(p)+len(p):] if p in l else l
def ortak(a,b):
    n=0
    while n<min(len(a),len(b)) and a[n]==b[n]: n+=1
    return n
for ad,f in [("varsayılan",sys.argv[2]),("--exact",sys.argv[3])]:
    e=open(f).read(); n=ortak(l,e); m=min(len(l),len(e))
    print(f"{p[:28]:28} {ad:10} aynı {n}/{m} karakter", "TAM" if n>=m-1 else "")
PY
done
