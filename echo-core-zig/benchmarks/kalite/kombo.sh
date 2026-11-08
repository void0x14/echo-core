#!/bin/bash
# kombo.sh "K:skip" ... → kalite + pp512 (+ TG varsa TG=1)
O=$(dirname "$0"); M=/home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf
for C in "$@"; do K=${C%%:*}; S=${C#*:}
  bash $O/kalite.sh x ECHO_FFN_KEEP=$K ECHO_FFN_ACT=$O/act.bin ECHO_SKIP=$S | sed "s/^x/K${K} [${S}]/"
  a=$(ECHO_FFN_KEEP=$K ECHO_FFN_ACT=$O/act.bin ECHO_SKIP=$S $O/n6 --llama-bench -m $M -r 10 -p 512 -n 0 2>&1 | grep -oE '"avg_ts":[0-9.]+')
  t=""; [ -n "$TG" ] && t=$(ECHO_FFN_KEEP=$K ECHO_FFN_ACT=$O/act.bin ECHO_SKIP=$S $O/n6 --llama-bench -m $M -r 10 -p 0 -n 128 --grammar 0 --lm-ivf 4096,128 2>&1 | grep -oE '"accepted":[0-9]+|"avg_ts":[0-9.]+' | tr '\n' ' ')
  echo "   pp512 $a  TG $t load=$(cut -d' ' -f1 /proc/loadavg)"
done
