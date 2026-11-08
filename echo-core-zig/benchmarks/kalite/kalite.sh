#!/bin/bash
# kalite.sh ad [ENV=..] → logits_<ad>.bin (256 adım), sonra kalite.py ile kıyas
O=$(dirname "$0"); ad=$1; shift
env "$@" $O/n6 --llama-bench --dump-logits $O/logits_$ad.bin -m /home/void0x14/Documents/Cachyos-Documents-Backup/Vibe-Coding-Projects/echo-core/ArastirmaModelleri/Qwen3.5-0.8B-Q8_0.gguf --tokens $(cat $O/tok.txt) --steps $(python3 -c "print(','.join(['1']*768))") --ctx 1024 >/dev/null 2>&1 || echo HATA
python3 -I $O/kalite.py $O $ad
