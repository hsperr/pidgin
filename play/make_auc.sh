#!/bin/bash
cd "$(dirname "$0")"
for i in $(seq -w 0 599); do [ -e auc/tr$i.npz ] || python3 gen_d.py auc/tr$i.npz 2000 $((62000000 + 10#$i * 2000)) >> gen_auc.log 2>&1; done
