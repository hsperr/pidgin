#!/bin/bash
# claim auction shards without positions (test first, eps 0; train eps 0.2); exit when all done
cd "$(dirname "$0")"
while true; do
  did=0
  for f in auc/test.npz auc/tr*.npz; do
    s=$(basename ${f%.npz}); [ -e pos/$s.pt ] && continue
    mkdir pos/$s.lock 2>/dev/null || continue
    eps=0.2; [ $s = test ] && eps=0
    python3 gen_play.py $f pos/$s.pt $eps >> pos/worker.log 2>&1; did=1; break
  done
  if [ $did = 0 ]; then pgrep -f make_auc.sh >/dev/null || exit 0; sleep 20; fi
done
