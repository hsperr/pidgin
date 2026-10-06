#!/usr/bin/env bash
# Turn auction shards into training positions (gen_play.py): test shard first with no
# random cards, training shards with 20% random cards. Several workers can run at once.
cd "$(dirname "$0")"
while true; do
  did=0
  for f in auc/test.npz auc/tr*.npz; do
    s=$(basename ${f%.npz}); [ -e pos/$s.pt ] && continue
    mkdir pos/$s.lock 2>/dev/null || continue
    eps=0.2; [ $s = test ] && eps=0
    python3 gen_play.py $f pos/$s.pt $eps >> pos/worker.log 2>&1; did=1; break
  done
  if [ $did = 0 ]; then pgrep -f make_auc.sh >/dev/null || exit 0; /bin/sleep 20; fi
done
