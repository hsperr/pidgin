#!/usr/bin/env bash
# Auctions for the Q-net: Pidgin V1 bids greedily on deals from the DDS dataset.
# A 2,000-deal test shard, then 600 training shards of 2,000 deals (qnet/auc/).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p auc
DATA="${BRIDGE_DATA:-../data/dds_results_100M.npy}"
gen() { python ../scripts/generate_auctions.py --model PidginV1 --data "$DATA" "$@"; }
[ -e auc/test.npz ] || gen --n 2000 --start 61990000 --out auc/test.npz >> gen_auc.log 2>&1
for i in $(seq -w 0 599); do
  [ -e auc/tr$i.npz ] || gen --n 2000 --start $((62000000 + 10#$i * 2000)) --seed $((10#$i + 1)) \
    --out auc/tr$i.npz >> gen_auc.log 2>&1
done
