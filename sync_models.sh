#!/usr/bin/env bash
# Copy the four-seat model code and the latest snapshots from ~/code/bridge_new.
#
#   ./sync_models.sh          # then ./deploy.sh
#
# The server needs the exact net classes the checkpoints were trained with, so
# the bridgezero modules are a frozen copy, like emergent/.
set -euo pipefail
SRC="${SRC:-$HOME/code/bridge_new}"
# The D5OWN4XC branch was merged into bridge_new's main on 2026-09-22, so the bidding
# code comes from the checkout itself; it used to come from a worktree that is now gone.
CODE="${CODE:-$SRC}"
cd "$(dirname "$0")"

echo "==> bridgezero code from $CODE"
rm -rf bridgezero
for f in __init__.py \
         bridge/__init__.py bridge/auction.py bridge/calls.py bridge/deals.py bridge/scoring.py \
         contract/__init__.py contract/data.py contract/environment.py contract/model.py \
         contract/prefixes.py contract/targets.py \
         cooperative/__init__.py cooperative/actor_critic.py \
         fourseat/__init__.py fourseat/model.py fourseat/state.py; do
  mkdir -p "bridgezero/$(dirname "$f")"
  cp "$CODE/bridgezero/$f" "bridgezero/$f"
done

# The card play desk (/play) always tracks $SRC: the play code lives only there.
for f in bridge/play.py play/__init__.py play/data.py play/model.py play/match.py play/search.py; do
  mkdir -p "bridgezero/$(dirname "$f")"
  cp "$SRC/bridgezero/$f" "bridgezero/$f"
done

echo "==> teaching rules (the /table rule-of-thumb card) from $SRC/experiments/teaching_D"
mkdir -p emergent/teaching
cp "$SRC/experiments/teaching_D/situations.py" "$SRC/experiments/teaching_D/rules.json" emergent/teaching/

echo "==> snapshots"
copy_ck() {  # copy_ck <checkpoint> <name>: actor weights only, no critic
  python3 - "$1" "models/$2" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
ck.pop("critic", None)
torch.save(ck, sys.argv[2])
print(f"   {sys.argv[2]}  step {ck.get('step')}")
PY
}
copy() { copy_ck "$SRC/runs/$1" "$2"; }   # copy <run>/<file> <name>, from bridge_new
copy E46_brlstyle_league_l1.0_40k/ckpt_step40000.pt E46_s40k.pt   # +0.89 IMP/board vs E28 s50k

echo "==> bidding snapshots from bridge_public (the light-opening runs; /debug's menu)"
PUBLIC=${PUBLIC:-$HOME/code/bridge_public}
copy_ck "$PUBLIC/runs/box_pull/g_s2o_hi/3/ckpt_step60000.pt" hi3_s60k.pt          # +0.35 vs D, light openings
copy_ck "$PUBLIC/runs/exp_20260930/g_s2o_hi3_lo/ckpt_step28000.pt" lo_s28k.pt     # light-opening penalty

echo "==> /bench analysis (frozen copies of bridge_public/tools)"
cp "$PUBLIC/tools/weakspots.py" "$PUBLIC/tools/simplicity.py" emergent/analysis/

echo "==> card play desk"
copy play_E48_wideleagueH/last.pt play_E48_wideleagueH.pt  # +0.33 IMP/board over league E
copy play_E48_leagueE/last.pt play_E48_leagueE.pt   # +6.03 IMP/board over random
# The frozen benchmark: its auctions come with real dd_tricks, so /play can show a
# board the E48 numbers were actually measured on. 4 MB; the page reads the first
# PLAY_BENCH_LIMIT rows (4000 by default).
cp "$SRC/data/play/bench_100k.npz" models/bench_100k.npz
python3 -c "import json;print('   models/play_models.json:', [m['id'] for m in json.load(open('models/play_models.json'))])"

# models/corpus_<id>.json ("what this call meant") is built offline, not copied here:
#   python selfplay.py CKPT selfplay.npz 1000000       # dealer North, no vul, greedy self-play
#   python corpus.py selfplay.npz models/corpus_<id>.json 200 8
# Both scripts live in bridge_new/experiments/explain/.
