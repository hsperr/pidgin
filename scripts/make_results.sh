#!/usr/bin/env bash
# Rerun the measurements behind docs/results.md. Output goes to results/.
# Needs the released weights (scripts/get_models.sh) and data/dds_results_100M.npy.
# About 10 minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
M=server/models
R=results
mkdir -p "$R"
V1=four:$M/D_cw_s75k.pt
V2=four:$M/pidginv2_bid_s40000.pt

# Bidding: duplicate matches on the 10,000 held-out deals x 16 boards, double-dummy IMPs.
"$PY" tools/match.py --a "$V2" --b "$V1"       --out "$R/v2_vs_v1"
"$PY" tools/match.py --a "$V1" --b rule:sayc   --out "$R/v1_vs_sayc"
"$PY" tools/match.py --a "$V2" --b rule:sayc   --out "$R/v2_vs_sayc"
# Punisher: V1's calls, but it doubles exactly the contracts that go down double dummy.
"$PY" tools/match.py --a "$V2" --b "punish:$M/D_cw_s75k.pt" --out "$R/v2_vs_punisher"
"$PY" tools/match.py --a "$V1" --b "punish:$M/D_cw_s75k.pt" --out "$R/v1_vs_punisher"
# Mean and standard error per board, clustered by deal (the 16 boards of a deal move together).
for m in v2_vs_v1 v1_vs_sayc v2_vs_sayc v2_vs_punisher v1_vs_punisher; do
  "$PY" - "$R/$m/boards.npz" <<'PY' | tee "$R/$m/summary.txt"
import sys, numpy as np
z = np.load(sys.argv[1])
imp, deal = z["imps_A"].astype(float), z["deal_index"]
_, inv = np.unique(deal, return_inverse=True)
per = np.bincount(inv, imp) / np.bincount(inv)
print(f"{sys.argv[1]}: {len(imp)} boards, {len(per)} deals, "
      f"IMPs/board {imp.mean():+.3f} +/- {per.std(ddof=1) / np.sqrt(len(per)):.3f} (s.e.)")
PY
done

# Openings on 100,000 held-out hands per seat; detail tables for seat 1, not vulnerable.
for name in D_cw_s75k pidginv2_bid_s40000; do
  "$PY" tools/openings.py "$M/$name.pt" --examples 0 > "$R/openings_$name.txt"
done
"$PY" tools/openings.py "brl:$M/brl_fsp_weights.npz" --examples 0 > "$R/openings_brl.txt"

# Simplicity: code words in self-play auctions.
for name in D_cw_s75k pidginv2_bid_s40000; do
  "$PY" tools/simplicity.py "$M/$name.pt" --boards 8000 > "$R/simplicity_$name.txt"
done

# Card play: duplicate on fixed contracts, the same defenders at both tables.
B=$M/bench_100k.npz
H=$M/play_E48_wideleagueH.pt
E=$M/play_E48_leagueE.pt
"$PY" -m training.play.match "$B" --challenger "$H" --reference random --opponent "$H" --deals 50000 > "$R/play_H_vs_random.txt"
"$PY" -m training.play.match "$B" --challenger "$E" --reference random --opponent "$H" --deals 50000 > "$R/play_E_vs_random.txt"
"$PY" -m training.play.match "$B" --challenger "$H" --reference "$E"   --opponent "$H" --deals 50000 > "$R/play_H_vs_E.txt"
# Belief net, by number of calls heard. Needs the test shards from belief/gen.py in
# belief/data/test/ and a training checkpoint with its args (BELIEF_CKPT); skipped otherwise.
if [ -n "${BELIEF_CKPT:-}" ] && [ -d belief/data/test ]; then
  mkdir -p "$R/belief"
  "$PY" -u belief/analyze.py "$R/belief" "$BELIEF_CKPT" 20000 > "$R/belief/analyze.log"
fi
echo "done: $R/"
