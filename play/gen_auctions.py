"""Auctions for the card-play trainer: sampled calls, all dealers, all vulnerabilities.

    python gen_auctions.py CKPT OUT.npz N_DEALS [BATCH] [DEAL_START]

Unlike ``experiments/explain/selfplay.py`` (bid desk: dealer North, no vul, greedy) this
samples from the policy, so contracts spread out and the play net also meets bad ones.

Writes, all in deal order:
    calls    (N, MAX) int8, padded with -1
    hands    (N, 4, 52) uint8, seat order N E S W
    dealer   (N,) int8
    vul_ns   (N,) bool
    vul_ew   (N,) bool
    temp     (N,) float32, the sampling temperature that deal's auction used
    tricks   (N, 4, 5) int8, the double dummy table (the yardstick, never an input)
    deal_first  int, index of deal 0 inside dds_results_100M.npy
"""
import sys, time
import numpy as np, torch

import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.contract.data import load_range
from training.fourseat.model import load_fourseat_checkpoint
from training.fourseat.competitive import competitive_batch_class, set_any_seat_double
from training.fourseat.model import competitive_log_probs

DEALS = os.environ.get("BRIDGE_DATA", str(ROOT / "data" / "dds_results_100M.npy"))

ckpt, out_path, n_deals = sys.argv[1], sys.argv[2], int(sys.argv[3])
batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
deal_start = int(sys.argv[5]) if len(sys.argv) > 5 else 0   # the bid desk used the tail
# E46's policy is very confident, so temperature only bites above 1. Greedy makes 64% of its
# contracts, T=3 makes 59%, T=5 makes 48%. One temperature is drawn per deal from this list.
temps = [float(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "1,3,5").split(",")]

torch.set_num_threads(4)
net, ck = load_fourseat_checkpoint(ckpt)
net.eval()
set_any_seat_double(bool(ck.get("any_seat_double", False)))
cls = competitive_batch_class(net.redouble)
deals = load_range(DEALS, deal_start, n_deals)
if deals.n < n_deals:
    raise SystemExit(f"only {deals.n} deals from {deal_start}")

gen = torch.Generator().manual_seed(20260919)
calls_out, dealer_out, vns_out, vew_out, temp_out = [], [], [], [], []
t0 = time.time()
for start in range(0, n_deals, batch_size):
    idx = torch.arange(start, min(start + batch_size, n_deals))
    n = len(idx)
    dealer = torch.randint(4, (n,), generator=gen)
    vul_ns = torch.randint(2, (n,), generator=gen).bool()
    vul_ew = torch.randint(2, (n,), generator=gen).bool()
    temp = torch.tensor(temps)[torch.randint(len(temps), (n,), generator=gen)][:, None]
    st = cls.start(idx, dealer, vul_ns, vul_ew)
    with torch.no_grad():
        while not bool(st.ended.all()):
            live = ~st.ended
            hand = deals.hands[st.deal, st.actor_seat]
            legal = st.legal()
            lp = competitive_log_probs(net(hand, st.features()), legal, 1.0, static=True)
            hot = torch.where(legal, lp / temp, torch.full_like(lp, -1e30))
            probs = torch.nan_to_num(hot.softmax(-1), nan=0.0, posinf=0.0, neginf=0.0) * legal
            dead = probs.sum(-1) <= 0          # finished rows have no legal call left
            probs[dead, 0] = 1.0               # their action is dropped by ``live`` anyway
            action = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            st.apply(action, live)
    h = st.history.numpy()
    calls_out.append(h[:, :max(1, int(st.t.max()))].astype(np.int8))
    dealer_out.append(dealer.numpy().astype(np.int8))
    vns_out.append(vul_ns.numpy())
    vew_out.append(vul_ew.numpy())
    temp_out.append(temp[:, 0].numpy().astype(np.float32))
    print(f"{start + n}/{n_deals} auctions, {time.time() - t0:.0f}s", flush=True)

width = max(c.shape[1] for c in calls_out)
calls = np.full((n_deals, width), -1, dtype=np.int8)
row = 0
for c in calls_out:
    calls[row:row + len(c), :c.shape[1]] = c
    row += len(c)
np.savez_compressed(
    out_path,
    calls=calls,
    hands=(deals.hands.numpy() > 0).astype(np.uint8),
    dealer=np.concatenate(dealer_out),
    vul_ns=np.concatenate(vns_out),
    vul_ew=np.concatenate(vew_out),
    temp=np.concatenate(temp_out),
    tricks=deals.tricks.numpy().astype(np.int8),
    deal_first=np.int64(deals.first_index),
)
print("wrote", out_path, calls.shape, f"{time.time() - t0:.0f}s")
