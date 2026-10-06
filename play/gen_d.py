"""D greedy auctions on random deals -> npz in gen_auctions.py format.
python gen_d.py OUT.npz N DEAL_START"""
import os, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.match import play, FourSeatPlayer
from training.contract.data import load_range

out, n_deals, start0 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(4)
D = FourSeatPlayer(os.environ.get("BID_MODEL", str(ROOT / "server" / "models" / "D_cw_s75k.pt")))
# DDS_FILE/DDS_OFFSET: a slice of the 100M file whose row 0 is deal DDS_OFFSET (the GPU box has only a slice)
import os
DDS, OFF = os.environ.get("DDS_FILE", os.environ.get("BRIDGE_DATA", str(ROOT / "data" / "dds_results_100M.npy"))), int(os.environ.get("DDS_OFFSET", 0))
deals = load_range(DDS, start0 - OFF, n_deals)
g = torch.Generator().manual_seed(start0)
dealer = torch.randint(4, (n_deals,), generator=g)
vns = torch.randint(2, (n_deals,), generator=g).bool()
vew = torch.randint(2, (n_deals,), generator=g).bool()
hist, t0, B = [], time.time(), 4096
for s in range(0, n_deals, B):
    idx = torch.arange(s, min(s + B, n_deals))
    tab = play([D], deals, idx, dealer[idx], vns[idx], vew[idx], torch.zeros(len(idx), 2, dtype=torch.long))
    hist.append(tab.history.numpy())
    print(f"{s+len(idx)}/{n_deals} {time.time()-t0:.0f}s", flush=True)
w = max(h.shape[1] for h in hist)
calls = np.full((n_deals, w), -1, np.int8); r = 0
for h in hist:
    calls[r:r+len(h), :h.shape[1]] = h; r += len(h)
H = deals.hands.numpy()  # (n,4,52)?
print("hands shape", H.shape, "tricks", tuple(deals.tricks.shape))
np.savez(out, calls=calls, hands=H.astype(np.uint8), dealer=dealer.numpy().astype(np.int8),
         vul_ns=vns.numpy(), vul_ew=vew.numpy(), tricks=deals.tricks.numpy().astype(np.int8),
         deal_first=start0)
