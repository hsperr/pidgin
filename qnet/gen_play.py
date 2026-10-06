"""Play out auctions with the policy net (greedy (+eps random legal card) and DD-value every legal card
at every position. python gen_play.py AUC.npz OUT.pt EPS
Saves: seq (n,52) int8 cards in play order; vals (n,52,52) int8 tricks for the side on turn, -1 illegal;
plus the contracts' source path. Positions are rebuilt in the trainer by replaying seq."""
import os, sys, time, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.play.data import load_contracts
from training.play.match import NetPlayer
from training.bridge.play import PlayBatch
from training.play import search
from endplay.dds import solve_board
torch.set_num_threads(1)
inp, out, eps = sys.argv[1], sys.argv[2], float(sys.argv[3])
c = load_contracts(inp)
pl = NetPlayer(os.environ.get("PLAY_MODEL", str(ROOT / "server" / "models" / "play_E48_wideleagueH.pt"))); pl.start(c)
b = PlayBatch(c.owner, c.trump, c.declarer)
deals = search.make_deals(c)
vals = torch.full((len(c), 52, 52), -1, dtype=torch.int8)
g = torch.Generator().manual_seed(abs(hash(inp)) % 2**31)
t0 = time.time()
for step in range(52):
    legal = b.legal()
    ch = pl.choose(b, c, legal, step)
    rnd = torch.multinomial(legal.float(), 1, generator=g).squeeze(1)
    ch = torch.where(torch.rand(len(c), generator=g) < eps, rnd, ch)
    for i, d in enumerate(deals):
        for card, v in solve_board(d):
            vals[i, step, search.from_card(card)] = v
        d.play(search.to_card(int(ch[i])))
    assert ((vals[:, step] >= 0) == legal).all(), f"solver/legal mismatch at step {step}"
    b.play(ch)
torch.save({"seq": b.history.to(torch.int8), "vals": vals, "src": inp}, out + ".tmp")
import os; os.replace(out + ".tmp", out)
print(f"{inp}: {len(c)} deals, {time.time()-t0:.0f}s, declarer tricks mean {b.declarer_tricks().float().mean():.2f}", flush=True)
