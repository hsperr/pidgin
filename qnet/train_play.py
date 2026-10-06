"""Play Q-net from solver values: every card of the play, from scratch.
    python train_play.py --run B            (own hand, visible hand, played cards, contract, compact auction)
    python train_play.py --run B1 --belief  (+ rC auction-belief for the player on turn, 171 numbers)
    python train_play.py --run B2 --bhead --init runs/play_q/B/last.pt  (+ belief head: who holds each hidden
        card, and each hidden hand's suit lengths + HCP, from the actor's seat; for PIMC sampling, shape-first too)
Target per legal card: DD tricks for the side on turn minus the best legal card's (<= 0). MSE.
Logs to runs/play_q/<run>/train_log.jsonl every --every steps (dashboard picks it up)."""
import sys, glob, json, os, time, argparse
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.play.data import load_contracts, N_CALLS
from training.bridge.play import PlayBatch

p = argparse.ArgumentParser()
p.add_argument("--run", required=True); p.add_argument("--belief", action="store_true")
p.add_argument("--every", type=int, default=2000); p.add_argument("--bs", type=int, default=1024)
p.add_argument("--width", type=int, default=1024); p.add_argument("--depth", type=int, default=3)
p.add_argument("--drop", type=float, default=0.1); p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--threads", type=int, default=3)
p.add_argument("--bhead", action="store_true"); p.add_argument("--bw", type=float, default=0.1)
p.add_argument("--init", default=""); p.add_argument("--device", default="cpu")
p.add_argument("--step0", type=int, default=0)     # continue a run's step count (with --init from its last.pt)
p.add_argument("--reload", type=int, default=1000); p.add_argument("--steps", type=int, default=10**9)
a = p.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)
DEV = torch.device(a.device)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(ROOT / "runs" / "play_q" / a.run); os.makedirs(OUT, exist_ok=True)
LAST = 6

if a.belief:
    sys.path.insert(0, str(ROOT / "belief"))
    import train as T
    from train_shape import load_shape
    BNET, _ = load_shape(os.environ.get("SHAPE_NET", str(ROOT / "runs" / "belief" / "rC" / "last.pt")), "cpu")
    D75 = T.SYSTEMS.index("D75")

    @torch.no_grad()
    def belief(c):
        """(n, 4, 171) rC features per viewer seat (absolute), full auction, both sides told D75."""
        n = len(c); hist = torch.full((n, T.T_MAX), -1, dtype=torch.int8)
        w = min(T.T_MAX, c.calls.shape[1]); h = c.calls[:, :w].clone(); h[h >= N_CALLS] = -1
        hist[:, :w] = h.to(torch.int8)
        d = {"hist": hist, "dealer": c.dealer.to(torch.int8), "vul": (c.vul_ns.long() + 2 * c.vul_ew.long()).to(torch.int8),
             "owners": c.owner.to(torch.int8), "ns_sys": torch.full((n,), D75, dtype=torch.int8),
             "ew_sys": torch.full((n,), D75, dtype=torch.int8)}
        out = torch.zeros(n, 4, 171, dtype=torch.float16); g = torch.Generator().manual_seed(0)
        idx = torch.arange(n)
        for seat in range(4):
            bb = T.make_batch(d, idx, g, sys_drop=0.0, viewer=torch.tensor([seat]), prefix="full")
            ol, _ = BNET(bb)
            cards = T.sinkhorn(ol, bb["hand"]).exp() * (bb["hand"] < .5)[..., None]
            hl, ll = T.split_summary(BNET.summary)
            ehcp = (hl.softmax(-1) * torch.arange(T.HCP_BINS)).sum(-1)
            elen = (ll.softmax(-1) * torch.arange(14)).sum(-1).flatten(1)
            out[:, seat] = torch.cat([cards.flatten(1), ehcp / 10, elen / 4], 1).half()
        return out

def load_shard(path):
    """Per deal: everything needed to rebuild any position's features."""
    z = torch.load(path); src = z["src"] if os.path.isabs(z["src"]) else os.path.join(HERE, z["src"])
    c = load_contracts(src); seq = z["seq"].long(); n = len(c)
    b = PlayBatch(c.owner, c.trump, c.declarer)
    turn = torch.zeros(n, 52, dtype=torch.int8); won = torch.zeros(n, 52, 2, dtype=torch.int8)
    for t in range(52):
        turn[:, t] = b.to_play(); won[:, t] = b.tricks_won; b.play(seq[:, t])
    inv = torch.zeros(n, 52, dtype=torch.int8); inv.scatter_(1, seq, torch.arange(52, dtype=torch.int8).expand(n, 52).contiguous())
    bag = torch.zeros(n, 4, N_CALLS, dtype=torch.bool)       # absolute seat x call
    for i in range(c.calls.shape[1]):
        live = (i < c.n_calls).nonzero().squeeze(1)
        bag[live, (c.dealer[live] + i) % 4, c.calls[live, i]] = True
    last = torch.full((n, LAST), N_CALLS, dtype=torch.int8)
    for k in range(LAST):
        j = c.n_calls - 1 - k; ok = j >= 0
        last[ok, k] = c.calls[ok, j[ok]].to(torch.int8)
    d = {"owner": c.owner.to(torch.int8), "inv": inv, "turn": turn, "won": won, "vals": z["vals"],
         "decl": c.declarer.to(torch.int8), "trump": c.trump.to(torch.int8), "level": c.level.to(torch.int8),
         "dbl": c.doubled.to(torch.int8), "vns": c.vul_ns, "vew": c.vul_ew, "dealer": c.dealer.to(torch.int8),
         "bag": bag, "last": last}
    if a.belief: d["bel"] = belief(c)
    return d

def features(d, i, t):
    """Rows i (deal index), t (card number 0..51) -> (x, legal, target)."""
    owner, inv = d["owner"][i].long(), d["inv"][i].long()
    turn = d["turn"][i, t].long(); decl = d["decl"][i].long(); dummy = (decl + 2) % 4
    actor = torch.where(turn == dummy, decl, turn)                    # declarer plays dummy
    played = inv < t[:, None]
    hand = (owner == turn[:, None]) & ~played
    other = torch.where((turn == decl) | (turn == dummy), torch.where(turn == decl, dummy, decl), dummy)
    vis = (owner == other[:, None]) & ~played & (t > 0)[:, None]
    rel_owner = (owner - turn[:, None]) % 4
    played_by = F.one_hot(rel_owner, 4).bool() & played[..., None]                       # (B,52,4)
    start = (t // 4) * 4
    in_trick = played & (inv >= start[:, None])
    trick_pos = F.one_hot((inv - start[:, None]).clamp(0, 2), 3).bool() & in_trick[..., None]  # (B,52,3)
    role = F.one_hot((turn - decl) % 4, 4)
    side_t = turn % 2; won = d["won"][i, t].long()
    mine, theirs = won.gather(1, side_t[:, None]), won.gather(1, 1 - side_t[:, None])
    need = (d["level"][i].long() + 6)[:, None]
    vns, vew = d["vns"][i], d["vew"][i]
    vo = torch.where(side_t == 0, vns, vew).float()[:, None]; vt = torch.where(side_t == 0, vew, vns).float()[:, None]
    rot = (torch.arange(4, device=i.device)[None] + actor[:, None]) % 4                 # relative seat r -> absolute
    bag = d["bag"][i].gather(1, rot[..., None].expand(-1, -1, N_CALLS))
    x = [hand, vis, played_by.flatten(1), trick_pos.flatten(1), F.one_hot(t // 4, 13), F.one_hot(t % 4, 4), role,
         F.one_hot(d["trump"][i].long(), 5), F.one_hot(d["level"][i].long() - 1, 7), F.one_hot(d["dbl"][i].long(), 3),
         vo, vt, mine / 13, theirs / 13, need / 13, bag.flatten(1),
         F.one_hot(d["last"][i].long(), N_CALLS + 1).flatten(1), F.one_hot((d["dealer"][i].long() - actor) % 4, 4)]
    if a.belief: x.append(d["bel"][i, actor].float())
    x = torch.cat([z.float() for z in x], 1)
    v = d["vals"][i, t].long(); legal = v >= 0
    tgt = (v - v.max(1, keepdim=True).values).float()
    return x, legal, tgt

HCP_OF = torch.tensor([4, 3, 2, 1] + [0] * 9).repeat(4)               # card index = suit * 13 + rank, rank 0 = ace

def belief_targets(d, i, t):
    """Actor's view (declarer when dummy plays). Seats relative to the actor, as in the auction bag.
    card: (B,52) rel owner of each card; cmask: hidden cards (not own, not visible, not played).
    lens: (B,4,4) original suit lengths per rel seat; hcp: (B,4) original HCP; smask: (B,4) hidden seats."""
    owner, inv = d["owner"][i].long(), d["inv"][i].long()
    turn = d["turn"][i, t].long(); decl = d["decl"][i].long(); dummy = (decl + 2) % 4
    actor = torch.where(turn == dummy, decl, turn)
    rel = (owner - actor[:, None]) % 4
    played = inv < t[:, None]
    # seen whole from the actor's chair: itself, plus dummy (or declarer, for dummy's view) once the lead is out
    seen = torch.zeros(len(i), 4, dtype=torch.bool, device=i.device); seen[:, 0] = True
    dummy_rel = (dummy - actor) % 4
    seen[torch.arange(len(i), device=i.device), dummy_rel] |= (t > 0)                                 # everyone sees dummy
    cmask = ~seen.gather(1, rel) & ~played
    suit = torch.arange(52, device=i.device) // 13
    lens = torch.zeros(len(i), 4, 4, dtype=torch.long, device=i.device)
    lens.view(len(i), 16).scatter_add_(1, rel * 4 + suit[None], torch.ones_like(rel))
    hcp = torch.zeros(len(i), 4, dtype=torch.long, device=i.device).scatter_add_(
        1, rel, HCP_OF.to(i.device).expand(len(i), 52).contiguous())
    return rel, cmask, lens, hcp, ~seen

def belief_loss(out, bt):
    """Mean CE: hidden cards (owner among 4 rel seats) + 0.25 x (suit lengths + HCP of hidden seats)."""
    card_l, len_l, hcp_l = out; rel, cmask, lens, hcp, smask = bt
    card_l = card_l.masked_fill(~smask[:, None, :], -1e4)                # owner is one of the hidden seats
    ce_c = F.cross_entropy(card_l.reshape(-1, 4), rel.reshape(-1), reduction="none").view_as(rel)
    ce_c = (ce_c * cmask).sum() / cmask.sum().clamp(min=1)
    ce_l = F.cross_entropy(len_l.reshape(-1, 14), lens.reshape(-1), reduction="none").view(-1, 4, 4).mean(2)
    ce_h = F.cross_entropy(hcp_l.reshape(-1, 38), hcp.reshape(-1), reduction="none").view(-1, 4)
    ce_s = ((ce_l + ce_h) * smask).sum() / smask.sum().clamp(min=1)
    return ce_c, ce_l, ce_h, ce_s

def cat(ds):
    return {k: torch.cat([d[k] for d in ds]) for k in ds[0]}

def log(row):
    with open(f"{OUT}/train_log.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)

test = {k: v.to(DEV) for k, v in load_shard(f"{HERE}/pos/test.pt").items()}; nt = len(test["owner"])
ti = torch.arange(nt, device=DEV).repeat_interleave(52); tt = torch.arange(52, device=DEV).repeat(nt)
TX, TL, TT = [], [], []
for s in range(0, len(ti), 8192):
    x, l, tg = features(test, ti[s:s + 8192], tt[s:s + 8192]); TX.append(x); TL.append(l); TT.append(tg)
TX, TL, TT = torch.cat(TX), torch.cat(TL), torch.cat(TT)
if a.bhead: TB = [torch.cat(z) for z in zip(*(belief_targets(test, ti[s:s + 8192], tt[s:s + 8192])
                                              for s in range(0, len(ti), 8192)))]
t_turn = test["turn"][ti, tt].long(); t_decl = test["decl"][ti].long()
KIND = torch.where(tt == 0, 0, torch.where((t_turn - t_decl) % 2 == 0, 2, 1))   # 0 lead, 1 defence, 2 declarer
NCH = TL.sum(1)
D_IN = TX.shape[1]
# E48 picked every test card (eps 0), so its choice at each test position is the card in seq
E48_CH = torch.load(f"{HERE}/pos/test.pt")["seq"].long().to(DEV)[ti, tt]
E48_LOST = -TT.gather(1, E48_CH[:, None]).squeeze(1)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.inp = nn.Linear(D_IN, a.width)
        s.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(a.width), nn.Linear(a.width, a.width), nn.GELU(),
                                               nn.Dropout(a.drop), nn.Linear(a.width, a.width)) for _ in range(a.depth))
        s.out = nn.Sequential(nn.LayerNorm(a.width), nn.Linear(a.width, 52))
        if a.bhead:     # card owner (52 x 4 rel seats), suit lengths (4 seats x 4 suits x 0..13), HCP (4 x 0..37)
            s.bel = nn.Sequential(nn.LayerNorm(a.width), nn.Linear(a.width, 52 * 4 + 16 * 14 + 4 * 38))
    def forward(s, x, aux=False):
        h = s.inp(x)
        for blk in s.blocks: h = h + blk(h)
        if not aux: return s.out(h)
        b = s.bel(h); n = len(x)
        return s.out(h), (b[:, :208].view(n, 52, 4), b[:, 208:432].view(n, 4, 4, 14), b[:, 432:].view(n, 4, 38))

net = Net().to(DEV)
if a.init:      # warm start: trunk + Q head from an earlier run, new heads from scratch
    miss = net.load_state_dict(torch.load(a.init, weights_only=False, map_location=DEV)["net"], strict=False)
    print("init from", a.init, "missing:", miss.missing_keys, flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
json.dump(vars(a) | {"input_dim": D_IN}, open(f"{OUT}/run_args.json", "w"), indent=1)

@torch.no_grad()
def evaluate(step, loss_avg, n_deals, t0):
    net.eval(); Q = torch.cat([net(TX[s:s + 8192]) for s in range(0, len(TX), 8192)])
    bel_row = {}
    if a.bhead:
        tot = torch.zeros(4); cnt = torch.zeros(4); acc = torch.zeros(4); lsum = hsum = hmae = sn = 0.0
        for s in range(0, len(TX), 8192):
            _, out = net(TX[s:s + 8192], aux=True); bt = [z[s:s + 8192] for z in TB]
            rel, cmask, lens, hcp, smask = bt
            out = (out[0].masked_fill(~smask[:, None, :], -1e4),) + out[1:]
            ce = F.cross_entropy(out[0].reshape(-1, 4), rel.reshape(-1), reduction="none").view_as(rel)
            hit = (out[0].argmax(-1) == rel).float()
            tr = (tt[s:s + 8192] // 4).clamp(max=12)
            for k, (lo, hi) in enumerate(((0, 1), (1, 3), (3, 6), (6, 13))):
                m = cmask & ((tr >= lo) & (tr < hi))[:, None]
                tot[k] += (ce * m).sum().item(); acc[k] += (hit * m).sum().item(); cnt[k] += m.sum().item()
            _, ce_l, ce_h, _ = belief_loss(out, bt)
            lsum += (ce_l * smask).sum().item(); hsum += (ce_h * smask).sum().item(); sn += smask.sum().item()
            eh = (out[2].softmax(-1) * torch.arange(38, device=DEV)).sum(-1)
            hmae += ((eh - hcp).abs() * smask).sum().item()
        for k, nm in enumerate(("t1", "t2_3", "t4_6", "t7_13")):
            bel_row[f"belief.card_nll.{nm}"] = (tot[k] / cnt[k]).item()
            bel_row[f"belief.card_acc.{nm}"] = (acc[k] / cnt[k]).item()
        bel_row |= {"belief.len_nll": lsum / sn, "belief.hcp_nll": hsum / sn, "belief.hcp_mae": hmae / sn}
    ch = Q.masked_fill(~TL, -1e9).argmax(1)
    lost = -TT.gather(1, ch[:, None]).squeeze(1)
    best = lost == 0
    se = ((Q - TT) ** 2 * TL).sum(1) / TL.sum(1)
    row = {"step": step, "train.loss": loss_avg, "train.deals": n_deals, "train.positions": n_deals * 52,
           "time_h": (time.time() - t0) / 3600}
    multi = NCH > 1                                  # only decisions with a real choice
    for k, name in enumerate(("lead", "defence", "declarer")):
        m = (KIND == k) & multi
        row[f"tricks_lost.{name}"] = lost[m].mean().item()
        row[f"best_card.{name}"] = best[m].float().mean().item()
        row[f"e48.tricks_lost.{name}"] = E48_LOST[m].mean().item()
        row[f"e48.best_card.{name}"] = (E48_LOST[m] == 0).float().mean().item()
    row["tricks_lost.per_deal"] = lost.sum().item() / nt
    row["e48.tricks_lost.per_deal"] = E48_LOST.sum().item() / nt
    row["best_card.all"] = best[multi].float().mean().item()
    for lo, hi, nm in ((0, 1, "t1"), (1, 3, "t2_3"), (3, 6, "t4_6"), (6, 13, "t7_13")):
        m = (tt // 4 >= lo) & (tt // 4 < hi) & multi
        row[f"test_mse.{nm}"] = se[m].mean().item()
    log(row | bel_row); net.train()

data, host, seen, t0 = None, None, set(), time.time()
loss_sum, loss_n = 0.0, 0
for step in range(a.step0 + 1, a.steps + 1):
    if step == a.step0 + 1 or step % a.reload == 0:
        new = [f for f in sorted(glob.glob(f"{HERE}/pos/tr*.pt")) if f not in seen]
        if new:
            parts = []
            for f in new:
                try: parts.append(load_shard(f)); seen.add(f)
                except Exception as e: print("skip", f, e, flush=True)
            if parts:
                host = cat(([host] if host else []) + parts)
                data = None                                 # free the old device copy first: two do not fit
                if DEV.type == "cuda": torch.cuda.empty_cache()
                data = {k: v.to(DEV) for k, v in host.items()}
        while data is None:
            time.sleep(30); new = sorted(glob.glob(f"{HERE}/pos/tr*.pt"))
            if new: host = load_shard(new[0]); data = {k: v.to(DEV) for k, v in host.items()}; seen.add(new[0])
    n = len(data["owner"])
    i = torch.randint(n, (a.bs,), device=DEV); t = torch.randint(52, (a.bs,), device=DEV)
    x, l, tg = features(data, i, t)
    if a.bhead:
        q, out = net(x, aux=True); ce_c, _, _, ce_s = belief_loss(out, belief_targets(data, i, t))
        loss = (((q - tg) ** 2) * l).sum() / l.sum() + a.bw * (ce_c + 0.25 * ce_s)
    else:
        q = net(x); loss = (((q - tg) ** 2) * l).sum() / l.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    loss_sum += loss.item(); loss_n += 1
    if step % a.every == 0 or step == 200:
        evaluate(step, loss_sum / loss_n, n, t0); loss_sum, loss_n = 0.0, 0
        torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "args": vars(a), "input_dim": D_IN, "step": step}, f"{OUT}/last.pt")
        if step % 20000 == 0: torch.save({"net": {k: v.cpu() for k, v in net.state_dict().items()}, "args": vars(a), "input_dim": D_IN, "step": step}, f"{OUT}/ckpt_step{step}.pt")
