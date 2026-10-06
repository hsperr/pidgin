"""multi-system auction belief net.

Viewer = one seat. Input: own 13 cards, vulnerability, the auction so far (any prefix), and
a system tag for each side (our side / their side), each hidden with probability
--sys-drop. Output: for every card the viewer cannot see, who holds it (LHO, partner, RHO);
plus heads that guess both sides' systems. Metric: cross-entropy per hidden card in nats
(uniform prior ln 3 = 1.0986).

    python -u belief/train.py --out runs/belief/r1

Data: shards from gen.py (hands rebuilt from dds_results_100M.npy) + WBridge5 (system 0).
New shards are picked up every --reload steps while gen.py is still running.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.bridge.deals import load_dataset  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DDS = os.environ.get("BRIDGE_DATA", str(ROOT / "data" / "dds_results_100M.npy"))
# WBridge5 auctions (system 0); --no-wb5 trains without them.
WB5 = os.environ.get("BRIDGE_WB5", str(ROOT / "data" / "wbridge5" / "{}.npz"))
T_MAX = 48                     # longer auctions (rare X/XX wars) are dropped
N_CALLS = 38                   # training ids: 0..34 contracts, 35 Pass, 36 X, 37 XX
SYSTEMS = json.loads((DATA / "systems.json").read_text())
N_SYS = len(SYSTEMS)
UNKNOWN = N_SYS                # system token "not told"
N_SYS_V1 = 16                  # systems in round 1 (r0..rC checkpoints, the original 120-pair test set)
# Models carry their own n_sys (read from the checkpoint on load); system ids >= a model's n_sys
# (incl. UNKNOWN) are fed to it as its own "not told" token. See resize_sys.
HCP_BINS = 31                  # 0..30 HCP per hidden hand (more is clamped)
SUM_DIM = 3 * (HCP_BINS + 4 * 14)   # per hidden hand: HCP bins + 4 suits x length 0..13
CARD_HCP = torch.zeros(52)
for _s in range(4):
    CARD_HCP[_s * 13:_s * 13 + 4] = torch.tensor([4., 3., 2., 1.])


def summary_targets(target):
    """target (n,52) relative owners -> hcp (n,3) and lengths (n,3,4) for LHO/partner/RHO."""
    one = torch.stack([(target == j) for j in (1, 2, 3)], 1).float()          # (n,3,52)
    hcp = (one * CARD_HCP.to(target.device)).sum(-1).long().clamp(max=HCP_BINS - 1)
    lens = one.view(-1, 3, 4, 13).sum(-1).long()
    return hcp, lens


def split_summary(raw):
    """raw (n, SUM_DIM) -> hcp logits (n,3,HCP_BINS), length logits (n,3,4,14)."""
    r = raw.view(-1, 3, HCP_BINS + 56)
    return r[..., :HCP_BINS], r[..., HCP_BINS:].reshape(-1, 3, 4, 14)


def summary_loss(raw, target):
    hl, ll = split_summary(raw)
    hcp, lens = summary_targets(target)
    return (F.cross_entropy(hl.reshape(-1, HCP_BINS), hcp.reshape(-1))
            + F.cross_entropy(ll.reshape(-1, 14), lens.reshape(-1)))


# ---------------------------------------------------------------- data

def wb5_to_bz(calls: np.ndarray) -> np.ndarray:
    """Era-1 call ids (0 P, 1 X, 2 XX, 3 = 1C ...) -> training ids, -1 kept."""
    out = np.where(calls >= 3, calls - 3, np.choose(np.clip(calls, 0, 2), [35, 36, 37]))
    return np.where(calls < 0, -1, out).astype(np.int8)


class Rows:
    """All auctions in memory: owners (n,52) int8, dealer, vul, ns_sys, ew_sys, hist (n,T_MAX)."""

    def __init__(self):
        self.parts = {k: [] for k in ("owners", "dealer", "vul", "ns_sys", "ew_sys", "hist")}
        self.seen: set[str] = set()
        self.dds = load_dataset(DDS)[0]

    def _add(self, owners, dealer, vul, ns, ew, hist):
        hist = np.asarray(hist, dtype=np.int8)
        length = (hist >= 0).sum(1)
        ok = length <= T_MAX
        h = np.full((len(hist), T_MAX), -1, np.int8)
        w = min(T_MAX, hist.shape[1])
        h[:, :w] = hist[:, :w]
        for k, v in (("owners", owners), ("dealer", dealer), ("vul", vul), ("ns_sys", ns),
                     ("ew_sys", ew), ("hist", h)):
            self.parts[k].append(np.asarray(v)[ok].astype(np.int8))

    def add_wb5(self, split: str):
        z = np.load(WB5.format(split))
        n = len(z["deals"])
        self._add(z["deals"], z["dealers"], np.zeros(n), np.zeros(n), np.zeros(n), wb5_to_bz(z["calls"]))

    def add_shards(self, files, holdout=()) -> int:
        added = 0
        for f in files:
            if str(f) in self.seen or ".tmp" in f.name:
                continue
            self.seen.add(str(f))
            z = np.load(f)
            keep = ~(np.isin(z["ns_sys"], holdout) | np.isin(z["ew_sys"], holdout))
            if not keep.any():
                continue
            idx = z["deal_index"][keep]
            order = np.argsort(idx)                  # mmap reads are faster sorted
            owners = np.empty((len(idx), 52), np.int8)
            owners[order] = self.dds[idx[order]]
            self._add(owners, z["dealer"][keep], z["vul"][keep], z["ns_sys"][keep],
                      z["ew_sys"][keep], z["hist"][keep])
            added += int(keep.sum())
        return added

    def tensors(self):
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in self.parts.items() if v}


def make_batch(d, idx, gen, full_frac=0.5, sys_drop=0.5, viewer=None, prefix=None):
    """Build model inputs/targets for rows ``idx``. viewer/prefix None = random."""
    n = len(idx)
    hist = d["hist"][idx].long()
    length = (hist >= 0).sum(1)
    dealer = d["dealer"][idx].long()
    vul = d["vul"][idx].long()
    owners = d["owners"][idx].long()
    v = torch.randint(4, (n,), generator=gen) if viewer is None else viewer.expand(n).clone()
    if prefix is None:
        t = (torch.rand(n, generator=gen) * (length + 1).float()).long().clamp(max=length)
        t = torch.where(torch.rand(n, generator=gen) < full_frac, length, t)
    else:
        t = length if prefix == "full" else torch.clamp(torch.full_like(length, int(prefix)), max=length)
    k = torch.arange(T_MAX)[None]
    live = k < t[:, None]
    calls = torch.where(live, hist.clamp(min=0), torch.full_like(hist, N_CALLS))  # pad id
    seat = (dealer[:, None] + k - v[:, None]) % 4                                # 0 = me
    rel_owner = (owners - v[:, None]) % 4                                          # 0 = me
    vns, vew = (vul == 1) | (vul == 3), (vul == 2) | (vul == 3)
    ours = torch.where(v % 2 == 0, vns, vew)
    theirs = torch.where(v % 2 == 0, vew, vns)
    sys_ns, sys_ew = d["ns_sys"][idx].long(), d["ew_sys"][idx].long()
    sys_our = torch.where(v % 2 == 0, sys_ns, sys_ew)
    sys_their = torch.where(v % 2 == 0, sys_ew, sys_ns)
    drop_o = torch.rand(n, generator=gen) < sys_drop
    drop_t = torch.rand(n, generator=gen) < sys_drop
    return {
        "hand": (rel_owner == 0).float(),
        "ctx": torch.stack([ours.float(), theirs.float()], 1),
        "dealer": (dealer - v) % 4,
        "calls": calls, "seat": seat, "live": live,
        "sys_in": torch.stack([torch.where(drop_o, UNKNOWN, sys_our),
                               torch.where(drop_t, UNKNOWN, sys_their)], 1),
        "target": rel_owner,                       # 1 LHO, 2 partner, 3 RHO; 0 = own card
        "sys_target": torch.stack([sys_our, sys_their], 1),
        "sys_dropped": torch.stack([drop_o, drop_t], 1),
        "t": t, "t_len": length,
    } | trim(calls, seat, live, t)


def trim(calls, seat, live, t):
    m = int(t.max().clamp(min=1))
    return {"calls": calls[:, :m], "seat": seat[:, :m], "live": live[:, :m]}


# ---------------------------------------------------------------- model

SYS_COL = 52 + 2 + 4           # MLP input: system one-hot block starts after hand, ctx, dealer


def sys_count(sd):
    """Number of systems a state_dict was trained with (None if it has no system head)."""
    w = sd.get("sys_head.weight")
    return None if w is None else w.shape[0] // 2


@torch.no_grad()
def resize_sys(net, k):
    """Change a BeliefMLP/BeliefNet (or subclass) to k systems in place. Old system rows/columns
    are kept; the "not told" slot moves to index k; new systems start as copies of "not told"
    (input side) and of the mean system (output head). Shrinking drops the tail ids."""
    if k is None or k == net.n_sys:
        return net
    k0, m = net.n_sys, min(net.n_sys, k)
    src = list(range(m)) + [k0] * (k - m) + [k0]                 # new id -> old id (k = not told)
    if isinstance(net.inp if hasattr(net, "inp") else None, nn.Linear):
        old = net.inp
        cols = list(range(SYS_COL))
        for side in range(2):
            cols += [SYS_COL + side * (k0 + 1) + j for j in src]
        cols += list(range(SYS_COL + 2 * (k0 + 1), old.in_features))
        new = nn.Linear(len(cols), old.out_features).to(old.weight.device, old.weight.dtype)
        new.weight.copy_(old.weight[:, cols]); new.bias.copy_(old.bias)
        net.inp = new
        if hasattr(net, "n_base"):
            net.n_base += len(cols) - old.in_features
    if hasattr(net, "sys") and isinstance(net.sys, nn.Embedding):
        old = net.sys
        new = nn.Embedding(k + 1, old.embedding_dim).to(old.weight.device, old.weight.dtype)
        new.weight.copy_(old.weight[src]); net.sys = new
    old = net.sys_head
    w, b = old.weight.view(2, k0, -1), old.bias.view(2, k0)
    hw = torch.cat([w[:, :m], w.mean(1, keepdim=True).expand(2, k - m, -1)], 1)
    hb = torch.cat([b[:, :m], b.mean(1, keepdim=True).expand(2, k - m)], 1)
    new = nn.Linear(old.in_features, 2 * k).to(old.weight.device, old.weight.dtype)
    new.weight.copy_(hw.reshape(2 * k, -1)); new.bias.copy_(hb.reshape(-1))
    net.sys_head = new
    net.n_sys = k
    return net


class SysSized:
    """Mixin: load_state_dict first resizes the system slots to the checkpoint's count, so old
    16-system checkpoints load whatever systems.json says now."""

    def load_state_dict(self, state_dict, strict=True, assign=False):
        resize_sys(self, sys_count(state_dict))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def sys_tokens(self, sys_in):
        return sys_in.clamp(max=self.n_sys)

class BeliefNet(SysSized, nn.Module):
    """Tokens: [ctx] [our sys] [their sys] calls... cards(52). Card tokens read out owners."""

    def __init__(self, d=256, layers=6, heads=8, card_tokens=True, summary=False, n_sys=None):
        super().__init__()
        self.n_sys = n_sys = n_sys or N_SYS
        self.sum_head = nn.Linear(d, SUM_DIM) if summary else None
        self.summary = None
        self.card_tokens = card_tokens                  # False: read all 52 owners off the ctx token
        self.ctx = nn.Linear(52 + 2 + 4, d)
        self.sys = nn.Embedding(n_sys + 1, d)
        self.side = nn.Embedding(2, d)
        self.call = nn.Embedding(N_CALLS + 1, d)
        self.seat = nn.Embedding(4, d)
        self.pos = nn.Embedding(T_MAX, d)
        self.card = nn.Embedding(52, d)
        self.mine = nn.Embedding(2, d)
        self.kind = nn.Embedding(4, d)              # ctx, sys, call, card
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.owner = nn.Linear(d, 3) if card_tokens else nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 52 * 3))
        self.sys_head = nn.Linear(d, 2 * n_sys)

    def forward(self, b):
        n = b["hand"].shape[0]
        dev = b["hand"].device
        ctx = self.ctx(torch.cat([b["hand"], b["ctx"], F.one_hot(b["dealer"], 4).float()], 1))
        tok = [ctx[:, None] + self.kind.weight[0],
               self.sys(self.sys_tokens(b["sys_in"])) + self.side.weight[None] + self.kind.weight[1]]
        m = b["calls"].shape[1]                              # batches are trimmed to the longest prefix
        calls = (self.call(b["calls"]) + self.seat(b["seat"]) + self.pos.weight[None, :m]
                 + self.kind.weight[2])
        cards = (self.card.weight[None] + self.mine(b["hand"].long()) + self.kind.weight[3])
        parts = tok + [calls] + ([cards] if self.card_tokens else [])
        x = torch.cat(parts, 1)
        pad = torch.cat([torch.zeros(n, 3, dtype=torch.bool, device=dev), ~b["live"]]
                        + ([torch.zeros(n, 52, dtype=torch.bool, device=dev)] if self.card_tokens else []), 1)
        h = self.norm(self.enc(x, src_key_padding_mask=pad))
        if self.card_tokens:
            owner = self.owner(h[:, -52:])                   # (n, 52, 3): LHO, partner, RHO
        else:
            owner = self.owner(h[:, 0]).view(n, 52, 3)
        sys = self.sys_head(h[:, 0]).view(n, 2, self.n_sys)
        if self.sum_head is not None:
            self.summary = self.sum_head(h[:, 0])
        return owner, sys


class BeliefMLP(SysSized, nn.Module):
    """Same inputs as BeliefNet, flattened: one-hot (position, relative seat, call) grid.
    The transformer stalled near the prior; this reaches E27a's CE in a few hundred steps."""

    def __init__(self, d=1024, layers=4, summary=False, n_sys=None):
        super().__init__()
        self.n_sys = n_sys = n_sys or N_SYS
        self.sum_head = nn.Linear(d, SUM_DIM) if summary else None
        self.summary = None              # set by forward when sum_head exists
        n_in = 52 + 2 + 4 + 2 * (n_sys + 1) + T_MAX * 4 * (N_CALLS + 1)
        self.inp = nn.Linear(n_in, d)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                                    for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.owner = nn.Linear(d, 52 * 3)
        self.sys_head = nn.Linear(d, 2 * n_sys)

    def forward(self, b):
        n, m = b["calls"].shape
        # flat scatter: MPS mis-handles the 4-index advanced assignment
        grid = torch.zeros(n, T_MAX * 4 * (N_CALLS + 1), device=b["hand"].device)
        pos = torch.arange(m, device=grid.device)[None]
        flat = (pos * 4 + b["seat"]) * (N_CALLS + 1) + b["calls"]
        grid.scatter_(1, flat, b["live"].float())
        x = torch.cat([b["hand"], b["ctx"], F.one_hot(b["dealer"], 4).float(),
                       F.one_hot(self.sys_tokens(b["sys_in"]), self.n_sys + 1).flatten(1).float(), grid], 1)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        if self.sum_head is not None:
            self.summary = self.sum_head(h)
        return self.owner(h).view(n, 52, 3), self.sys_head(h).view(n, 2, self.n_sys)


def card_loss(owner_logits, target):
    """Mean CE over hidden cards (target 1..3); returns (loss, n_hidden, n_correct)."""
    hidden = target > 0
    lp = owner_logits.log_softmax(-1)
    tgt = (target - 1).clamp(min=0)
    nll = -lp.gather(-1, tgt[..., None]).squeeze(-1)
    correct = (lp.argmax(-1) == tgt) & hidden
    return (nll * hidden).sum() / hidden.sum(), hidden.sum(), correct.sum()


def sinkhorn(owner_logits, hand, iters=10):
    """Rescale to 13 cards per hidden seat; own cards get prob 0. Returns log-probs (n,52,3)."""
    lp = owner_logits.log_softmax(-1)
    hidden = (hand < 0.5)[..., None]
    for _ in range(iters):
        p = lp.exp() * hidden
        col = p.sum(1, keepdim=True).clamp(min=1e-6)            # expected cards per seat
        lp = lp + (math.log(13.0) - col.log())
        lp = lp.log_softmax(-1)
    return lp


# ---------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(net, test, dev, gen_seed=0, max_rows=None):
    """CE at full auction, every viewer seat. Per (our sys, their sys) known/unknown."""
    net.eval()
    out = {}
    n = len(test["hist"]) if max_rows is None else min(max_rows, len(test["hist"]))
    for tag, drop in (("known", 0.0), ("unknown", 1.0)):
        g = torch.Generator().manual_seed(gen_seed)
        tot = cnt = cor = sk = 0.0
        sys_cor = torch.zeros(2); sys_n = 0
        per = torch.zeros(N_SYS, 2)                  # their system -> (sum nll, count)
        for s in range(0, n, 4096):
            idx = torch.arange(s, min(s + 4096, n))
            for v in range(4):
                b = make_batch(test, idx, g, sys_drop=drop, viewer=torch.tensor([v]), prefix="full")
                bd = {k: x.to(dev) for k, x in b.items()}
                ol, sl = net(bd)
                tgt = bd["target"]
                hidden = tgt > 0
                lp = ol.log_softmax(-1)
                t1 = (tgt - 1).clamp(min=0)[..., None]
                nll = -lp.gather(-1, t1).squeeze(-1) * hidden
                nll_s = -sinkhorn(ol, bd["hand"]).gather(-1, t1).squeeze(-1) * hidden
                tot += nll.sum().item(); sk += nll_s.sum().item(); cnt += hidden.sum().item()
                cor += ((lp.argmax(-1) == t1.squeeze(-1)) & hidden).sum().item()
                sys_cor += (sl.argmax(-1) == bd["sys_target"]).float().sum(0).cpu(); sys_n += len(idx)
                their = b["sys_target"][:, 1]
                per[:, 0].index_add_(0, their, nll.sum(1).cpu())
                per[:, 1].index_add_(0, their, hidden.sum(1).float().cpu())
        out[tag] = {"ce": tot / cnt, "ce_sinkhorn": sk / cnt, "top1": cor / cnt,
                    "sys_acc_our": (sys_cor[0] / sys_n).item(), "sys_acc_their": (sys_cor[1] / sys_n).item(),
                    "ce_by_their_sys": {SYSTEMS[i]: round((per[i, 0] / per[i, 1]).item(), 4)
                                        for i in range(N_SYS) if per[i, 1] > 0}}
    net.train()
    return out


def test_pair_files(ids):
    """Test pair files whose two systems are both in ids."""
    ids = set(ids)
    return [f for f in sorted((DATA / "test").glob("pair_*.npz")) if ".tmp" not in f.name
            and {int(x) for x in f.stem.split("_")[1:3]} <= ids]


def load_test(holdout=(), ids=None):
    """Fixed test set: every pair among ``ids`` (default: the round-1 systems, so numbers stay
    comparable with r0..rC) + WBridge5 test."""
    ids = range(N_SYS_V1) if ids is None else ids
    files = test_pair_files(ids)
    n_gen = len([i for i in ids if i != 0])                 # every system but WBridge5
    if len(files) != n_gen * (n_gen + 1) // 2:
        raise SystemExit(f"test set incomplete: {len(files)} of {n_gen * (n_gen + 1) // 2} pairs")
    r = Rows()
    r.add_shards(files)
    r.add_wb5("test")
    t = r.tensors()
    perm = torch.from_numpy(np.random.default_rng(0).permutation(len(t["hist"])))
    return {k: v[perm] for k, v in t.items()}       # eval subsets cover every pair


# ---------------------------------------------------------------- train

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--no-card-tokens", action="store_true")
    p.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--sys-drop", type=float, default=0.5)
    p.add_argument("--sys-weight", type=float, default=0.1)
    p.add_argument("--summary", action="store_true", help="add HCP / suit-length heads per hidden hand")
    p.add_argument("--sum-weight", type=float, default=0.3)
    p.add_argument("--init", default="", help="checkpoint to start from (missing heads stay new)")
    p.add_argument("--holdout", default="", help="comma list of system names never trained on")
    p.add_argument("--no-wb5", action="store_true")
    p.add_argument("--wb5-frac", type=float, default=0.2, help="share of each batch from WBridge5")
    p.add_argument("--reload", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    dev = torch.device(args.device)
    holdout = [SYSTEMS.index(s) for s in args.holdout.split(",") if s]

    rows = Rows()
    if not args.no_wb5:
        rows.add_wb5("train")
    n_wb5 = sum(len(x) for x in rows.parts["hist"])
    rows.add_shards(sorted((DATA / "train").glob("shard_*.npz")), holdout)
    data = rows.tensors()
    test = load_test()
    print(f"train rows {len(data['hist'])}, test rows {len(test['hist'])}", flush=True)

    if args.arch == "mlp":
        net = BeliefMLP(args.d, args.layers, summary=args.summary).to(dev)
    else:
        net = BeliefNet(args.d, args.layers, card_tokens=not args.no_card_tokens, summary=args.summary).to(dev)
    if args.init:
        missing, unexpected = net.load_state_dict(torch.load(args.init, map_location="cpu")["net"], strict=False)
        resize_sys(net, N_SYS)                       # old checkpoint may have fewer systems
        print(f"init from {args.init}; new: {missing}; ignored: {unexpected}", flush=True)
        net.to(dev)
    print(f"params {sum(x.numel() for x in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 2000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    g = torch.Generator().manual_seed(0)
    log = (out / "log.jsonl").open("a")
    t0, run = time.time(), {"n": 0}
    for step in range(1, args.steps + 1):
        if step % args.reload == 0:
            added = rows.add_shards(sorted((DATA / "train").glob("shard_*.npz")), holdout)
            if added:
                data = rows.tensors()
                print(f"step {step}: +{added} rows, now {len(data['hist'])}", flush=True)
        # WBridge5 rows come first (n_wb5 of them); the rest are generated. Fixed mix per batch.
        n_gen = len(data["hist"]) - n_wb5
        k_wb5 = round(args.wb5_frac * args.batch) if n_wb5 else 0
        idx = torch.cat([torch.randint(n_wb5, (k_wb5,), generator=g) if k_wb5 else torch.zeros(0, dtype=torch.long),
                         n_wb5 + torch.randint(n_gen, (args.batch - k_wb5,), generator=g)])
        b = make_batch(data, idx, g, sys_drop=args.sys_drop)
        bd = {k: x.to(dev) for k, x in b.items()}
        ol, sl = net(bd)
        loss, nh, nc = card_loss(ol, bd["target"])
        sys_nll = F.cross_entropy(sl.reshape(-1, N_SYS), bd["sys_target"].reshape(-1), reduction="none")
        drop = bd["sys_dropped"].reshape(-1).float()
        sys_loss = (sys_nll * drop).sum() / drop.sum().clamp(min=1)
        total = loss + args.sys_weight * sys_loss
        if net.sum_head is not None:
            sum_loss = summary_loss(net.summary, bd["target"])
            total = total + args.sum_weight * sum_loss
            run["sum_nll"] = run.get("sum_nll", 0.0) + sum_loss.item()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():                    # cheap train-batch metrics, split at the lead view
            tgt = bd["target"]; hidden = (tgt > 0).float()
            lp = ol.log_softmax(-1); t1 = (tgt - 1).clamp(min=0)
            nll = (-lp.gather(-1, t1[..., None]).squeeze(-1) * hidden).sum(1)
            cor = ((lp.argmax(-1) == t1).float() * hidden).sum(1)
            full = (bd["t"] == bd["t_len"]).float()
            their_hidden = bd["sys_dropped"][:, 1].float()
            sys_ok = (sl[:, 1].argmax(-1) == bd["sys_target"][:, 1]).float()
            for key, val in (("nll_full", (nll * full).sum()), ("hid_full", (hidden.sum(1) * full).sum()),
                             ("nll_mid", (nll * (1 - full)).sum()), ("hid_mid", (hidden.sum(1) * (1 - full)).sum()),
                             ("cor_full", (cor * full).sum()), ("sys_ok", (sys_ok * their_hidden).sum()),
                             ("sys_n", their_hidden.sum())):
                run[key] = run.get(key, 0.0) + val.item()
        run["n"] += 1
        if step % 200 == 0:
            ce_full = run["nll_full"] / max(run["hid_full"], 1)
            rec = {"step": step, "ce_full": round(ce_full, 4), "gain": round(math.log(3) - ce_full, 4),
                   "ce_mid": round(run["nll_mid"] / max(run["hid_mid"], 1), 4),
                   "top1": round(run["cor_full"] / max(run["hid_full"], 1), 4),
                   "sys_acc": round(run["sys_ok"] / max(run["sys_n"], 1), 4),
                   **({"sum_nll": round(run["sum_nll"] / run["n"], 4)} if "sum_nll" in run else {}),
                   "lr": f'{sched.get_last_lr()[0]:.1e}', "rows": len(data["hist"]),
                   "min": round((time.time() - t0) / 60, 1)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            run = {"n": 0}
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(net, test, dev, max_rows=60000)
            rec = {"step": step, "eval": ev}
            print(json.dumps({"step": step, **{k: {kk: round(vv, 4) for kk, vv in v.items()
                                                   if not isinstance(vv, dict)} for k, v in ev.items()}}),
                  flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            ck = {"net": net.state_dict(), "args": vars(args), "systems": SYSTEMS, "step": step}
            torch.save(ck, out / "last.pt")
            if step % 10000 == 0:                        # kept, so analyze.py can redo old steps
                torch.save(ck, out / f"ckpt_step{step}.pt")


if __name__ == "__main__":
    main()
