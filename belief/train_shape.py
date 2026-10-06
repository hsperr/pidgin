"""Variant: a dedicated shape net.

train.BeliefMLP (card-owner, system, HCP/length summary heads) plus a pattern head: for each
hidden hand (LHO, partner, RHO) a softmax over all 560 suit-length patterns (S,H,D,C summing to
13). Patterns that need more cards in a suit than are hidden from the viewer are masked out.

Loss = card CE + sys_weight * sys CE + w * (pattern CE + summary CE (HCP + per-suit length)).

    python -u belief/train_shape.py --out belief/runs/rC

Load a checkpoint with load_shape(path, dev). Forward returns (owner, sys) like BeliefMLP and
sets net.summary (B, SUM_DIM) and net.pattern (B,3,560) masked logits.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

PATTERNS = torch.tensor([p for p in itertools.product(range(14), repeat=4) if sum(p) == 13])  # (560,4) S H D C
N_PAT = len(PATTERNS)
_LOOKUP = torch.full((14 ** 4,), -1, dtype=torch.long)
_LOOKUP[(PATTERNS * torch.tensor([14 ** 3, 14 ** 2, 14, 1])).sum(1)] = torch.arange(N_PAT)


def pattern_index(lens):
    """lens (...,4) long -> pattern ids (...)."""
    return _LOOKUP.to(lens.device)[(lens * torch.tensor([14 ** 3, 14 ** 2, 14, 1], device=lens.device)).sum(-1)]


def pattern_mask(hand):
    """hand (B,52) -> (B,560) bool: pattern fits the hidden card count of every suit."""
    t = 13 - hand.view(-1, 4, 13).sum(-1).round().long()                       # (B,4)
    return (PATTERNS.to(hand.device)[None] <= t[:, None]).all(-1)


class ShapeMLP(T.BeliefMLP):
    def __init__(self, d=1024, layers=4):
        super().__init__(d, layers, summary=True)
        self.pat_head = nn.Linear(d, 3 * N_PAT)
        self.pattern = None

    def forward(self, b):
        n, m = b["calls"].shape
        grid = torch.zeros(n, T.T_MAX * 4 * (T.N_CALLS + 1), device=b["hand"].device)
        pos = torch.arange(m, device=grid.device)[None]
        flat = (pos * 4 + b["seat"]) * (T.N_CALLS + 1) + b["calls"]
        grid.scatter_(1, flat, b["live"].float())
        x = torch.cat([b["hand"], b["ctx"], F.one_hot(b["dealer"], 4).float(),
                       F.one_hot(self.sys_tokens(b["sys_in"]), self.n_sys + 1).flatten(1).float(), grid], 1)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        self.summary = self.sum_head(h)
        pl = self.pat_head(h).view(n, 3, N_PAT)
        self.pattern = pl.masked_fill(~pattern_mask(b["hand"])[:, None], -1e4)
        return self.owner(h).view(n, 52, 3), self.sys_head(h).view(n, 2, self.n_sys)


@torch.no_grad()
def init_pattern_from_lengths(net):
    """pattern logit(j,p) = sum_s length logit(j,s,P[p,s]): exactly r2's independent-suit pattern model."""
    W, bias = net.sum_head.weight, net.sum_head.bias                          # (3*87, d)
    per = T.HCP_BINS + 56
    for j in range(3):
        r = j * per + T.HCP_BINS + torch.arange(4)[None] * 14 + PATTERNS                     # (560,4)
        net.pat_head.weight[j * N_PAT:(j + 1) * N_PAT] = W[r].sum(1)
        net.pat_head.bias[j * N_PAT:(j + 1) * N_PAT] = bias[r].sum(1)


def load_shape(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = ShapeMLP(a["d"], a["layers"])
    net.load_state_dict(ck["net"])
    return net.to(dev).eval(), ck


def pattern_loss(pl, target):
    """pl (n,3,560) logits, target (n,52) -> per-row mean CE over the 3 hands (n,), correct (n,3)."""
    _, lens = T.summary_targets(target)
    idx = pattern_index(lens)                                                  # (n,3)
    lp = pl.float().log_softmax(-1)
    nll = -lp.gather(-1, idx[..., None]).squeeze(-1)
    return nll.mean(1), (lp.argmax(-1) == idx)


@torch.no_grad()
def eval_pattern(net, test, dev, n=20000, drop=0.0):
    """Full auction, every viewer: pattern NLL / exact top-1 per hidden hand."""
    net.eval()
    g = torch.Generator().manual_seed(0)
    nll = torch.zeros(3); cor = torch.zeros(3); cnt = 0
    for s in range(0, n, 4096):
        idx = torch.arange(s, min(s + 4096, n))
        for v in range(4):
            b = T.make_batch(test, idx, g, sys_drop=drop, viewer=torch.tensor([v]), prefix="full")
            bd = {k: x.to(dev) for k, x in b.items()}
            net(bd)
            _, lens = T.summary_targets(b["target"])
            pi = pattern_index(lens)
            lp = net.pattern.float().cpu().log_softmax(-1)
            nll += -lp.gather(-1, pi[..., None]).squeeze(-1).sum(0)
            cor += (lp.argmax(-1) == pi).float().sum(0); cnt += len(idx)
    net.train()
    return {"pattern_nll": (nll / cnt).tolist(), "pattern_top1": (cor / cnt).tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=60_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--d", type=int, default=1024)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--sys-drop", type=float, default=0.5)
    p.add_argument("--sys-weight", type=float, default=0.1)
    p.add_argument("--w", type=float, default=0.5, help="weight of pattern + summary (HCP, length) CE")
    p.add_argument("--init", default=str(T.HERE / "runs/r2/last.pt"))
    p.add_argument("--holdout", default="ep_wj,E28")
    p.add_argument("--wb5-frac", type=float, default=0.2)
    p.add_argument("--max-shards", type=int, default=0, help="smoke tests: load only this many shards")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-rows", type=int, default=60000)
    p.add_argument("--ckpt-every", type=int, default=10000)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    dev = torch.device(args.device)
    holdout = [T.SYSTEMS.index(s) for s in args.holdout.split(",") if s]

    t0 = time.time()
    rows = T.Rows()
    rows.add_wb5("train")
    n_wb5 = sum(len(x) for x in rows.parts["hist"])
    shards = sorted((T.DATA / "train").glob("shard_*.npz"))
    rows.add_shards(shards[:args.max_shards] if args.max_shards else shards, holdout)
    data = rows.tensors()
    test = T.load_test()
    print(f"train rows {len(data['hist'])}, test rows {len(test['hist'])}, load {time.time() - t0:.0f}s", flush=True)

    net = ShapeMLP(args.d, args.layers)
    if args.init:
        missing, unexpected = net.load_state_dict(torch.load(args.init, map_location="cpu")["net"], strict=False)
        T.resize_sys(net, T.N_SYS)                   # old checkpoint may have fewer systems
        print(f"init from {args.init}; new: {missing}; ignored: {unexpected}", flush=True)
        if "pat_head.weight" in missing:
            init_pattern_from_lengths(net)
            print("pattern head = sum of the r2 length-head logits (r2's implied pattern)", flush=True)
    net.to(dev)
    print(f"params {sum(x.numel() for x in net.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    g = torch.Generator().manual_seed(0)
    log = (out / "log.jsonl").open("a")
    t0, t_last, run = time.time(), time.time(), {"n": 0}
    for step in range(1, args.steps + 1):
        n_gen = len(data["hist"]) - n_wb5
        k_wb5 = round(args.wb5_frac * args.batch)
        idx = torch.cat([torch.randint(n_wb5, (k_wb5,), generator=g),
                         n_wb5 + torch.randint(n_gen, (args.batch - k_wb5,), generator=g)])
        b = T.make_batch(data, idx, g, sys_drop=args.sys_drop)
        bd = {k: x.to(dev) for k, x in b.items()}
        ol, sl = net(bd)
        loss, _, _ = T.card_loss(ol, bd["target"])
        sys_nll = F.cross_entropy(sl.reshape(-1, sl.shape[-1]), bd["sys_target"].reshape(-1), reduction="none")
        drop = bd["sys_dropped"].reshape(-1).float()
        sys_loss = (sys_nll * drop).sum() / drop.sum().clamp(min=1)
        pat_nll, pat_cor = pattern_loss(net.pattern, bd["target"])
        pat_loss = pat_nll.mean()
        sum_loss = T.summary_loss(net.summary, bd["target"])
        total = loss + args.sys_weight * sys_loss + args.w * (pat_loss + sum_loss)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
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
                             ("sys_n", their_hidden.sum()), ("pat_nll", (pat_nll * full).sum()),
                             ("pat_cor", (pat_cor.float().mean(1) * full).sum()), ("n_full", full.sum()),
                             ("sum_nll", sum_loss)):
                run[key] = run.get(key, 0.0) + val.item()
        run["n"] += 1
        if step % 200 == 0:
            ce_full = run["nll_full"] / max(run["hid_full"], 1)
            rec = {"step": step, "ce_full": round(ce_full, 4), "gain": round(math.log(3) - ce_full, 4),
                   "ce_mid": round(run["nll_mid"] / max(run["hid_mid"], 1), 4),
                   "top1": round(run["cor_full"] / max(run["hid_full"], 1), 4),
                   "sys_acc": round(run["sys_ok"] / max(run["sys_n"], 1), 4),
                   "pattern_nll": round(run["pat_nll"] / max(run["n_full"], 1), 4),
                   "pattern_top1": round(run["pat_cor"] / max(run["n_full"], 1), 4),
                   "sum_nll": round(run["sum_nll"] / run["n"], 4),
                   "lr": f'{sched.get_last_lr()[0]:.1e}', "rows": len(data["hist"]),
                   "s_per_step": round((time.time() - t_last) / 200, 3),
                   "min": round((time.time() - t0) / 60, 1)}
            t_last = time.time()
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            run = {"n": 0}
        if step % args.eval_every == 0 or step == args.steps:
            ev = T.evaluate(net, test, dev, max_rows=args.eval_rows)
            ev["known"].update(eval_pattern(net, test, dev, n=min(args.eval_rows, 20000)))
            rec = {"step": step, "eval": ev}
            print(json.dumps({"step": step, **{k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                   for kk, vv in v.items() if not isinstance(vv, dict)}
                                               for k, v in ev.items()}}), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            ck = {"net": net.state_dict(), "args": vars(args), "systems": T.SYSTEMS, "step": step}
            torch.save(ck, out / "last.pt")
            if step % args.ckpt_every == 0:
                torch.save(ck, out / f"ckpt_step{step}.pt")
            t_last = time.time()


if __name__ == "__main__":
    main()
