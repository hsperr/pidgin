"""E57: how the belief sharpens over the auction, for one checkpoint.

For every viewer seat and every number of calls heard (0, 1, 2, ... and the full auction)
on a fixed test subset, with systems told and hidden:
    ce, ce_honors (A K Q J), top1      per hidden card
    len_mae                            |expected - true| suit length, per hidden hand and suit
    hcp_mae                            |expected - true| HCP per hidden hand
    sys_acc, fam_acc                   their system / system family, when hidden
Expected lengths/HCP use the Sinkhorn (13 cards per hand) probabilities.
References (same rows): random, and an oracle that knows every hidden hand's exact shape.

    python -u belief/analyze.py runs/r0 [CKPT] [ROWS]
Writes RUN/analysis.json (one entry per checkpoint step).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

FAMILY = {"wb5": "wbridge5", "brl_fsp": "brl", "brl_sl": "brl",
          "D75": "pidgin", "e2b": "pidgin", "E46": "pidgin", "E49B": "pidgin", "E28": "pidgin",
          "ep_21gf": "ep_natural", "ep_sayc": "ep_natural", "ep_gib": "ep_natural",
          "ep_wb5": "ep_natural", "ep_ben": "ep_natural",
          "ep_acol": "ep_acol", "ep_wj": "ep_polish", "ep_prec": "ep_precision"}
HCP = torch.zeros(52)
for s in range(4):
    HCP[s * 13:s * 13 + 4] = torch.tensor([4., 3., 2., 1.])
HONOR = HCP > 0
MAX_K = 24                     # calls heard; longer prefixes are rare


def load_net(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if a.get("arch", "transformer") == "mlp":
        net = T.BeliefMLP(a["d"], a["layers"], summary=a.get("summary", False))
    else:
        net = T.BeliefNet(a["d"], a["layers"], card_tokens=not a.get("no_card_tokens", False),
                          summary=a.get("summary", False))
    net.load_state_dict(ck["net"])
    return net.to(dev).eval(), ck


@torch.no_grad()
def measure(net, test, n_rows, dev):
    fam_names = sorted(set(FAMILY.values()))
    fam_of = torch.tensor([fam_names.index(FAMILY[s]) for s in T.SYSTEMS])
    length = (test["hist"][:n_rows] >= 0).sum(1)
    out = {}
    for tag, drop in (("told", 0.0), ("hidden", 1.0)):
        rows = []
        for k in list(range(MAX_K + 1)) + ["full"]:
            ok = torch.arange(n_rows) if k == "full" else torch.nonzero(length >= k).squeeze(1)
            if len(ok) < 2000:
                continue
            acc = {key: 0.0 for key in ("nll", "n", "nll_h", "n_h", "cor", "len", "n_len", "hcp",
                                        "n_hcp", "sys", "fam", "n_sys", "orc", "orc_cor")}
            g = torch.Generator().manual_seed(1)
            for s in range(0, len(ok), 4096):
                idx = ok[s:s + 4096]
                for v in range(4):
                    b = T.make_batch(test, idx, g, sys_drop=drop, viewer=torch.tensor([v]), prefix=k)
                    bd = {key: x.to(dev) for key, x in b.items()}
                    ol, sl = net(bd)
                    ol, sl = ol.float().cpu(), sl.float().cpu()
                    tgt = b["target"]; hid = tgt > 0; t1 = (tgt - 1).clamp(min=0)
                    lp = ol.log_softmax(-1)
                    nll = -lp.gather(-1, t1[..., None]).squeeze(-1)
                    acc["nll"] += (nll * hid).sum().item(); acc["n"] += hid.sum().item()
                    hh = hid & HONOR[None]
                    acc["nll_h"] += (nll * hh).sum().item(); acc["n_h"] += hh.sum().item()
                    acc["cor"] += ((lp.argmax(-1) == t1) & hid).sum().item()
                    p = T.sinkhorn(ol, b["hand"]).exp() * hid[..., None]          # (n, 52, 3)
                    true = torch.stack([(tgt == j) for j in (1, 2, 3)], -1).float()  # (n, 52, 3)
                    e_len = p.view(-1, 4, 13, 3).sum(2); t_len = true.view(-1, 4, 13, 3).sum(2)
                    acc["len"] += (e_len - t_len).abs().sum().item(); acc["n_len"] += t_len.numel()
                    e_h = (p * HCP[None, :, None]).sum(1); t_h = (true * HCP[None, :, None]).sum(1)
                    acc["hcp"] += (e_h - t_h).abs().sum().item(); acc["n_hcp"] += t_h.numel()
                    their = b["sys_target"][:, 1]; guess = sl[:, 1].argmax(-1)
                    acc["sys"] += (guess == their).sum().item()
                    acc["fam"] += (fam_of[guess] == fam_of[their]).sum().item(); acc["n_sys"] += len(idx)
                    # shape oracle: P(seat) = remaining length / hidden cards in the suit
                    q = t_len / t_len.sum(-1, keepdim=True).clamp(min=1)              # (n, 4, 3)
                    q = q[:, torch.arange(52) // 13]                                   # (n, 52, 3)
                    po = q.gather(-1, t1[..., None]).squeeze(-1).clamp(min=1e-9)
                    acc["orc"] += (-po.log() * hid).sum().item()
                    acc["orc_cor"] += ((q.argmax(-1) == t1) & hid).sum().item()
            rows.append({"k": k, "rows": len(ok),
                         "ce": acc["nll"] / acc["n"], "ce_honors": acc["nll_h"] / acc["n_h"],
                         "top1": acc["cor"] / acc["n"], "len_mae": acc["len"] / acc["n_len"],
                         "hcp_mae": acc["hcp"] / acc["n_hcp"], "sys_acc": acc["sys"] / acc["n_sys"],
                         "fam_acc": acc["fam"] / acc["n_sys"], "oracle_shape_ce": acc["orc"] / acc["n"],
                         "oracle_shape_top1": acc["orc_cor"] / acc["n"]})
            print(tag, k, {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in rows[-1].items()},
                  flush=True)
        out[tag] = rows
    return out


def main():
    run = Path(sys.argv[1])
    ckpt = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else run / "last.pt"
    n_rows = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net, ck = load_net(ckpt, dev)
    test = T.load_test()
    res = measure(net, test, n_rows, dev)
    f = run / "analysis.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[str(ck["step"])] = {"ckpt": ckpt.name, "rows": n_rows, **res}
    f.write_text(json.dumps(data))
    print("wrote", f, "step", ck["step"])


if __name__ == "__main__":
    main()
