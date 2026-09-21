"""Experiment 3: two-way dialogue.

    A sees H_A                     -> sends m1
    B sees H_B and m1              -> sends m2   (an answer)
    A sees H_A, m1 and m2          -> picks the contract

The interesting part is m2. B only speaks after hearing m1, so the same
symbol from B is free to mean different things after different m1. Nothing
assigns any meaning; both nets are trained only by the shared score.

Training uses exact enumeration over every (m1, m2) pair, so the objective

    E[R] = sum_m1 pi_A1(m1|Ha) sum_m2 pi_B(m2|Hb,m1) sum_c pi_A2(c|Ha,m1,m2) R[c]

is differentiable with no sampling noise.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.exp1 import MLP, Rewards, load_deals, SUBSET_CONTRACTS, FULL_CONTRACTS


class OpenerA(nn.Module):
    """H_A -> m1"""
    def __init__(self, n_m1, hidden=256):
        super().__init__()
        self.net = MLP(52, n_m1, hidden=hidden)

    def forward(self, ha):
        return self.net(ha)[0]


class ResponderB(nn.Module):
    """H_B, m1 -> m2"""
    def __init__(self, n_m1, n_m2, hidden=256):
        super().__init__()
        self.n_m1 = n_m1
        self.net = MLP(52 + n_m1, n_m2, hidden=hidden)

    def forward(self, hb, m1):
        x = torch.cat([hb, F.one_hot(m1, self.n_m1).float()], -1)
        return self.net(x)[0]


class ChooserA(nn.Module):
    """H_A, m1, m2 -> contract"""
    def __init__(self, n_m1, n_m2, n_contracts, hidden=256):
        super().__init__()
        self.n_m1, self.n_m2 = n_m1, n_m2
        self.net = MLP(52 + n_m1 + n_m2, n_contracts, hidden=hidden)

    def forward(self, ha, m1, m2):
        x = torch.cat([ha,
                       F.one_hot(m1, self.n_m1).float(),
                       F.one_hot(m2, self.n_m2).float()], -1)
        return self.net(x)[0]


def exact_loss(opener, responder, chooser, ha, hb, R, cfg):
    """Expected reward over the whole dialogue. Fully enumerated."""
    B, M1, M2 = ha.shape[0], cfg.n_m1, cfg.n_m2
    dev = ha.device

    pi1 = F.softmax(opener(ha), -1)                                # (B, M1)

    # B answers every possible m1
    hb_r = hb.repeat_interleave(M1, 0)                             # (B*M1, 52)
    m1_r = torch.arange(M1, device=dev).repeat(B)                  # (B*M1,)
    pi2 = F.softmax(responder(hb_r, m1_r), -1).view(B, M1, M2)     # (B, M1, M2)

    # A picks a contract for every (m1, m2)
    ha_r = ha.repeat_interleave(M1 * M2, 0)                        # (B*M1*M2, 52)
    m1_rr = torch.arange(M1, device=dev).repeat_interleave(M2).repeat(B)
    m2_rr = torch.arange(M2, device=dev).repeat(B * M1)
    pi_c = F.softmax(chooser(ha_r, m1_rr, m2_rr), -1).view(B, M1, M2, -1)

    q2 = torch.einsum("bijc,bc->bij", pi_c, R)      # value of each (m1, m2)
    q1 = (pi2 * q2).sum(-1)                         # value of each m1
    expected = (pi1 * q1).sum(-1)

    ent = (-(pi1 * torch.log(pi1 + 1e-9)).sum(-1).mean()
           - (pi2 * torch.log(pi2 + 1e-9)).sum(-1).mean()
           - (pi_c * torch.log(pi_c + 1e-9)).sum(-1).mean())
    return -expected.mean() - cfg.entropy * ent


@torch.no_grad()
def play(opener, responder, chooser, ha, hb, cfg):
    """Greedy dialogue. Returns (contract, m1, m2)."""
    m1 = opener(ha).argmax(-1)
    m2 = responder(hb, m1).argmax(-1)
    c = chooser(ha, m1, m2).argmax(-1)
    return c, m1, m2


def run(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    contracts = FULL_CONTRACTS if cfg.full_contracts else SUBSET_CONTRACTS
    if not cfg.allow_pass:
        contracts = [c for c in contracts if c[1] is not None]
    names = [c[0] for c in contracts]

    north, south, tricks, north_np, south_np = load_deals(cfg.deals, dev)
    n = len(north)
    n_test = min(cfg.n_test, n // 5)
    perm = np.random.permutation(n)
    test_idx = torch.tensor(perm[:n_test], device=dev)
    train_idx = torch.tensor(perm[n_test:], device=dev)
    rew = Rewards(tricks, contracts, cfg.vulnerable, dev)
    best = rew.table.max(1).values

    opener = OpenerA(cfg.n_m1, cfg.hidden).to(dev)
    responder = ResponderB(cfg.n_m1, cfg.n_m2, cfg.hidden).to(dev)
    chooser = ChooserA(cfg.n_m1, cfg.n_m2, len(contracts), cfg.hidden).to(dev)
    params = list(opener.parameters()) + list(responder.parameters()) \
        + list(chooser.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    print(f"deals={n} train={len(train_idx)} test={n_test} "
          f"contracts={len(contracts)}  m1={cfg.n_m1} m2={cfg.n_m2} "
          f"(={cfg.n_m1*cfg.n_m2} dialogues, {np.log2(cfg.n_m1*cfg.n_m2):.2f} bits)")
    print(f"oracle (both hands visible) mean score = {best[test_idx].mean():.1f}")

    hist, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        b = train_idx[torch.randint(len(train_idx), (cfg.batch,), device=dev)]
        loss = exact_loss(opener, responder, chooser,
                          north[b], south[b], rew[b] / 100.0, cfg)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            s = evaluate(opener, responder, chooser, north, south, rew, best,
                         test_idx, cfg)
            s.update(step=step, secs=round(time.time() - t0, 1))
            hist.append(s)
            print(f"step {step:6d}  score {s['score']:7.1f}  "
                  f"regret {s['regret']:6.1f}  "
                  f"H(m1) {s['h_m1']:.2f}  H(m2|m1) {s['h_m2_given_m1']:.2f}  "
                  f"total {s['h_m1']+s['h_m2_given_m1']:.2f} bits")

    if not getattr(cfg, "quiet", False):
        report(opener, responder, chooser, north, south, north_np, south_np,
               rew, test_idx, names, cfg)
    with torch.no_grad():
        c, _, _ = play(opener, responder, chooser,
                       north[test_idx], south[test_idx], cfg)
        hist[-1]["scores"] = (rew[test_idx].gather(1, c[:, None])
                              .squeeze(1).cpu().numpy())
    if cfg.out:
        os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
        json.dump({"config": vars(cfg), "history": hist}, open(cfg.out, "w"),
                  indent=2)
        print("wrote", cfg.out)
    return hist[-1]


def _entropy_bits(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@torch.no_grad()
def evaluate(opener, responder, chooser, north, south, rew, best, idx, cfg):
    c, m1, m2 = play(opener, responder, chooser, north[idx], south[idx], cfg)
    score = rew[idx].gather(1, c[:, None]).squeeze(1)
    m1n, m2n = m1.cpu().numpy(), m2.cpu().numpy()

    h1 = _entropy_bits(np.bincount(m1n, minlength=cfg.n_m1).astype(float))
    h2 = 0.0
    for a in range(cfg.n_m1):
        sel = m1n == a
        if sel.sum() == 0:
            continue
        h2 += sel.mean() * _entropy_bits(
            np.bincount(m2n[sel], minlength=cfg.n_m2).astype(float))
    return dict(score=round(score.mean().item(), 2),
                regret=round((best[idx] - score).mean().item(), 2),
                h_m1=round(h1, 4), h_m2_given_m1=round(h2, 4),
                m1_used=int(len(np.unique(m1n))),
                m2_used=int(len(np.unique(m2n))))


HCP_W = np.array([4, 3, 2, 1] + [0] * 9)
SUITS = ["S", "H", "D", "C"]


def _describe(hands):
    sh = hands.reshape(-1, 4, 13)
    lens = sh.sum(2)
    return lens.mean(0), (sh * HCP_W).sum((1, 2)).mean()


@torch.no_grad()
def report(opener, responder, chooser, north, south, north_np, south_np,
           rew, idx, names, cfg):
    c, m1, m2 = play(opener, responder, chooser, north[idx], south[idx], cfg)
    i = idx.cpu().numpy()
    m1n, m2n, cn = m1.cpu().numpy(), m2.cpu().numpy(), c.cpu().numpy()
    ha, hb = north_np[i], south_np[i]

    print("\n=== ROUND 1: what does A's opening message mean? (A's hand) ===")
    print(f"{'m1':>4} {'share':>7} {'HCP':>6}   S     H     D     C")
    for a in range(cfg.n_m1):
        sel = m1n == a
        if sel.sum() == 0:
            print(f"{a:>4}   (unused)"); continue
        L, h = _describe(ha[sel])
        print(f"{a:>4} {sel.mean():>7.3f} {h:>6.2f} "
              + " ".join(f"{x:5.2f}" for x in L))

    print("\n=== ROUND 2: what does B's answer mean, given what A said? "
          "(B's hand) ===")
    print("if the same m2 means different things after different m1, the "
          "protocol is context dependent.")
    for a in range(cfg.n_m1):
        sela = m1n == a
        if sela.sum() == 0:
            continue
        print(f"\n after A said m1={a}  ({sela.mean():.0%} of deals)")
        print(f"  {'m2':>4} {'share':>7} {'HCP':>6}   S     H     D     C"
              f"   final contract")
        for b in range(cfg.n_m2):
            sel = sela & (m2n == b)
            if sel.sum() == 0:
                continue
            L, h = _describe(hb[sel])
            u, k = np.unique(cn[sel], return_counts=True)
            o = np.argsort(-k)
            bids = ", ".join(f"{names[u[x]]} {k[x]/sel.sum():.0%}" for x in o[:2])
            print(f"  {b:>4} {sel.sum()/sela.sum():>7.3f} {h:>6.2f} "
                  + " ".join(f"{x:5.2f}" for x in L) + f"   {bids}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/deals50k.npz")
    p.add_argument("--n-m1", type=int, default=4, dest="n_m1")
    p.add_argument("--n-m2", type=int, default=4, dest="n_m2")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-test", type=int, default=10000, dest="n_test")
    p.add_argument("--eval-every", type=int, default=3000, dest="eval_every")
    p.add_argument("--vulnerable", action="store_true")
    p.add_argument("--full-contracts", action="store_true", dest="full_contracts")
    p.add_argument("--allow-pass", action="store_true", dest="allow_pass")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
