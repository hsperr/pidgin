"""Two-way dialogue with a decider that predicts scores instead of a policy.

    A sees H_A             -> m1
    B sees H_B, m1         -> m2
    A sees H_A, m1, m2     -> picks the contract

Three losses, kept apart so none of them can freeze:

  chooser   q(H_A, m1, m2) -> a predicted score for every contract.
            MSE against the true scores, weighted by how often that
            conversation actually happens. Supervised, cannot collapse.

  B         maximises  sum_m2 pi2(m2 | H_B, m1) * value(m1, m2)
  A         maximises  sum_m1 pi1(m1 | H_A) * sum_m2 pi2(...) * value(m1, m2)

            where value(m1, m2) is what this deal scores if A bids whatever
            the chooser rates best after that exchange.

Both message gradients flow through a softmax over a handful of symbols, so
they stay alive. Nothing says what a symbol means.
"""
import argparse, json, os, time

# MPS counts system-wide memory against its cap, so reading a multi-GB deal
# file can make it refuse allocations while plenty of memory is actually
# free. Our own footprint here is ~1 GiB.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.exp1 import Rewards, load_deals, SUBSET_CONTRACTS, FULL_CONTRACTS
from emergent.exp3 import OpenerA, ResponderB, ChooserA

HCP_W = np.array([4, 3, 2, 1] + [0] * 9)


def all_q(chooser, ha, M1, M2):
    """q for every (m1, m2): (B, M1, M2, C).

    The messages go in as one-hot INPUTS, so the trunk processes hand and
    conversation together at every layer. Emitting one output head per
    conversation instead is about 2x faster per step but strictly less
    expressive -- the message can then only shift the final hidden state,
    not change how the hand is read. Measured at equal wall clock, that
    version scores 58.7 with A using 0.16 bits, against 65.3 and 1.97 bits
    here: A stops communicating entirely. The extra passes are worth it.
    """
    B, dev = ha.shape[0], ha.device
    ha_r = ha.repeat_interleave(M1 * M2, 0)
    m1 = torch.arange(M1, device=dev).repeat_interleave(M2).repeat(B)
    m2 = torch.arange(M2, device=dev).repeat(B * M1)
    return chooser(ha_r, m1, m2).view(B, M1, M2, -1)


def all_pi2(responder, hb, M1):
    """B's reply distribution for every possible opening: (B, M1, M2)."""
    B, dev = hb.shape[0], hb.device
    hb_r = hb.repeat_interleave(M1, 0)
    m1 = torch.arange(M1, device=dev).repeat(B)
    return F.softmax(responder(hb_r, m1), -1).view(B, M1, -1)


def run(cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = cfg.device
    contracts = FULL_CONTRACTS if cfg.full_contracts else SUBSET_CONTRACTS
    contracts = [c for c in contracts if c[1] is not None]
    names = [c[0] for c in contracts]
    M1, M2, C = cfg.n_m1, cfg.n_m2, len(contracts)

    north, south, tricks, north_np, south_np = load_deals(cfg.deals, dev)
    n = len(north)
    perm = np.random.permutation(n)
    n_test = min(cfg.n_test, n // 5)
    test = torch.tensor(perm[:n_test], device=dev)
    train = torch.tensor(perm[n_test:], device=dev)
    rew = Rewards(tricks, contracts, cfg.vulnerable, dev)
    par = rew.table.max(1).values
    game = torch.tensor([j for j, nm in enumerate(names)
                         if nm in ("3NT", "4H", "4S", "5C", "5D", "6NT")],
                        device=dev)

    opener = OpenerA(M1, cfg.hidden).to(dev)
    responder = ResponderB(M1, M2, cfg.hidden).to(dev)
    chooser = ChooserA(M1, M2, C, cfg.hidden).to(dev)
    opt_a = torch.optim.Adam(opener.parameters(), lr=cfg.lr)
    opt_b = torch.optim.Adam(responder.parameters(), lr=cfg.lr)
    opt_c = torch.optim.Adam(chooser.parameters(), lr=cfg.lr)

    print(f"deals={n} train={len(train)} test={n_test} contracts={C}")
    print(f"m1={M1} m2={M2} -> {M1*M2} conversations "
          f"({np.log2(M1*M2):.2f} bits) device={dev}")
    print(f"par (knows the answer) = {par[test].mean():.1f}\n")

    nets = dict(opener=opener, responder=responder, chooser=chooser)
    opts = dict(opener=opt_a, responder=opt_b, chooser=opt_c)
    hist, start = [], 1
    if cfg.resume and os.path.exists(cfg.resume):
        ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
        for k, m in nets.items():
            m.load_state_dict(ck["nets"][k])
        for k, o in opts.items():
            o.load_state_dict(ck["opts"][k])
        hist, start = ck["hist"], ck["step"] + 1
        print(f"resumed from {cfg.resume} at step {ck['step']}\n")

    def save(step):
        if not cfg.save:
            return
        os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
        torch.save({"step": step, "hist": hist, "config": vars(cfg),
                    "nets": {k: m.state_dict() for k, m in nets.items()},
                    "opts": {k: o.state_dict() for k, o in opts.items()}},
                   cfg.save)

    t0 = time.time()
    for step in range(start, cfg.steps + 1):
        b = train[torch.randint(len(train), (cfg.batch,), device=dev)]
        R = rew[b] / 100.0
        ha, hb = north[b], south[b]

        pi1 = F.softmax(opener(ha), -1)                       # (B,M1)
        pi2 = all_pi2(responder, hb, M1)                      # (B,M1,M2)
        q = all_q(chooser, ha, M1, M2)                        # (B,M1,M2,C)
        pi1_d, pi2_d, q_d = pi1.detach(), pi2.detach(), q.detach()

        # --- chooser: learn each contract's score.
        #
        # Weight by pi2 only. B's reply is what carries information about B's
        # hand, so that weight is what makes m2 mean anything. Do NOT weight
        # by pi1: m1 is A's own choice and tells A nothing, and weighting by
        # it starves the heads for openings A currently avoids. Those heads
        # then stay random, look terrible, and A never tries them again --
        # a collapse that pins A to a single opening.
        w = pi2_d.unsqueeze(-1)                               # (B,M1,M2,1)
        loss_c = (w * (q - R[:, None, None, :]) ** 2).sum(-1).mean()

        # --- what each conversation is worth on this deal
        with torch.no_grad():
            greedy = q_d.argmax(-1)                           # (B,M1,M2)
            value = torch.gather(
                R[:, None, None, :].expand(-1, M1, M2, -1), 3,
                greedy.unsqueeze(-1)).squeeze(-1)             # (B,M1,M2)

        # --- B: reply so that what A then bids scores well.
        #     Mix a floor into A's opening distribution, so B stays competent
        #     at answering openings A is not currently fond of. Without it A
        #     can never safely try one.
        eps = cfg.explore
        w1 = (1 - eps) * pi1_d + eps / M1                     # (B,M1)
        ent2 = -(pi2 * torch.log(pi2 + 1e-9)).sum(-1)         # (B,M1)
        loss_b = -(w1 * ((pi2 * value).sum(-1)
                         + cfg.entropy * ent2)).sum(-1).mean()

        # --- A: open so that the whole exchange scores well
        q1 = (pi2_d * value).sum(-1)                          # (B,M1)
        ent1 = -(pi1 * torch.log(pi1 + 1e-9)).sum(-1).mean()
        loss_a = -(pi1 * q1).sum(-1).mean() - cfg.entropy * ent1

        opt_a.zero_grad(); opt_b.zero_grad(); opt_c.zero_grad()
        (loss_a + loss_b + loss_c).backward()
        for m in (opener, responder, chooser):
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt_a.step(); opt_b.step(); opt_c.step()

        if step % 2000 == 0:
            free_cache(dev)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            free_cache(dev)
            s = evaluate(opener, responder, chooser, north, south, rew, par,
                         test, game, cfg)
            free_cache(dev)
            s.update(step=step, secs=round(time.time() - t0, 1))
            hist.append(s)
            print(f"step {step:6d}  score {s['score']:7.1f}  "
                  f"{s['pct_par']:5.1%} of par  "
                  f"H(m1) {s['h_m1']:.2f}  H(m2|m1) {s['h_m2_given_m1']:.2f}  "
                  f"game {s['pct_game']:4.1%} (par {s['par_game']:.1%})")
            save(step)

    free_cache(dev)
    report(opener, responder, chooser, north, south, north_np, south_np,
           rew, test, names, cfg)
    if cfg.out:
        os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
        json.dump({"config": vars(cfg), "history": hist},
                  open(cfg.out, "w"), indent=2)
        print("wrote", cfg.out)
    return hist[-1]


@torch.no_grad()
def play(opener, responder, chooser, ha, hb, chunk=50000):
    """Greedy dialogue, in slices so a big test set fits in GPU memory."""
    m1s, m2s, cs = [], [], []
    for i in range(0, len(ha), chunk):
        a, b = ha[i:i + chunk], hb[i:i + chunk]
        n = len(a)
        r = torch.arange(n, device=a.device)
        m1 = opener(a).argmax(-1)
        m2 = responder(b, m1).argmax(-1)
        cs.append(chooser(a, m1, m2).argmax(-1))
        m1s.append(m1); m2s.append(m2)
    return torch.cat(m1s), torch.cat(m2s), torch.cat(cs)


def free_cache(device):
    if str(device).startswith("mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _bits(counts):
    p = counts / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


@torch.no_grad()
def evaluate(opener, responder, chooser, north, south, rew, par, idx, game, cfg):
    m1, m2, c = play(opener, responder, chooser, north[idx], south[idx])
    R = rew[idx]
    sc = R.gather(1, c[:, None]).squeeze(1)
    a, b = m1.cpu().numpy(), m2.cpu().numpy()
    h1 = _bits(np.bincount(a, minlength=cfg.n_m1).astype(float))
    h2 = sum((a == k).mean() * _bits(np.bincount(b[a == k],
                                                 minlength=cfg.n_m2).astype(float))
             for k in range(cfg.n_m1) if (a == k).sum())
    return dict(score=round(sc.mean().item(), 2),
                par=round(par[idx].mean().item(), 2),
                pct_par=round((sc.mean() / par[idx].mean()).item(), 4),
                h_m1=round(h1, 4), h_m2_given_m1=round(h2, 4),
                pct_game=round(torch.isin(c, game).float().mean().item(), 4),
                par_game=round(torch.isin(R.argmax(1), game).float()
                               .mean().item(), 4))


@torch.no_grad()
def report(opener, responder, chooser, north, south, north_np, south_np,
           rew, idx, names, cfg):
    m1, m2, c = play(opener, responder, chooser, north[idx], south[idx])
    i = idx.cpu().numpy()
    a, b, cn = m1.cpu().numpy(), m2.cpu().numpy(), c.cpu().numpy()
    ha, hb = north_np[i], south_np[i]

    def facts(h):
        sh = h.reshape(-1, 4, 13)
        return sh.sum(2).mean(0), (sh * HCP_W).sum((1, 2))

    print("\n=== ROUND 1: A's opening (A's hand) ===")
    print(f"{'m1':>4} {'share':>7} {'HCP':>6} {'sd':>5}   S     H     D     C")
    for k in range(cfg.n_m1):
        sel = a == k
        if sel.sum() == 0:
            print(f"{k:>4}   (unused)"); continue
        L, hp = facts(ha[sel])
        print(f"{k:>4} {sel.mean():>7.3f} {hp.mean():>6.2f} {hp.std():>5.2f}  "
              + " ".join(f"{x:5.2f}" for x in L))

    print("\n=== ROUND 2: B's reply, given what A said (B's hand) ===")
    for k in range(cfg.n_m1):
        sa = a == k
        if sa.sum() == 0:
            continue
        print(f"\n after A said m1={k}  ({sa.mean():.0%} of deals)")
        print(f"  {'m2':>4} {'share':>7} {'HCP':>6} {'sd':>5}   S     H     D"
              f"     C   final contract")
        for j in range(cfg.n_m2):
            sel = sa & (b == j)
            if sel.sum() == 0:
                continue
            L, hp = facts(hb[sel])
            u, cnt = np.unique(cn[sel], return_counts=True)
            o = np.argsort(-cnt)
            bids = ", ".join(f"{names[u[x]]} {cnt[x]/sel.sum():.0%}"
                             for x in o[:2])
            print(f"  {j:>4} {sel.sum()/sa.sum():>7.3f} {hp.mean():>6.2f} "
                  f"{hp.std():>5.2f}  " + " ".join(f"{x:5.2f}" for x in L)
                  + f"   {bids}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/pgx1M.npz")
    p.add_argument("--n-m1", type=int, default=5, dest="n_m1")
    p.add_argument("--n-m2", type=int, default=5, dest="n_m2")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--explore", type=float, default=0.3,
                   help="floor mixed into A's opening distribution when "
                        "training B, so unused openings stay answerable")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-test", type=int, default=100000, dest="n_test")
    p.add_argument("--eval-every", type=int, default=3000, dest="eval_every")
    p.add_argument("--vulnerable", action="store_true")
    p.add_argument("--full-contracts", action="store_true", dest="full_contracts")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="", help="write the metric history here")
    p.add_argument("--save", default="",
                   help="write weights + optimiser state here at every eval")
    p.add_argument("--resume", default="",
                   help="continue from a checkpoint written by --save")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
