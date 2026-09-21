"""Can a network pick the right contract when it sees BOTH hands?

The best message A could ever send is its whole hand. So this is the ceiling
on any communication protocol. It also reports the TRAIN score, so we can
tell two very different failures apart:

  cannot fit train        -> the contract decision is the problem, not the channel
  fits train, fails test  -> a generalisation problem
  fits both, still << par -> the deal genuinely does not determine the contract
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.exp1 import MLP, Rewards, load_deals, SUBSET_CONTRACTS, FULL_CONTRACTS


class SuitEncoder(nn.Module):
    """One hand -> a vector, with the four suits sharing all the weights.

    A hand is a 4 x 13 grid: suit by rank. The same small net reads each
    suit, so it is trained by four times as much data as a flat 52-bit
    input layer would give it, and what it learns about spades transfers to
    clubs. The four outputs are then concatenated in LADDER order and never
    pooled, because suits are not interchangeable here -- 1C < 1D < 1H < 1S.

    This encodes how a deck is built, not how bridge is bid. No high card
    points, no suit quality, nothing a human would have had to teach it.
    """
    def __init__(self, d=64, conv=False):
        super().__init__()
        self.conv = conv
        if conv:
            # a rank window, so "AKQ together" and "AKQ apart" can differ
            self.f = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
            self.proj = nn.Sequential(nn.Linear(32 * 13, d), nn.ReLU())
        else:
            self.f = nn.Sequential(nn.Linear(13, d), nn.ReLU(),
                                   nn.Linear(d, d), nn.ReLU())
        self.out_dim = 4 * d

    def forward(self, hand):
        B = hand.shape[0]
        suits = hand.view(B * 4, 13)
        if self.conv:
            h = self.f(suits.unsqueeze(1)).flatten(1)
            h = self.proj(h)
        else:
            h = self.f(suits)
        return h.view(B, -1)


class FullInfo(nn.Module):
    """Sees the partnership's two hands, or with --all-hands the whole deal."""
    def __init__(self, n_contracts, hidden=512, layers=3, n_hands=2,
                 encoder="flat", enc_dim=64):
        super().__init__()
        if encoder == "flat":
            self.enc, d = None, 52
        else:
            self.enc = SuitEncoder(enc_dim, conv=(encoder == "conv"))
            d = self.enc.out_dim
        self.net = MLP(d * n_hands, n_contracts, hidden=hidden, layers=layers)

    def forward(self, *hands):
        if self.enc is not None:
            hands = [self.enc(h) for h in hands]
        return self.net(torch.cat(hands, -1))[0]


@torch.no_grad()
def report_split(net, hands, rew, idx, names, label, game_ids):
    R = rew[idx]
    pick = net(*[h[idx] for h in hands]).argmax(-1)
    sc = R.gather(1, pick[:, None]).squeeze(1)
    par = R.max(1).values
    pc = R.argmax(1)
    hi_pick = torch.isin(pick, game_ids).float().mean().item()
    hi_par = torch.isin(pc, game_ids).float().mean().item()
    exact = (pick == pc).float().mean().item()
    print(f"  {label:>10}: score {sc.mean():7.1f}   par {par.mean():7.1f}   "
          f"reaches {sc.mean()/par.mean():5.1%} of par   "
          f"picks par contract {exact:5.1%}   "
          f"bids game/slam {hi_pick:5.1%} (par {hi_par:.1%})")
    return sc.mean().item(), par.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/pgx500k.npz")
    p.add_argument("--n-train", type=int, default=100000, dest="n_train")
    p.add_argument("--n-test", type=int, default=50000, dest="n_test")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--entropy", type=float, default=0.02,
                   help="REQUIRED: without it the softmax collapses "
                        "one-hot and the gradient vanishes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--full-contracts", action="store_true", dest="full_contracts")
    p.add_argument("--objective", default="regress",
                   choices=["regress", "policy"],
                   help="regress: predict every contract's score and take the "
                        "best (fully supervised, cannot collapse). "
                        "policy: maximise expected reward through a softmax, "
                        "which freezes if it ever goes one-hot.")
    p.add_argument("--encoder", default="flat",
                   choices=["flat", "suit", "conv"],
                   help="flat: the raw 52 bits. suit: one shared little net "
                        "reads each of the four suits. conv: same, but with "
                        "a rank window so card sequences can matter.")
    p.add_argument("--enc-dim", type=int, default=64, dest="enc_dim")
    p.add_argument("--all-hands", action="store_true", dest="all_hands",
                   help="feed ALL FOUR hands, so the deal fully determines "
                        "the double dummy answer")
    cfg = p.parse_args()

    contracts = FULL_CONTRACTS if cfg.full_contracts else SUBSET_CONTRACTS
    contracts = [c for c in contracts if c[1] is not None]
    names = [c[0] for c in contracts]

    import numpy as _np
    raw = _np.load(cfg.deals)
    def T(k):
        return torch.tensor(raw[k], dtype=torch.float32, device=cfg.device)
    if cfg.all_hands:
        if "east" not in raw.files:
            raise SystemExit("this deal file has no east/west hands; "
                             "re-export with emergent.pgx")
        hands = [T("north"), T("east"), T("south"), T("west")]
        what = "ALL FOUR hands (the whole deal, 208 bits)"
    else:
        hands = [T("north"), T("south")]
        what = "both partnership hands (104 bits)"
    tricks = raw["tricks"]
    rew = Rewards(tricks, contracts, False, cfg.device)
    n = len(hands[0])
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    perm = np.random.permutation(n)
    test = torch.tensor(perm[:cfg.n_test], device=cfg.device)
    train = torch.tensor(perm[cfg.n_test:cfg.n_test + cfg.n_train],
                         device=cfg.device)
    game_ids = torch.tensor(
        [j for j, nm in enumerate(names)
         if nm in ("3NT", "4H", "4S", "5C", "5D", "6NT")], device=cfg.device)

    print(f"train {len(train)} deals, test {len(test)} deals, "
          f"{len(contracts)} contracts, device {cfg.device}")
    print(f"input: {what}")
    print(f"objective: {cfg.objective}")
    print(f"encoder:   {cfg.encoder}")
    net = FullInfo(len(contracts), cfg.hidden, cfg.layers,
                   n_hands=len(hands), encoder=cfg.encoder,
                   enc_dim=cfg.enc_dim).to(cfg.device)
    nparam = sum(p.numel() for p in net.parameters())
    print(f"network: {cfg.layers} x {cfg.hidden}, {nparam/1e6:.2f}M parameters\n")
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    for step in range(1, cfg.steps + 1):
        b = train[torch.randint(len(train), (cfg.batch,), device=cfg.device)]
        R = rew[b] / 100.0
        out = net(*[h[b] for h in hands])
        if cfg.objective == "regress":
            # we know every contract's score exactly, so just learn it
            loss = F.mse_loss(out, R)
        else:
            pi = F.softmax(out, -1)
            loss = -(pi * R).sum(-1).mean()
            if cfg.entropy:
                loss = loss + cfg.entropy * (pi * torch.log(pi + 1e-9)).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % (cfg.steps // 10) == 0 or step == cfg.steps:
            print(f"step {step:6d}")
            report_split(net, hands, rew, train, names, "TRAIN", game_ids)
            report_split(net, hands, rew, test, names, "test", game_ids)

    print("\n\n================ CEILING ================")
    print("the best message A could send is its whole hand, so this is the")
    print("upper bound for ANY protocol built on these networks.\n")
    tr, tr_par = report_split(net, hands, rew, train, names, "TRAIN", game_ids)
    te, te_par = report_split(net, hands, rew, test, names, "test", game_ids)
    print(f"\n  for reference on the same deal distribution:")
    print(f"    silent partner  (own hand only)     ~ 15")
    print(f"    16 learned messages                 ~ 47")
    print(f"    FULL INFO (both hands)              {te:7.1f}")
    print(f"    par (knows the double dummy answer) {te_par:7.1f}")
    gap = te - 47
    print(f"\n  room a perfect language could still buy: {gap:.0f} points")
    print(f"  room no language can ever buy (par - full info): "
          f"{te_par - te:.0f} points")


if __name__ == "__main__":
    main()
