"""Experiment 1: one-message signaling game.

Sender  : sees own 52-card hand        -> picks a message in {0..M-1}
Receiver: sees own hand + that message -> picks a contract
Both get the same reward = duplicate bridge score of that contract,
scored on the real double-dummy tricks of the deal.

Nothing tells either net what a message means.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.deck import hcp, suit_lengths
from emergent.scoring import (SUBSET_CONTRACTS, FULL_CONTRACTS,
                              build_score_table, contract_strain_dd)


class MLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, layers=2):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.policy = nn.Linear(d, n_out)
        self.value = nn.Linear(d, 1)

    def forward(self, x):
        h = self.body(x)
        return self.policy(h), self.value(h).squeeze(-1)


class Sender(nn.Module):
    """hand -> message logits"""
    def __init__(self, n_messages, **kw):
        super().__init__()
        self.net = MLP(52, n_messages, **kw)

    def forward(self, hand):
        return self.net(hand)


class Receiver(nn.Module):
    """hand + one-hot message -> contract logits"""
    def __init__(self, n_messages, n_contracts, **kw):
        super().__init__()
        self.n_messages = n_messages
        self.net = MLP(52 + n_messages, n_contracts, **kw)

    def forward(self, hand, message):
        m = F.one_hot(message, self.n_messages).float()
        return self.net(torch.cat([hand, m], dim=-1))


class Rewards:
    """reward[deal, contract] from double-dummy tricks + bridge scoring."""
    def __init__(self, tricks, contracts, vulnerable=False, device="cpu"):
        score_tbl = build_score_table(contracts, vulnerable)   # (C, 14)
        strain = contract_strain_dd(contracts)                 # (C,)
        t = tricks[:, strain].astype(np.int64)                 # (N, C)
        r = score_tbl[np.arange(len(contracts))[None, :], t]   # (N, C)
        self.table = torch.tensor(r, dtype=torch.float32, device=device)

    def __getitem__(self, idx):
        return self.table[idx]


def exact_loss(sender, receiver, ha, hb, R, cfg):
    """Exact expected reward. No sampling, no variance.

    E[R] = sum_m pi_A(m|Ha) * sum_c pi_B(c|Hb,m) * R[deal, c]
    Everything is differentiable, so one backward pass trains both nets.
    """
    B = ha.shape[0]
    m_logits, _ = sender(ha)
    if cfg.random_sender:
        pi_m = torch.full_like(m_logits, 1.0 / cfg.n_messages)
    else:
        pi_m = F.softmax(m_logits, -1)                     # (B, M)

    hb_rep = hb.repeat_interleave(cfg.n_messages, 0)       # (B*M, 52)
    msgs = torch.arange(cfg.n_messages, device=ha.device).repeat(B)
    c_logits, _ = receiver(hb_rep, msgs)                   # (B*M, C)
    pi_c = F.softmax(c_logits, -1).view(B, cfg.n_messages, -1)

    q_m = torch.einsum("bmc,bc->bm", pi_c, R)              # value of each message
    expected = (pi_m * q_m).sum(-1)                        # (B,)

    ent_m = -(pi_m * torch.log(pi_m + 1e-9)).sum(-1).mean()
    ent_c = -(pi_c * torch.log(pi_c + 1e-9)).sum(-1).mean()
    ent = ent_m + ent_c
    return -expected.mean() - cfg.entropy * ent, ent


def load_deals(path, device):
    d = np.load(path)
    north = torch.tensor(d["north"], dtype=torch.float32, device=device)
    south = torch.tensor(d["south"], dtype=torch.float32, device=device)
    return north, south, d["tricks"], d["north"], d["south"]


def entropy(logits):
    p = F.softmax(logits, -1)
    return -(p * F.log_softmax(logits, -1)).sum(-1)


def run(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    contracts = FULL_CONTRACTS if cfg.full_contracts else SUBSET_CONTRACTS
    if not cfg.allow_pass:
        # Pass belongs to Experiment 6. With Pass available, passing every deal
        # is close to optimal and nothing has to be communicated.
        contracts = [c for c in contracts if c[1] is not None]
    names = [c[0] for c in contracts]

    north, south, tricks, north_np, south_np = load_deals(cfg.deals, dev)
    n = len(north)
    n_test = min(cfg.n_test, n // 5)
    perm = np.random.permutation(n)
    test_idx = torch.tensor(perm[:n_test], device=dev)
    train_idx = torch.tensor(perm[n_test:], device=dev)
    rew = Rewards(tricks, contracts, cfg.vulnerable, dev)

    # oracle: best contract if you could see both hands
    best = rew.table.max(dim=1).values
    print(f"deals={n} train={len(train_idx)} test={n_test} "
          f"contracts={len(contracts)} messages={cfg.n_messages}")
    print(f"oracle (both hands visible) mean score = "
          f"{best[test_idx].mean().item():.1f}")

    sender = Sender(cfg.n_messages, hidden=cfg.hidden).to(dev)
    receiver = Receiver(cfg.n_messages, len(contracts), hidden=cfg.hidden).to(dev)
    opt = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()),
                           lr=cfg.lr)

    hist = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        batch = train_idx[torch.randint(len(train_idx), (cfg.batch,), device=dev)]
        ha, hb, R = north[batch], south[batch], rew[batch] / 100.0

        if cfg.exact:
            loss, ent = exact_loss(sender, receiver, ha, hb, R, cfg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(sender.parameters()) +
                                     list(receiver.parameters()), 1.0)
            opt.step()
            if step % cfg.eval_every == 0 or step == cfg.steps:
                stats = evaluate(sender, receiver, north, south, rew, best,
                                 test_idx, cfg)
                stats.update(step=step, secs=round(time.time() - t0, 1))
                hist.append(stats)
                print(f"step {step:6d}  score {stats['score']:7.1f}  "
                      f"regret {stats['regret']:6.1f}  "
                      f"msg_entropy {stats['msg_entropy']:.3f} bits  "
                      f"used {stats['messages_used']}/{cfg.n_messages}")
            continue

        m_logits, v_a = sender(ha)
        if cfg.random_sender:                      # baseline 3
            msg = torch.randint(cfg.n_messages, (cfg.batch,), device=dev)
        else:
            msg = torch.distributions.Categorical(logits=m_logits).sample()
        c_logits, v_b = receiver(hb, msg)
        contract = torch.distributions.Categorical(logits=c_logits).sample()

        reward = R.gather(1, contract[:, None]).squeeze(1)

        adv_b = (reward - v_b).detach()
        adv_a = (reward - v_a).detach()
        logp_c = F.log_softmax(c_logits, -1).gather(1, contract[:, None]).squeeze(1)
        logp_m = F.log_softmax(m_logits, -1).gather(1, msg[:, None]).squeeze(1)

        loss_pi = -(logp_c * adv_b).mean()
        loss_v = F.mse_loss(v_b, reward)
        if not cfg.random_sender:
            loss_pi = loss_pi - (logp_m * adv_a).mean()
            loss_v = loss_v + F.mse_loss(v_a, reward)
        ent = entropy(m_logits).mean() + entropy(c_logits).mean()
        loss = loss_pi + cfg.value_weight * loss_v - cfg.entropy * ent

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(sender.parameters()) +
                                 list(receiver.parameters()), 1.0)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            stats = evaluate(sender, receiver, north, south, rew, best,
                             test_idx, cfg)
            stats.update(step=step, secs=round(time.time() - t0, 1))
            hist.append(stats)
            print(f"step {step:6d}  score {stats['score']:7.1f}  "
                  f"regret {stats['regret']:6.1f}  "
                  f"msg_entropy {stats['msg_entropy']:.3f} bits  "
                  f"used {stats['messages_used']}/{cfg.n_messages}")

    final = hist[-1]
    if cfg.return_models:
        return dict(sender=sender, receiver=receiver, rew=rew, best=best,
                    test_idx=test_idx, names=names, north=north, south=south,
                    north_np=north_np, south_np=south_np, stats=final)
    report(sender, receiver, north, south, north_np, rew, best, test_idx,
           names, cfg)
    if cfg.out:
        os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
        with open(cfg.out, "w") as f:
            json.dump({"config": vars(cfg), "history": hist}, f, indent=2)
        print("wrote", cfg.out)
    return final


@torch.no_grad()
def evaluate(sender, receiver, north, south, rew, best, idx, cfg):
    """Greedy (argmax) play on held-out deals."""
    ha, hb = north[idx], south[idx]
    m_logits, _ = sender(ha)
    msg = (torch.randint(cfg.n_messages, (len(idx),), device=ha.device)
           if cfg.random_sender else m_logits.argmax(-1))
    c_logits, _ = receiver(hb, msg)
    contract = c_logits.argmax(-1)
    R = rew[idx]
    score = R.gather(1, contract[:, None]).squeeze(1)
    counts = torch.bincount(msg, minlength=cfg.n_messages).float()
    p = counts / counts.sum()
    ent = -(p[p > 0] * p[p > 0].log2()).sum().item()
    return dict(score=round(score.mean().item(), 2),
                regret=round((best[idx] - score).mean().item(), 2),
                msg_entropy=round(ent, 4),
                messages_used=int((counts > 0).sum().item()),
                msg_freq=[round(x, 4) for x in p.tolist()])


@torch.no_grad()
def report(sender, receiver, north, south, north_np, rew, best, idx, names, cfg):
    ha = north[idx]
    msg = sender(ha)[0].argmax(-1).cpu().numpy()
    hands = north_np[idx.cpu().numpy()]
    print("\n--- what did each message come to mean? (sender hands) ---")
    print(f"{'msg':>4} {'share':>7} {'HCP':>6} {'S':>5} {'H':>5} {'D':>5} {'C':>5}"
          f"  top contracts receiver picks")
    c_logits, _ = receiver(south[idx], torch.tensor(msg, device=ha.device))
    picks = c_logits.argmax(-1).cpu().numpy()
    for m in range(cfg.n_messages):
        sel = msg == m
        if sel.sum() == 0:
            print(f"{m:>4} {0.0:>7.3f}   (unused)")
            continue
        h = hands[sel]
        lens = h.reshape(-1, 4, 13).sum(2).mean(0)
        hcps = (h.reshape(-1, 4, 13) * np.array([4, 3, 2, 1] + [0] * 9)).sum((1, 2))
        uniq, cnt = np.unique(picks[sel], return_counts=True)
        top = ", ".join(f"{names[u]} {c/sel.sum():.0%}"
                        for u, c in sorted(zip(uniq, cnt), key=lambda x: -x[1])[:3])
        print(f"{m:>4} {sel.mean():>7.3f} {hcps.mean():>6.2f} "
              f"{lens[0]:>5.2f} {lens[1]:>5.2f} {lens[2]:>5.2f} {lens[3]:>5.2f}  {top}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/deals50k.npz")
    p.add_argument("--n-messages", type=int, default=5, dest="n_messages")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.02)
    p.add_argument("--value-weight", type=float, default=0.5, dest="value_weight")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-test", type=int, default=5000, dest="n_test")
    p.add_argument("--eval-every", type=int, default=1000, dest="eval_every")
    p.add_argument("--vulnerable", action="store_true")
    p.add_argument("--full-contracts", action="store_true", dest="full_contracts")
    p.add_argument("--allow-pass", action="store_true", dest="allow_pass",
                   help="include Pass (Experiment 6 onward)")
    p.add_argument("--exact", action="store_true",
                   help="train on exact expected reward instead of sampling")
    p.add_argument("--random-sender", action="store_true", dest="random_sender")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sweep-messages", type=int, nargs="+",
                   default=[1, 2, 5, 10, 20], dest="sweep_messages")
    p.add_argument("--n-seeds", type=int, default=3, dest="n_seeds")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--return-models", action="store_true",
                   dest="return_models")
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
