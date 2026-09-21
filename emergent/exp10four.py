"""FOUR: a real four-seat auction. Two partnerships, doubles, zero sum.

    N(0) E(1) S(2) W(3), North deals and calls first, turn t is seat t % 4.
    A call is one of 35 bids (1C..7NT, strictly ascending), Pass, or Double.
    Double is legal only against the opponents' standing bid, and only once
    (no redouble). Three passes after a bid or a double end the auction;
    four passes to start pass the board out for 0; otherwise the horizon
    ends it and the last bid stands.

    NS maximise the NS duplicate score, EW maximise its negative. There is
    nothing else in the objective.

## Why doubles are not optional

Take them away and the game collapses. The worst an undoubled contract can
cost non-vulnerable is 7NT down thirteen, 650, and a realistic sacrifice
costs 150-350. Every game is worth 400-620. So bidding over the opponents
is nearly free: whoever speaks last wins the board, and the auction becomes
an escalation to 7NT on every deal, which teaches neither side anything.
The double is the only call in bridge that makes speaking expensive. 4Sx
down three is -500 against the 420 it stole, and that is what turns "bid
again" into a decision instead of a reflex.

So: 35 bids + Pass + Double, and the doubled score table is exact.

## The history is still free, and still fixed size

exp9's trick generalises. Bids strictly ascend, so the SET of (rung, seat)
bids fixes the order they were made in, and a double of rung k can only sit
between k's bid and the next bid. Two 35 x 4 grids are therefore the whole
auction, losslessly:

    hist_bid[k, s]   rung k was bid by seat s
    hist_dbl[k, s]   rung k was doubled by seat s

and the passes in between are implied: between consecutive calls by seats
si -> sj there are exactly (sj - si - 1) mod 4 of them, before the first
call there are as many as the first bidder's seat index, and after the last
call there are t - (turn of that call) - 1, which the turn index t supplies.
Five scalars finish the state: t, the standing rung, the number of
consecutive passes, whether the contract is doubled, and whether it belongs
to the actor's side.

Each head is fed the seat axis ROTATED to its own seat, so slot 0 is always
me, 1 LHO, 2 partner, 3 RHO. One shared rung embedding table per head, used
for the bid sums, the double sums, the standing contract, and the output
keys, exactly as in exp9. The output has L + 2 entries: bid k, Pass, Double.

Nobody ever sees another hand.

## The trainer is exp9's, section by section

One head per turn t = 0..H-2 (turn H-1 is a forced Pass and needs none).
A call is priced by TRUE-score rollout: fork the state into every legal
call, play the remaining turns greedily with the frozen heads -- each head
maximising ITS OWN side's score -- apply the end rules, read the real NS
score off the double-dummy table, and multiply by the actor's sign. Pass
and Double are ordinary branches; a Pass that ends the auction, a Pass that
does not, and a Double that gets outbid later are all just rollouts.

Training states generalise `sampled_states`. The actor's OWN previous call
(turn t-4) is enumerated at equal weight -- the counterfactual rule, drop
your own reach -- and every other seat is sampled from its floored greedy
policy with the probability mass carried as the state weight. Calls that
would have ended the auction before turn t are removed from that mass, so
an unreachable state gets weight 0 instead of a wrong target.

Loss: MSE against the rollout target over the legal calls, weighted by the
state weight, every head updated every step, Adam, grad clip 1.0.
"""
import argparse, json, os, time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.exp1 import FULL_CONTRACTS
from emergent.exp3q import free_cache, _bits
from emergent.fullinfo import SuitEncoder
from emergent.scoring import contract_strain_dd
from emergent.scoring4 import build_score_table4, contract_levels, contract_strains

NEG_INF = -1e9
SEATS = "NESW"


# ---------------------------------------------------------------- the state

class St:
    """A batch of auction positions, in ABSOLUTE seat coordinates.

    Every tensor has a leading row dimension. `deal` says which deal each
    row belongs to, so hands and double-dummy tricks are looked up rather
    than carried around.
    """
    __slots__ = ("bid", "dbl", "last", "lastseat", "dblflag", "npass",
                 "alive", "deal")

    def __init__(self, bid, dbl, last, lastseat, dblflag, npass, alive, deal):
        self.bid, self.dbl = bid, dbl
        self.last, self.lastseat = last, lastseat
        self.dblflag, self.npass = dblflag, npass
        self.alive, self.deal = alive, deal

    @staticmethod
    def empty(deal, L):
        n, dev = deal.shape[0], deal.device
        z = lambda: torch.zeros(n, dtype=torch.long, device=dev)
        return St(torch.zeros(n, L, 4, device=dev),
                  torch.zeros(n, L, 4, device=dev),
                  torch.full((n,), -1, dtype=torch.long, device=dev),
                  torch.full((n,), -1, dtype=torch.long, device=dev),
                  z(), z(),
                  torch.ones(n, dtype=torch.bool, device=dev), deal)

    def repeat(self, k):
        r = lambda x: x.repeat_interleave(k, 0)
        return St(r(self.bid), r(self.dbl), r(self.last), r(self.lastseat),
                  r(self.dblflag), r(self.npass), r(self.alive), r(self.deal))

    def __getitem__(self, sl):
        return St(self.bid[sl], self.dbl[sl], self.last[sl], self.lastseat[sl],
                  self.dblflag[sl], self.npass[sl], self.alive[sl],
                  self.deal[sl])

    @property
    def n(self):
        return self.last.shape[0]


def legal_mask(st, seat, L):
    """(N, L+2). Bids above the standing one, Pass always, Double when legal."""
    n, dev = st.n, st.last.device
    rungs = torch.arange(L, device=dev)
    bids = rungs[None, :] > st.last[:, None]
    passes = torch.ones(n, 1, dtype=torch.bool, device=dev)
    dbl = ((st.last >= 0) & (((st.lastseat - seat) % 2) == 1)
           & (st.dblflag == 0))[:, None]
    return torch.cat([bids, passes, dbl], 1)


def ends_mask(st, L):
    """(N, L+2). Which calls would END the auction here. Only a Pass can."""
    nxt = st.npass + 1
    p = ((nxt >= 3) & (st.last >= 0)) | (nxt >= 4)
    out = torch.zeros(st.n, L + 2, dtype=torch.bool, device=st.last.device)
    out[:, L] = p
    return out


def apply_call(st, call, seat, L):
    """One turn. Dead rows are untouched whatever `call` says."""
    n, dev = st.n, st.last.device
    rows = torch.arange(n, device=dev)
    act = st.alive
    is_bid = act & (call < L)
    is_pass = act & (call == L)
    is_dbl = act & (call == L + 1)

    bid = st.bid.clone()
    bid[rows, call.clamp(max=L - 1), seat] = torch.where(
        is_bid, torch.ones(n, device=dev), bid[rows, call.clamp(max=L - 1), seat])
    dblh = st.dbl.clone()
    lc = st.last.clamp(min=0)
    dblh[rows, lc, seat] = torch.where(is_dbl, torch.ones(n, device=dev),
                                       dblh[rows, lc, seat])

    last = torch.where(is_bid, call.clamp(max=L - 1), st.last)
    lastseat = torch.where(is_bid, torch.full_like(st.lastseat, seat),
                           st.lastseat)
    dblflag = torch.where(is_bid, torch.zeros_like(st.dblflag),
                          torch.where(is_dbl, torch.ones_like(st.dblflag),
                                      st.dblflag))
    npass = torch.where(is_pass, st.npass + 1,
                        torch.where(act, torch.zeros_like(st.npass), st.npass))
    over = is_pass & (((npass >= 3) & (last >= 0)) | (npass >= 4))
    return St(bid, dblh, last, lastseat, dblflag, npass, st.alive & ~over,
              st.deal)


def declarer_of(st, strain_si, L):
    """(N,) the seat that first named the final strain for the winning side.

    Bids ascend, so bid order is rung order and the first rung of that
    strain held by either member of the side is the one that fixes declarer.
    Rows with no contract return seat 0; their score is forced to 0 anyway.
    """
    n, dev = st.n, st.last.device
    c = st.last.clamp(min=0)
    side = (st.lastseat.clamp(min=0) % 2)
    s1, s2 = side, side + 2
    g = lambda s: st.bid.gather(2, s.view(-1, 1, 1).expand(n, L, 1)).squeeze(2)
    b1, b2 = g(s1), g(s2)
    same = strain_si[None, :] == strain_si[c][:, None]
    cand = same & ((b1 + b2) > 0)
    rungs = torch.arange(L, device=dev)[None, :].expand(n, L)
    k0 = torch.where(cand, rungs, torch.full_like(rungs, L)).min(1).values
    k0 = k0.clamp(max=L - 1)
    mine = b1.gather(1, k0[:, None]).squeeze(1) > 0
    return torch.where(mine, s1, s2)


def ns_score(st, tricks_all, tbl4, strain_dd, strain_si, L, seat_map=None):
    """(N,) the NS duplicate score of the finished auction, in points.

    `seat_map` turns a TABLE seat (0..3 in call order) into the seat the
    double-dummy table is indexed by. It is the identity except in the
    rotated fairness evaluation, where the hands were dealt round by one.
    """
    has = st.last >= 0
    c = st.last.clamp(min=0)
    decl = declarer_of(st, strain_si, L)
    dd_seat = decl if seat_map is None else seat_map[decl]
    tr = tricks_all[st.deal, dd_seat, strain_dd[c]].long()
    raw = tbl4[c, st.dblflag, tr]
    signed = torch.where((decl % 2) == 0, raw, -raw)
    return torch.where(has, signed, torch.zeros_like(signed))


# ---------------------------------------------------------------- the head

class AuctionNet4(nn.Module):
    """One turn, one seat. (my hand, auction so far) -> score of every call.

    Output k < L is "bid rung k", L is Pass, L+1 is Double. The output keys
    are the same rung embeddings the input uses, so a rung's input and
    output weights are one set of numbers.
    """
    def __init__(self, L, hidden=512, d_hand=64, d_rung=48, layers=2):
        super().__init__()
        self.L, self.d = L, d_rung
        self.hand_enc = SuitEncoder(d_hand)
        self.rung = nn.Embedding(L, d_rung)
        self.no_bid = nn.Parameter(torch.zeros(d_rung))
        self.pass_key = nn.Parameter(torch.randn(d_rung) * 0.1)
        self.dbl_key = nn.Parameter(torch.randn(d_rung) * 0.1)
        d_in = self.hand_enc.out_dim + 9 * d_rung + 6
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.proj = nn.Linear(d, d_rung)
        self.bias = nn.Parameter(torch.zeros(L + 2))

    def _sums(self, h, seat):
        """(N, L, 4) absolute -> (N, 4*d) relative to `seat`, as embeddings."""
        n = h.shape[0]
        idx = [(seat + j) % 4 for j in range(4)]
        rel = h[:, :, idx]                                   # (N, L, 4) me..RHO
        flat = rel.permute(0, 2, 1).reshape(n * 4, self.L)
        return (flat @ self.rung.weight).view(n, 4 * self.d)

    def forward(self, hand, st, seat, t):
        n = hand.shape[0]
        hv = self.hand_enc(hand)
        bs = self._sums(st.bid, seat)
        ds = self._sums(st.dbl, seat)
        lastv = torch.where(st.last[:, None] >= 0,
                            self.rung(st.last.clamp(min=0)),
                            self.no_bid[None].expand(n, self.d))
        mine = ((st.lastseat >= 0) & ((st.lastseat % 2) == (seat % 2))).float()
        scal = torch.stack([
            torch.full((n,), t / 10.0, device=hand.device),
            st.last.float() / self.L,
            st.npass.float() / 3.0,
            st.dblflag.float(),
            mine,
            (st.last >= 0).float(),
        ], -1)
        h = self.body(torch.cat([hv, bs, ds, lastv, scal], -1))
        keys = torch.cat([self.rung.weight, self.pass_key[None],
                          self.dbl_key[None]], 0)
        return self.proj(h) @ keys.T + self.bias


def q_at(net, hands, st, t):
    """The turn-t head's value for every call at these positions."""
    seat = t % 4
    return net(hands[seat][st.deal].float(), st, seat, t)


# ------------------------------------------------------------- the rollouts

def sign_of(t):
    return 1.0 if (t % 4) % 2 == 0 else -1.0


@torch.no_grad()
def greedy_finish(nets, hands, st, t0, L, H):
    """Play turns t0..H-2 greedily; each head maximises its OWN side's score.

    Turn H-1 is a forced Pass, which can only end an auction that was going
    to end anyway, and never changes the contract, so it is skipped.
    """
    for u in range(t0, H - 1):
        seat = u % 4
        m = legal_mask(st, seat, L)
        q = q_at(nets[u], hands, st, u)
        pick = q.masked_fill(~m, NEG_INF).argmax(-1)
        pick = torch.where(st.alive, pick, torch.full_like(pick, L))
        st = apply_call(st, pick, seat, L)
    return st


@torch.no_grad()
def rollout_value(nets, hands, st, t, L, H, score_fn, chunk=200000):
    """(N, L+2): the TRUE score after each call, the rest played greedily,
    signed for the seat acting at turn t.

    Selection comes from the frozen heads, evaluation from the double-dummy
    score table. Never from a head's own maximum, which is biased upward and
    compounds through every head above (RESULTS.md section 9).
    """
    A, dev = L + 2, st.last.device
    seat = t % 4
    sign = sign_of(t)
    big = st.repeat(A)
    calls = torch.arange(A, device=dev).repeat(st.n)
    big = apply_call(big, calls, seat, L)
    out = torch.empty(big.n, device=dev)
    for i in range(0, big.n, chunk):
        sl = slice(i, min(i + chunk, big.n))
        end = greedy_finish(nets, hands, big[sl], t + 1, L, H)
        out[sl] = score_fn(end)
    return (sign * out).view(st.n, A)


@torch.no_grad()
def sampled_states(nets, hands, deal, t, L, H, S, eps):
    """(St, weight): practice positions for turn t, one batch of B*S rows.

    The actor's own previous call (turn t-4) is ENUMERATED at equal weight
    when S covers the action space, and otherwise drawn uniformly over the
    legal calls -- either way its own taste never decides how much it learns
    below the calls it avoids. Any earlier call of its own is uniform too.

    Every other seat is sampled from its floored greedy policy, and the row
    carries the probability mass that sample stands for, so the partner's
    and the opponents' reach still weight the state.

    A call that would have ENDED the auction before turn t is struck out of
    that mass: the row would not be a turn-t position at all. What is left
    is the reachability weight, 0 for a state that cannot happen.
    """
    dev = deal.device
    N = deal.shape[0] * S
    d = deal.repeat_interleave(S, 0)
    st = St.empty(d, L)
    w = torch.ones(N, device=dev)
    seat_t = t % 4
    enum_u = t - 4
    for u in range(t):
        seat = u % 4
        ok = legal_mask(st, seat, L) & ~ends_mask(st, L)
        okf = ok.float()
        if seat == seat_t:
            if u == enum_u and S == L + 2:
                call = torch.arange(L + 2, device=dev).repeat(deal.shape[0])
                w = w * ok.gather(1, call[:, None]).squeeze(1).float()
                call = torch.where(ok.gather(1, call[:, None]).squeeze(1),
                                   call, torch.full_like(call, L))
            else:
                call = torch.multinomial(okf + 1e-12, 1).squeeze(1)
        else:
            q = q_at(nets[u], hands, st, u)
            m = legal_mask(st, seat, L)
            one = F.one_hot(q.masked_fill(~m, NEG_INF).argmax(-1), L + 2).float()
            f = m.float()
            f = f / f.sum(-1, keepdim=True).clamp(min=1)
            p = ((1 - eps) * one + eps * f) * okf
            w = w * p.sum(-1)
            p = torch.where(p.sum(-1, keepdim=True) > 0, p, okf)
            call = torch.multinomial(p + 1e-12, 1).squeeze(1)
        st = apply_call(st, call, seat, L)
    return st, w


# --------------------------------------------------------------- evaluation

@torch.no_grad()
def play(nets, hands, deal, L, H):
    """Greedy auction on every row. -> final state and the call at each turn."""
    st = St.empty(deal, L)
    picks = torch.full((deal.shape[0], H), -1, dtype=torch.long,
                       device=deal.device)
    ncalls = torch.zeros(deal.shape[0], dtype=torch.long, device=deal.device)
    live = st.alive
    for t in range(H):
        seat = t % 4
        if t == H - 1:
            live = st.alive          # still going when the horizon arrived
            pick = torch.full_like(st.last, L)               # forced Pass
        else:
            m = legal_mask(st, seat, L)
            q = q_at(nets[t], hands, st, t)
            pick = q.masked_fill(~m, NEG_INF).argmax(-1)
        picks[:, t] = torch.where(st.alive, pick, torch.full_like(pick, -1))
        ncalls = ncalls + st.alive.long()
        st = apply_call(st, pick, seat, L)
    return st, picks, ncalls, live


def coop_par(tricks_all, tbl4, strain_dd, dev):
    """(N,) the best NS score over all contracts and both NS declarers,
    undoubled: what NS could reach if EW never spoke. >= 0, since passing
    out is always available."""
    n = tricks_all.shape[0]
    L = tbl4.shape[0]
    best = torch.zeros(n, device=dev)
    for seat in (0, 2):
        tr = tricks_all[:, seat, :].long()                    # (n, 5)
        s = tbl4[torch.arange(L, device=dev)[None, :], 0,
                 tr[:, strain_dd][:, :]]                      # (n, L)
        best = torch.maximum(best, s.max(1).values)
    return best.clamp(min=0)


@torch.no_grad()
def evaluate(nets, hands, tricks_all, tbl4, strain_dd, strain_si, lvl,
             idx, L, H, par, chunk=20000):
    dev = idx.device

    def one(rot):
        hh = [hands[(j + rot) % 4] for j in range(4)]
        sm = torch.tensor([(j + rot) % 4 for j in range(4)], device=dev)
        sc, dbl, decl, con, nc, by3, open_call = [], [], [], [], [], [], []
        for i in range(0, len(idx), chunk):
            s = idx[i:i + chunk]
            st, picks, ncalls, live = play(nets, hh, s, L, H)
            raw = ns_score(st, tricks_all, tbl4, strain_dd, strain_si, L, sm)
            # the model's seats 0/2 hold original seats rot, rot+2
            sc.append(raw if rot % 2 == 0 else -raw)
            dbl.append(st.dblflag.float() * (st.last >= 0).float())
            decl.append(declarer_of(st, strain_si, L))
            con.append(st.last)
            nc.append(ncalls)
            by3.append(~live)
            open_call.append(picks[:, 0])
        return [torch.cat(x) for x in (sc, dbl, decl, con, nc, by3, open_call)]

    sc, dbl, decl, con, nc, by3, op = one(0)
    sc_rot = one(1)[0]
    has = con >= 0
    lv = lvl[con.clamp(min=0)]
    ns_decl = ((decl % 2) == 0) & has
    opd = op.cpu().numpy()
    return dict(
        ns=round(float(sc.mean()), 2),
        ns_rot=round(float(sc_rot.mean()), 2),
        passout=round(float((~has).float().mean()), 4),
        end3=round(float(by3.float().mean()), 4),
        calls=round(float(nc.float().mean()), 2),
        doubled=round(float(dbl.mean()), 4),
        ns_decl=round(float(ns_decl.float().mean()), 4),
        lvl12=round(float(((lv <= 2) & has).float().mean()), 4),
        lvl34=round(float(((lv >= 3) & (lv <= 4) & has).float().mean()), 4),
        lvl5=round(float(((lv == 5) & has).float().mean()), 4),
        slam=round(float((lv >= 6).float().mean()), 4),
        bits=round(_bits(np.bincount(opd[opd >= 0], minlength=L + 2)
                         .astype(float)), 3),
        par=round(float(par.mean()), 2),
    )


def log_line(s):
    return (f"s:{s['step']:<6} ns:{s['ns']:7.1f} ns_rot:{s['ns_rot']:7.1f} "
            f"sum:{s['ns'] + s['ns_rot']:+7.1f} "
            f"po:{s['passout']:5.1%} x:{s['doubled']:5.1%} "
            f"nsd:{s['ns_decl']:5.1%} calls:{s['calls']:4.1f} "
            f"end3:{s['end3']:5.1%} bits:{s['bits']:.2f} "
            f"| par:{s['par']:.0f} mse:{s['mse']:.3f} {s['secs']:.0f}s")


def lvl_line(s):
    return (f"{'':>8}levels  1-2:{s['lvl12']:5.1%}  3-4:{s['lvl34']:5.1%}  "
            f"5:{s['lvl5']:5.1%}  slam:{s['slam']:5.1%}")


# ------------------------------------------------------------------- driver

def load_deals4(path, device, limit=0):
    d = np.load(path)
    for k in ("north", "east", "south", "west"):
        if k not in d.files:
            raise SystemExit(
                f"{path} has no '{k}'; the four-seat game needs all four "
                f"hands. Re-export with emergent.pgx (pgx1M.npz has them).")
    n = len(d["north"]) if not limit else min(limit, len(d["north"]))
    hands = [torch.tensor(np.asarray(d[k][:n]), dtype=torch.uint8,
                          device=device)
             for k in ("north", "east", "south", "west")]
    tricks = torch.tensor(np.asarray(d["tricks_all"][:n]), dtype=torch.uint8,
                          device=device)
    return hands, tricks


def run(cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    dev = cfg.device
    contracts = [c for c in FULL_CONTRACTS if c[1] is not None]
    names = [c[0] for c in contracts]
    L, H = len(contracts), cfg.horizon

    hands, tricks_all = load_deals4(cfg.deals, dev, cfg.limit)
    n = hands[0].shape[0]
    tbl4 = torch.tensor(build_score_table4(contracts, cfg.vulnerable),
                        device=dev)
    strain_dd = torch.tensor(contract_strain_dd(contracts), device=dev)
    strain_si = torch.tensor(contract_strains(contracts), device=dev)
    lvl = torch.tensor(contract_levels(contracts), device=dev)

    perm = np.random.permutation(n)
    n_test = min(cfg.n_test, n // 5)
    test = torch.tensor(perm[:n_test], device=dev)
    train = torch.tensor(perm[n_test:], device=dev)
    par = coop_par(tricks_all[test], tbl4, strain_dd, dev)

    nets = [AuctionNet4(L, cfg.hidden, cfg.d_hand, cfg.d_rung).to(dev)
            for _ in range(H - 1)]
    opts = [torch.optim.Adam(m.parameters(), lr=cfg.lr) for m in nets]
    nparam = sum(p.numel() for m in nets for p in m.parameters())

    print(f"FOUR-SEAT auction  horizon={H}  seed={cfg.seed}  dev={dev}")
    print(f"deals={n} train={len(train)} test={n_test} rungs={L} "
          f"calls={L + 2} (bids, Pass, Double)")
    print(f"{H - 1} heads, {nparam/1e6:.2f}M parameters, "
          f"seats {' '.join(SEATS[t % 4] for t in range(H))}")
    print(f"{cfg.steps} steps, batch {cfg.batch}, {cfg.states} states/deal")
    print(f"cooperative NS par (EW silent, best NS declarer) = "
          f"{par.mean():.1f}\n")

    score_fn = lambda s: ns_score(s, tricks_all, tbl4, strain_dd, strain_si, L)
    hist_log, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        b = train[torch.randint(len(train), (cfg.batch,), device=dev)]
        total = 0.0
        for t in range(H - 2, -1, -1):
            if t == 0:
                st, w = St.empty(b, L), torch.ones(cfg.batch, device=dev)
            else:
                st, w = sampled_states(nets, hands, b, t, L, H, cfg.states,
                                       cfg.explore)
            with torch.no_grad():
                tgt = rollout_value(nets, hands, st, t, L, H, score_fn,
                                    cfg.roll_chunk) / 100.0
            mask = legal_mask(st, t % 4, L).float() * w[:, None]
            q = q_at(nets[t], hands, st, t)
            err = (q - tgt) ** 2
            total = total + (err * mask).sum() / mask.sum().clamp(min=1e-6)

        for o in opts:
            o.zero_grad()
        total.backward()
        for m in nets:
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts:
            o.step()

        if step % 200 == 0:
            free_cache(dev)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            free_cache(dev)
            for m in nets:
                m.eval()
            s = evaluate(nets, hands, tricks_all, tbl4, strain_dd, strain_si,
                         lvl, test, L, H, par)
            for m in nets:
                m.train()
            s.update(step=step, secs=round(time.time() - t0, 1),
                     mse=round(float(total.item()), 4))
            hist_log.append(s)
            print(log_line(s), flush=True)
            print(lvl_line(s), flush=True)
            if cfg.save:
                os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
                torch.save({"step": step, "hist": hist_log,
                            "config": vars(cfg), "names": names,
                            "nets": [m.state_dict() for m in nets]}, cfg.save)
            if cfg.out:
                os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
                json.dump({"config": vars(cfg), "history": hist_log},
                          open(cfg.out, "w"), indent=2)
            free_cache(dev)
    if cfg.out:
        print("wrote", cfg.out)
    return hist_log[-1] if hist_log else {}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/pgx1M.npz")
    p.add_argument("--horizon", type=int, default=8,
                   help="turns in the auction; the last one is a forced Pass")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500, dest="eval_every")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--states", type=int, default=16,
                   help="practice states per deal per turn. At L+2 (37) the "
                        "actor's own previous call is fully enumerated; "
                        "below that it is drawn uniformly over its legal "
                        "calls, which is the same equal weighting")
    p.add_argument("--explore", type=float, default=0.2,
                   help="epsilon on the other seats' greedy policies while "
                        "sampling practice states")
    p.add_argument("--roll-chunk", type=int, default=200000, dest="roll_chunk",
                   help="rows of the forked rollout to run at once")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--d-hand", type=int, default=64, dest="d_hand")
    p.add_argument("--d-rung", type=int, default=48, dest="d_rung")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-test", type=int, default=50000, dest="n_test")
    p.add_argument("--limit", type=int, default=0,
                   help="use only the first N deals of the file")
    p.add_argument("--vulnerable", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="")
    p.add_argument("--save", default="")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
