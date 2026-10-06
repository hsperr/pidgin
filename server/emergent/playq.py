"""The B2g play Q-net (PidginV2's card play): one value per legal card, trained on solver values.

A port of the trainer's own `features` and `Net` (bridge_public, scratchpad playq/train_play.py,
run B2g, step 540,000), so play-time inputs match training exactly. With search it plays as in
the 4,000-board PidginV2 match: PIMC over E48's belief head for the layouts, 20 layouts,
declarer every trick, defence from trick index 2, ties to B2g's values. Without search (or on a
turn the searcher skips) it plays B2g's best value.

`QPlayBot.choose_card` has the signature of `engine.choose_card`, which hands over to it.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from bridgezero.play.data import N_CALLS

LAST = 6                      # the last six calls, one-hot, after the call bag


def position(c, batch, step):
    """The trainer's per-deal dict for one stateless request: only index `step` is filled."""
    n = len(c)
    bag = torch.zeros(n, 4, N_CALLS, dtype=torch.bool)       # absolute seat x call
    for i in range(c.calls.shape[1]):
        live = (i < c.n_calls).nonzero().squeeze(1)
        bag[live, (c.dealer[live] + i) % 4, c.calls[live, i]] = True
    last = torch.full((n, LAST), N_CALLS, dtype=torch.int8)
    for k in range(LAST):
        j = c.n_calls - 1 - k
        ok = j >= 0
        last[ok, k] = c.calls[ok, j[ok]].to(torch.int8)
    inv = torch.full((n, 52), 127, dtype=torch.int8)
    if step:
        inv.scatter_(1, batch.history[:, :step],
                     torch.arange(step, dtype=torch.int8).expand(n, step).contiguous())
    turn = torch.zeros(n, 52, dtype=torch.int8)
    won = torch.zeros(n, 52, 2, dtype=torch.int8)
    turn[:, step] = batch.to_play().to(torch.int8)
    won[:, step] = batch.tricks_won.to(torch.int8)
    return {"owner": c.owner.to(torch.int8), "inv": inv, "turn": turn, "won": won,
            "decl": c.declarer.to(torch.int8), "trump": c.trump.to(torch.int8),
            "level": c.level.to(torch.int8), "dbl": c.doubled.to(torch.int8),
            "vns": c.vul_ns, "vew": c.vul_ew, "dealer": c.dealer.to(torch.int8),
            "bag": bag, "last": last}


def features(d, i, t):
    """Rows i (deal index), t (card number 0..51) -> (B, 899). The trainer's, minus the target."""
    owner, inv = d["owner"][i].long(), d["inv"][i].long()
    turn = d["turn"][i, t].long()
    decl = d["decl"][i].long()
    dummy = (decl + 2) % 4
    actor = torch.where(turn == dummy, decl, turn)                    # declarer plays dummy
    played = inv < t[:, None]
    hand = (owner == turn[:, None]) & ~played
    other = torch.where((turn == decl) | (turn == dummy), torch.where(turn == decl, dummy, decl), dummy)
    vis = (owner == other[:, None]) & ~played & (t > 0)[:, None]
    rel_owner = (owner - turn[:, None]) % 4
    played_by = F.one_hot(rel_owner, 4).bool() & played[..., None]
    start = (t // 4) * 4
    in_trick = played & (inv >= start[:, None])
    trick_pos = F.one_hot((inv - start[:, None]).clamp(0, 2), 3).bool() & in_trick[..., None]
    role = F.one_hot((turn - decl) % 4, 4)
    side_t = turn % 2
    won = d["won"][i, t].long()
    mine, theirs = won.gather(1, side_t[:, None]), won.gather(1, 1 - side_t[:, None])
    need = (d["level"][i].long() + 6)[:, None]
    vns, vew = d["vns"][i], d["vew"][i]
    vo = torch.where(side_t == 0, vns, vew).float()[:, None]
    vt = torch.where(side_t == 0, vew, vns).float()[:, None]
    rot = (torch.arange(4)[None] + actor[:, None]) % 4                  # relative seat r -> absolute
    bag = d["bag"][i].gather(1, rot[..., None].expand(-1, -1, N_CALLS))
    x = [hand, vis, played_by.flatten(1), trick_pos.flatten(1), F.one_hot(t // 4, 13), F.one_hot(t % 4, 4), role,
         F.one_hot(d["trump"][i].long(), 5), F.one_hot(d["level"][i].long() - 1, 7), F.one_hot(d["dbl"][i].long(), 3),
         vo, vt, mine / 13, theirs / 13, need / 13, bag.flatten(1),
         F.one_hot(d["last"][i].long(), N_CALLS + 1).flatten(1), F.one_hot((d["dealer"][i].long() - actor) % 4, 4)]
    return torch.cat([z.float() for z in x], 1)


class QNet(nn.Module):
    """The trainer's Net with its belief head (unused here, kept so the weights load strict)."""

    def __init__(self, d_in, width, depth):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                                  nn.Dropout(0.0), nn.Linear(width, width)) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52))
        self.bel = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 52 * 4 + 16 * 14 + 4 * 38))

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(h)


class _PriorSwap:
    """E48's net as the searcher sees it (auction encoding, belief head for the layouts),
    with B2g's values in place of E48's policy: the card when nothing is searched, and the
    tie-break among cards the solver cannot separate."""

    def __init__(self, real):
        self.real, self.q_now = real, None

    def auction(self, *a):
        return self.real.auction(*a)

    def __call__(self, feats, legal):
        out = dict(self.real(feats, legal))
        out["log_probs"] = self.q_now
        return out


class QPlayBot:
    family = "playq"

    def __init__(self, path, sampler_net):
        """`sampler_net`: the PlayNet whose belief head draws the PIMC layouts (E48's)."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = QNet(ck["input_dim"], ck["width"], ck["depth"])
        self.net.load_state_dict(ck["net"])
        self.net.eval()
        self.step = ck["step"]
        self.sampler_net = sampler_net
        self._searcher = None

    @torch.no_grad()
    def values(self, contracts, batch):
        """(n, 52) B2g values at the batch's current step, -1e9 where illegal."""
        step = batch.t
        d = position(contracts, batch, step)
        n = batch.n
        x = features(d, torch.arange(n), torch.full((n,), step))
        return self.net(x).masked_fill(~batch.legal(), -1e9)

    def searcher(self):
        from emergent import engine
        if self._searcher is None:
            from bridgezero.play.search import PIMCPlayer
            p = PIMCPlayer.from_net(self.sampler_net, engine.CONFIG.samples, budget_ms=engine.CONFIG.budget_ms,
                                    defence=engine.CONFIG.defence, defence_from_trick=engine.CONFIG.defence_from)
            p.net = _PriorSwap(p.net)
            self._searcher = p
        return self._searcher

    @torch.no_grad()
    def choose_card(self, contracts, batch, search=None, seed=None):
        """(card, top) as `engine.choose_card`: top is B2g's best four legal cards as [(card, value)],
        values in tricks below the best card (0 = best)."""
        from emergent import engine
        q = self.values(contracts, batch)
        card = int(q[0].argmax())
        if engine.search_on(search):
            with engine.SEARCH_LOCK:
                player = self.searcher()
                player.net.q_now = q
                player.start(contracts)
                if seed is None:
                    card = int(player.choose(batch, contracts, batch.legal(), batch.t)[0])
                else:
                    kept = player.sampler, player.budget_ms
                    player.sampler, player.budget_ms = engine.seat_free_sampler(seed), None
                    try:
                        card = int(player.choose(batch, contracts, batch.legal(), batch.t)[0])
                    finally:
                        player.sampler, player.budget_ms = kept
        legal = batch.legal()[0]
        order = [int(c) for c in q[0].argsort(descending=True) if legal[int(c)]]
        best = float(q[0, order[0]])
        return card, [(c, float(q[0, c]) - best) for c in order[:4]]
