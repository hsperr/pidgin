"""Vectorized silent-opponent auctions, endpoint targets, and prefix sampling.

Because opponents always Pass, a cooperative auction is the alternating call
sequence of the two active players ``a0`` (first to act) and ``a1``. Opponent
Passes are implicit and never stored, so they cannot be non-Pass by construction.
Only 36 action slots exist (35 contracts + Pass); X/XX are unrepresentable.

``CooperativeAuction`` in ``environment.py`` is the slow reference; tests check
that both implementations agree on legality, observations, and scores.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch

from ..bridge.calls import PASS
from .data import TorchDeals
from .environment import AUCTION_FEATURES, N_COOP_ACTIONS, OWN_BIDS, PARTNER_BIDS, PARTNER_PASSED
from .targets import TorchScorer

MAX_DECISIONS = 40
CONTRACT_STRAIN = torch.arange(35) % 5  # bidding order C,D,H,S,NT


@dataclass
class CoopBatch:
    deal: torch.Tensor        # (B,) index into TorchDeals
    side: torch.Tensor        # active partnership 0=NS 1=EW
    dealer: torch.Tensor      # absolute seat
    vul: torch.Tensor         # active side vulnerable 0/1
    bidder: torch.Tensor      # (B,35) -1 or active index 0=a0 1=a1
    first_pass: torch.Tensor  # a0 passed before any bid
    last: torch.Tensor        # standing contract or -1
    k: torch.Tensor           # active decisions made so far
    ended: torch.Tensor
    history: torch.Tensor     # (B,MAX_DECISIONS) active calls, -1 padding

    @classmethod
    def start(cls, deal, side, dealer, vul) -> "CoopBatch":
        b = len(deal)
        dev = deal.device
        return cls(deal, side, dealer, vul,
                   torch.full((b, 35), -1, dtype=torch.long, device=dev),
                   torch.zeros(b, dtype=torch.bool, device=dev),
                   torch.full((b,), -1, dtype=torch.long, device=dev),
                   torch.zeros(b, dtype=torch.long, device=dev),
                   torch.zeros(b, dtype=torch.bool, device=dev),
                   torch.full((b, MAX_DECISIONS), -1, dtype=torch.long, device=dev))

    @classmethod
    def cat(cls, batches: list["CoopBatch"]) -> "CoopBatch":
        return cls(**{f.name: torch.cat([getattr(b, f.name) for b in batches]) for f in fields(cls)})

    def __len__(self) -> int:
        return len(self.deal)

    def subset(self, rows: torch.Tensor) -> "CoopBatch":
        return CoopBatch(**{f.name: getattr(self, f.name)[rows] for f in fields(self)})

    @property
    def a0(self) -> torch.Tensor:
        return (self.dealer + (self.side - self.dealer) % 2) % 4

    @property
    def actor_index(self) -> torch.Tensor:
        return self.k % 2

    @property
    def actor_seat(self) -> torch.Tensor:
        return (self.a0 + 2 * self.actor_index) % 4

    def legal(self) -> torch.Tensor:
        """``(B,36)``: contracts above the standing one, and Pass."""
        contracts = torch.arange(35, device=self.last.device)[None] > self.last[:, None]
        mask = torch.cat([contracts, torch.ones_like(contracts[:, :1])], dim=1)
        return mask & ~self.ended[:, None]

    def pass_ends(self) -> torch.Tensor:
        return (self.last >= 0) | ((self.k == 1) & self.first_pass)

    def strain_first(self) -> torch.Tensor:
        """``(B,5)`` active index of the first partnership bidder per strain, -1 if none."""
        by_level = self.bidder.view(-1, 7, 5)
        has = by_level >= 0
        first_level = has.long().argmax(1)
        first = by_level.gather(1, first_level.unsqueeze(1)).squeeze(1)
        return torch.where(has.any(1), first, torch.full_like(first, -1))

    def features(self) -> torch.Tensor:
        """Actor-relative public auction features ``(B,77)``; see environment.py."""
        j = self.actor_index[:, None]
        own = (self.bidder == j).float()
        partner = ((self.bidder >= 0) & (self.bidder != j)).float()
        a0_acts = self.actor_index == 0
        passed = torch.stack([self.first_pass & a0_acts, self.first_pass & ~a0_acts], 1).float()
        dealer_rel = torch.nn.functional.one_hot((self.dealer - self.actor_seat) % 4, 4).float()
        out = torch.cat([own, partner, passed, dealer_rel, self.vul[:, None].float()], 1)
        assert out.shape[1] == AUCTION_FEATURES
        return out

    def apply(self, action: torch.Tensor, rows: torch.Tensor) -> None:
        """In-place: rows (bool mask) make ``action[rows]``."""
        r = rows.nonzero().squeeze(1)
        if len(r) == 0:
            return
        a = action[r]
        if not bool(self.legal()[r, a].all()):
            raise ValueError("illegal cooperative call in batch")
        self.history[r, self.k[r]] = a
        bid = a < PASS
        rb = r[bid]
        self.bidder[rb, a[bid]] = self.actor_index[rb]
        self.last[rb] = a[bid]
        rp = r[~bid]
        ends = self.pass_ends()[rp]
        self.ended[rp[ends]] = True
        opening = rp[~ends]
        if not bool((self.k[opening] == 0).all()):
            raise AssertionError("non-ending pass after the first decision")
        self.first_pass[opening] = True
        self.k[r] += 1

    def call_lists(self) -> list[list[int]]:
        return [[int(x) for x in row if x >= 0] for row in self.history.cpu()]


def endpoint_values(scores: torch.Tensor, batch: CoopBatch) -> torch.Tensor:
    """Map ``(B,2,35)`` actor/partner declarer scores to ``(B,36)`` endpoint values.

    A new contract is declared by whichever partner first named its strain, or by
    the actor if nobody has. Pass keeps the standing contract (zero if none).
    """
    first = batch.strain_first()
    j = batch.actor_index[:, None]
    declarer_by_strain = torch.where(first < 0, torch.zeros_like(first), (first != j).long())
    declarer = declarer_by_strain[:, CONTRACT_STRAIN.to(first.device)]
    contract_values = scores.gather(1, declarer.unsqueeze(1)).squeeze(1)
    standing = contract_values.gather(1, batch.last.clamp(min=0)[:, None]).squeeze(1)
    pass_value = torch.where(batch.last >= 0, standing, torch.zeros_like(standing))
    return torch.cat([contract_values, pass_value[:, None]], dim=1)


@torch.no_grad()
def exact_endpoint(batch: CoopBatch, deals: TorchDeals, scorer: TorchScorer):
    """Exact endpoint points ``(B,36)``, cooperative ceiling ``(B,)``, labels ``(B,2,5)``."""
    rel = deals.rel_tricks(batch.deal, batch.actor_seat)
    exact = scorer.exact(rel, batch.vul)
    return endpoint_values(exact, batch), scorer.ceiling(exact), rel


def expected_endpoint(trick_probs: torch.Tensor, batch: CoopBatch,
                      scorer: TorchScorer) -> torch.Tensor:
    return endpoint_values(scorer.expected(trick_probs, batch.vul), batch)


@torch.no_grad()
def final_scores(batch: CoopBatch, deals: TorchDeals, scorer: TorchScorer):
    """Partnership score of each ended auction and its cooperative ceiling."""
    if not bool(batch.ended.all()):
        raise ValueError("auctions have not ended")
    rel = deals.rel_tricks(batch.deal, batch.a0)       # declarer 0=a0, 1=a1
    exact = scorer.exact(rel, batch.vul)
    strain = batch.last.clamp(min=0) % 5
    declarer = batch.strain_first().gather(1, strain[:, None]).squeeze(1).clamp(min=0)
    score = exact[torch.arange(len(batch)), declarer, batch.last.clamp(min=0)]
    score = torch.where(batch.last >= 0, score, torch.zeros_like(score))
    return score, scorer.ceiling(exact), declarer


def mask_partner(sub: CoopBatch, feats: torch.Tensor, rows=None) -> torch.Tensor:
    """Zero partner bids ``[35:70]`` and partner-passed-first ``[71]``."""
    out = feats.clone()
    out[:, PARTNER_BIDS] = 0
    out[:, PARTNER_PASSED] = 0
    return out


def standing_contract_only(sub: CoopBatch, feats: torch.Tensor, rows=None) -> torch.Tensor:
    """Keep only the standing contract, bidder-agnostic, in ``[0:35]``.

    Own and partner bid bits and both passed-first flags ``[0:72]`` are zeroed;
    dealer and vulnerability are kept.
    """
    bid = (feats[:, OWN_BIDS] + feats[:, PARTNER_BIDS]) > 0
    # Bids strictly ascend, so the highest bid is the standing contract.
    ladder = torch.arange(35, device=feats.device).expand_as(bid)
    standing = torch.where(bid, ladder, torch.full_like(ladder, -1)).max(1).values
    out = feats.clone()
    out[:, OWN_BIDS.start:PARTNER_PASSED + 1] = 0
    out[:, OWN_BIDS] = torch.nn.functional.one_hot(standing + 1, 36)[:, 1:].to(feats.dtype)
    return out


# Observation regimes: what every network input sees of the public auction
# features. Legality, the standing contract, the declarer mapping in
# ``endpoint_values``, termination, and scoring always use the real batch, so
# a masked regime hides feature bits, not all partner information.
OBSERVATIONS = {"intact": None, "partner_feature_hidden": mask_partner,
                "standing_contract_only": standing_contract_only}
_EVERYWHERE = ("at every network input (behavior, learner, target net, validation, "
               "prefix metrics, evaluation)")
OBSERVATION_NOTES = {
    "intact": "all public auction features",
    "partner_feature_hidden": (
        "partner-feature-hidden: partner bid bits [35:70] and partner-passed-first [71] "
        f"are zeroed {_EVERYWHERE}; the real legal mask, standing contract, and "
        "expected-rule declarer mapping remain public"),
    "standing_contract_only": (
        "standing-contract-only: [0:35] holds only the current standing contract with no "
        "bidder identity; own and partner bid history and both passed-first flags are "
        f"zeroed {_EVERYWHERE}; dealer and vulnerability are kept. Not message-free: the "
        "standing contract is partner's last bid when partner made it, partner-passed-first "
        "is derivable from dealer position with no contract, and the expected rule's "
        "declarer mapping uses the real auction"),
}


def observe(batch: CoopBatch, observation: str = "intact") -> torch.Tensor:
    if observation not in OBSERVATIONS:
        raise ValueError(f"unknown observation regime {observation!r}")
    feats = batch.features()
    transform = OBSERVATIONS[observation]
    return feats if transform is None else transform(batch, feats)


def net_outputs(net, deals: TorchDeals, batch: CoopBatch, features: torch.Tensor | None = None,
                observation: str = "intact") -> dict[str, torch.Tensor]:
    hand = deals.hands[batch.deal, batch.actor_seat]
    return net(hand, observe(batch, observation) if features is None else features)


@torch.no_grad()
def decision_values(net, deals: TorchDeals, batch: CoopBatch, scorer: TorchScorer,
                    rule: str, features: torch.Tensor | None = None,
                    observation: str = "intact") -> torch.Tensor:
    out = net_outputs(net, deals, batch, features, observation)
    if rule == "expected":
        return expected_endpoint(torch.softmax(out["trick_logits"], -1), batch, scorer)
    if rule == "q":
        return out["contract_q"]
    if rule == "policy":
        return out["policy_logits"]
    raise ValueError(f"unknown decision rule {rule!r}")


@torch.no_grad()
def sample_prefixes(deals: TorchDeals, n: int, generator: torch.Generator,
                    online=None, target=None, max_depth: int = 6, opening_prob: float = 0.25,
                    window: int = 10, epsilon: float = 0.2, temperature: float = 1.0,
                    device="cpu", observation: str = "intact") -> tuple[CoopBatch, dict]:
    """Silent-opponent prefixes stopped at a chosen active decision depth.

    - depth 0 (the opening decision) with ``opening_prob``, else uniform 1..max_depth;
    - the target actor's own earlier calls come from ``online``, partner calls from
      ``target``; each is mixed with ``epsilon`` uniform exploration;
    - bids are limited to ``window`` steps up the ladder;
    - a Pass that would end the auction before the target depth is excluded,
      which is equivalent to rejecting terminated prefixes;
    - ``online``/``target`` of None means uniform behavior.
    A prefix stuck at 7NT stops early; its only legal call is Pass.
    """
    def rand(high):
        return torch.randint(high, (n,), generator=generator).to(device)

    batch = CoopBatch.start(rand(deals.n), rand(2), rand(4), rand(2))
    depth = torch.where(torch.rand(n, generator=generator).to(device) < opening_prob,
                        torch.zeros(n, dtype=torch.long, device=device),
                        1 + rand(max_depth))
    wanted = depth.clone()
    ladder = torch.arange(35, device=device)[None]
    for _ in range(max_depth):
        rows = ~batch.ended & (batch.k < depth)
        if not rows.any():
            break
        in_window = (ladder > batch.last[:, None]) & (ladder <= batch.last[:, None] + window)
        allowed = torch.cat([in_window, ~batch.pass_ends()[:, None]], 1)
        stuck = rows & ~allowed.any(1)
        depth[stuck] = batch.k[stuck]
        rows &= ~stuck
        logits = torch.zeros(n, N_COOP_ACTIONS, device=device)
        own_turn = batch.actor_index == depth % 2
        for net, pick in ((online, rows & own_turn), (target, rows & ~own_turn)):
            if net is not None and pick.any():
                sub = batch.subset(pick)
                logits[pick] = net_outputs(net, deals, sub, observation=observation)[
                    "policy_logits"] / temperature
        probs = torch.softmax(logits.masked_fill(~allowed, -1e9), -1)
        uniform = allowed.float() / allowed.float().sum(1, keepdim=True).clamp(min=1)
        probs = (1 - epsilon) * probs + epsilon * uniform
        # Sample on CPU with the CPU generator so every device draws the same calls.
        action = torch.multinomial((probs.clamp(min=0) + 1e-12 * allowed).cpu(), 1,
                                   generator=generator).squeeze(1).to(device)
        batch.apply(action, rows)
    if bool(batch.ended.any()) or not bool((batch.k == depth).all()):
        raise AssertionError("prefix sampler failed to stop at the requested decision")
    return batch, {"depth": depth, "wanted_depth": wanted}
