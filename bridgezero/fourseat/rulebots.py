"""Batched rule bidders: other systems for the league and for matches (``rule:NAME``).

Each bot reads its hand and the auction so far and makes one natural call. Three
styles, so a net that trains against them meets auctions it does not write itself:

- ``sayc``: a plain natural system. 1NT 15-17, five-card majors, 2C strong, weak twos,
  simple raises and new suits, sound overcalls, competes to the level of its fit.
- ``weakclub``: weak-club openings. 1C shows 0-10 HCP, 1D 11-14 without a major,
  1H/1S 11-14 with five, 1NT 15-17 balanced, 2C 18+. The rest as ``sayc``.
- ``happy``: bids a lot. Opens 9+ with a five-card suit, overcalls on 5-card suits with
  5+ HCP, competes one level past its fit.

They never double or redouble. Calls: 0..34 bids (level*5 + strain, strains C D H S NT),
35 Pass. Cards: suit*13 + rank, suits S H D C, rank 0 = ace.
"""

from __future__ import annotations

import torch

from ..bridge.calls import PASS

STYLES = ("sayc", "weakclub", "happy")
SUIT_TO_STRAIN = torch.tensor([3, 2, 1, 0])      # card suit S H D C -> bid strain
HCP = torch.tensor([4, 3, 2, 1] + [0] * 9, dtype=torch.float32)


def bid(level: int | torch.Tensor, strain: int | torch.Tensor):
    return (level - 1) * 5 + strain


class RuleBot:
    """``act(hands, history, t, dealer) -> calls``; see the module docstring."""

    is_rule = True

    def __init__(self, style: str):
        if style not in STYLES:
            raise ValueError(f"unknown rule bot {style!r}: want one of {STYLES}")
        self.style = style
        self.name = f"rule:{style}"

    @torch.no_grad()
    def act(self, hands: torch.Tensor, history: torch.Tensor, t: torch.Tensor,
            dealer: torch.Tensor) -> torch.Tensor:
        B, dev = len(hands), hands.device
        rows = torch.arange(B, device=dev)
        hand = hands.float().view(B, 4, 13)
        hcp = (hand * HCP.to(dev)).sum((1, 2))
        length = hand.sum(2).long()                                   # (B,4) S H D C
        balanced = (length.min(1).values >= 2) & ((length == 2).sum(1) <= 1)
        seat = (dealer + t) % 4

        # the auction so far, relative to this seat: 0 self, 1 LHO, 2 partner, 3 RHO
        H = history.shape[1]
        pos = torch.arange(H, device=dev)[None]
        made = (pos < t[:, None]) & (history >= 0)
        rel = ((dealer[:, None] + pos) - seat[:, None]) % 4
        is_bid = made & (history < PASS)
        last = torch.where(is_bid, history, -1).max(1).values           # standing bid, -1 none
        ours = is_bid & (rel % 2 == 0)
        theirs = is_bid & (rel % 2 == 1)
        mine = is_bid & (rel == 0)
        partner = is_bid & (rel == 2)
        first_bid = torch.where(is_bid, pos, H).min(1).values
        opener_rel = torch.where(first_bid < H, rel.gather(1, first_bid.clamp(max=H - 1)[:, None]).squeeze(1), -1)
        opening_call = torch.where(first_bid < H, history.gather(1, first_bid.clamp(max=H - 1)[:, None]).squeeze(1), -1)
        partner_last = torch.where(partner, history, -1).max(1).values
        we_bid, they_bid, i_bid = ours.any(1), theirs.any(1), mine.any(1)
        we_hold = (last >= 0) & (torch.where(ours, history, -1).max(1).values == last)

        call = torch.full((B,), PASS, dtype=torch.long, device=dev)
        undecided = torch.ones(B, dtype=torch.bool, device=dev)

        def take(cond, value):
            nonlocal call, undecided
            value = torch.as_tensor(value, device=dev).expand(B) if not torch.is_tensor(value) else value
            ok = cond & undecided & (value > last) & (value <= 34)
            call = torch.where(ok, value, call)
            undecided = undecided & ~ok

        def cheapest(strain):
            """Lowest legal bid in ``strain`` (B,) over the standing bid."""
            level = torch.where(last < 0, 1, last // 5 + 1 + (strain <= last % 5).long())
            return bid(level, strain), level

        longest = length.argmax(1)                  # first suit wins ties: S before H ...
        long_len = length.max(1).values
        long_strain = SUIT_TO_STRAIN.to(dev)[longest]
        major = torch.where(length[:, 0] >= length[:, 1], 0, 1)      # S if at least as long
        major_len = length[rows, major]
        major_strain = SUIT_TO_STRAIN.to(dev)[major]
        minor_strain = torch.where(length[:, 2] > length[:, 3], 1, 0)   # D if longer, else C

        opening = last < 0
        happy = self.style == "happy"

        # ------------------------------------------------------------ openings
        if self.style == "weakclub":
            take(opening & (hcp <= 10), bid(1, 0))
            take(opening & (hcp >= 18), bid(2, 0))
            take(opening & balanced & (hcp >= 15) & (hcp <= 17), bid(1, 4))
            take(opening & (hcp >= 11) & (major_len >= 5), bid(1, major_strain))
            take(opening & (hcp >= 11), bid(1, 1))
        else:
            take(opening & (hcp >= 22), bid(2, 0))
            take(opening & balanced & (hcp >= 20) & (hcp <= 21), bid(2, 4))
            take(opening & balanced & (hcp >= 15) & (hcp <= 17), bid(1, 4))
            light = (hcp >= 9) & (long_len >= 5) if happy else hcp >= 12
            take(opening & light & (major_len >= 5), bid(1, major_strain))
            take(opening & light & (long_len >= 5) & (long_strain <= 1), bid(1, long_strain))
            take(opening & (hcp >= 12), bid(1, minor_strain))
            weak_two = opening & (long_len >= 6) & (hcp >= 5) & (hcp <= 10) & (long_strain >= 1)
            take(weak_two, bid(2, long_strain))

        # -------------------------------------------- partner opened: responses
        partner_opened = (opener_rel == 2) & ~i_bid
        p_strain = torch.where(partner_last >= 0, partner_last % 5, -1)
        p_major = (p_strain == 2) | (p_strain == 3)
        p_suit = torch.where(p_strain.clamp(min=0) < 4, 3 - p_strain.clamp(min=0, max=3), 0)
        support = length[rows, p_suit]
        weakclub_1c = (opening_call == 0) & (self.style == "weakclub")
        nt_opening = opening_call == 4
        resp = partner_opened & ~nt_opening & ~weakclub_1c
        take(resp & p_major & (support >= 3) & (hcp >= 13), bid(4, p_strain))
        take(resp & p_major & (support >= 3) & (hcp >= 10), bid(3, p_strain))
        take(resp & p_major & (support >= 3) & (hcp >= 6), bid(2, p_strain))
        new_suit, new_level = cheapest(long_strain)
        take(resp & (long_len >= 4) & (new_level == 1) & (hcp >= 6), new_suit)
        take(resp & (long_len >= 5) & (new_level == 2) & (hcp >= 10), new_suit)
        take(resp & balanced & (hcp >= 13), bid(3, 4))
        take(resp & (hcp >= 6) & (hcp <= 10), bid(1, 4))
        take(partner_opened & nt_opening & (hcp >= 10), bid(3, 4))
        take(partner_opened & nt_opening & (hcp >= 8), bid(2, 4))
        take(partner_opened & weakclub_1c & (hcp >= 8), new_suit)

        # ------------------------------------------ they opened: overcalls
        overcall = (opener_rel % 2 == 1) & ~we_bid
        need1, need2 = (5, 7) if happy else (8, 10)
        oc, oc_level = cheapest(long_strain)
        take(overcall & (long_len >= 5) & (oc_level == 1) & (hcp >= need1) & (hcp <= 16), oc)
        take(overcall & (long_len >= 5) & (oc_level == 2) & (hcp >= need2) & (hcp <= 16), oc)
        take(overcall & balanced & (hcp >= 15) & (hcp <= 18), cheapest(torch.full_like(last, 4))[0])

        # ------------------------------------------- later: raise or compete
        fit_len = support + torch.where(partner_last >= 0, 4, 0)       # assume partner has 4+
        fit = (partner_last >= 0) & (p_strain < 4) & (support >= 3)
        raise_bid, raise_level = cheapest(p_strain.clamp(min=0))
        total = hcp + torch.where(opener_rel == 2, 12, 7)               # partner's minimum
        extra = 1 if happy else 0
        compete = fit & they_bid & ~we_hold & (raise_level <= fit_len - 6 + extra)
        take(compete, raise_bid)
        game_strain = torch.where(fit & p_major, p_strain, 4)
        game = bid(torch.where(game_strain == 4, 3, 4), game_strain)
        take(we_bid & (partner_last >= 0) & (total >= 25), game)
        return call
