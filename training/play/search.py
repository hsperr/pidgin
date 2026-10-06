"""Perfect-information Monte Carlo for card play. Inference only.

The net is trained by pure self play and never sees a solver. This module is what a
finished checkpoint may do at the table: guess where the hidden cards are, solve the
resulting *complete* deals double dummy, and play the card that averages best. Nothing
here is importable from a training path, and nothing here writes a gradient.

Two halves:

* :class:`LayoutSampler` draws complete legal layouts of the cards this seat cannot see,
  from the net's belief head or from card counting alone. Every draw respects the cards
  already seen, how many cards each unseen seat still holds, and every void the play has
  revealed.
* :class:`PIMCPlayer` plugs into :mod:`training.play.match` in place of ``NetPlayer``.

The solver conversions live here so that the package, not an experiment script, owns the
one mapping between our card indices and :mod:`endplay`.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from training.bridge.deals import RANKS, deal_to_pbn
from training.bridge.play import N_CARDS, N_SEATS, PlayBatch
from training.play.data import Contracts
from training.play.model import PlayNet, encode

try:                                              # endplay is the optional `dds` extra
    from endplay.dds import solve_all_boards, solve_board
    from endplay.types import Card, Deal, Denom, Player, Rank

    DENOMS = (Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt)
    PLAYERS = (Player.north, Player.east, Player.south, Player.west)
    DDS_RANKS = (Rank.RA, Rank.RK, Rank.RQ, Rank.RJ, Rank.RT, Rank.R9, Rank.R8,
                 Rank.R7, Rank.R6, Rank.R5, Rank.R4, Rank.R3, Rank.R2)
    HAVE_ENDPLAY = True
except ImportError:                               # the sampler alone needs no solver
    HAVE_ENDPLAY = False
    DENOMS = PLAYERS = DDS_RANKS = ()

N_SUITS = 4
SOLVE_CHUNK = 64                                  # boards per solver call


def to_card(index: int):
    return Card(suit=DENOMS[index // 13], rank=DDS_RANKS[index % 13])


def from_card(card) -> int:
    return DENOMS.index(card.suit) * 13 + DDS_RANKS.index(card.rank)


def make_deals(contracts: Contracts) -> list:
    """One :class:`endplay.types.Deal` per board, set to the opening lead."""
    out = []
    for i in range(len(contracts)):
        deal = Deal(deal_to_pbn(contracts.owner[i].numpy()))
        deal.trump = DENOMS[int(contracts.trump[i])]
        deal.first = PLAYERS[(int(contracts.declarer[i]) + 1) % 4]
        out.append(deal)
    return out


def best_cards(deals: list, rows: list[int]) -> dict[int, int]:
    """Solve the named boards and return each one's best card, as our card index."""
    picked = {}
    for start in range(0, len(rows), SOLVE_CHUNK):
        part = rows[start:start + SOLVE_CHUNK]
        for row, solved in zip(part, solve_all_boards([deals[r] for r in part])):
            best, best_tricks = None, -1
            for card, tricks in solved:
                if tricks > best_tricks:
                    best, best_tricks = card, tricks
            picked[row] = from_card(best)
    return picked


# -- what the seat on turn may legally know ---------------------------------


def trick_leaders(batch: PlayBatch) -> torch.Tensor:
    """``(n, 13)`` the seat that led each trick. Only tricks already started are set."""
    leaders = torch.zeros(batch.n, 13, dtype=torch.long, device=batch.device)
    leaders[:, 0] = (batch.declarer + 1) % N_SEATS
    if batch.trick_no > 0:
        leaders[:, 1:] = batch.trick_winner[:, :12]
    return leaders


def revealed_voids(batch: PlayBatch) -> torch.Tensor:
    """``(n, 4, 4)`` bool: seat has shown out of suit, so it cannot hold one.

    A seat that failed to follow the led suit is void in it forever. This is the only
    hard information the play gives away, and it is public: everyone at the table sees it.
    """
    void = torch.zeros(batch.n, N_SEATS, N_SUITS, dtype=torch.bool, device=batch.device)
    if batch.t == 0:
        return void
    leaders = trick_leaders(batch)
    steps = torch.arange(batch.t, device=batch.device)
    trick, pos = steps // N_SEATS, steps % N_SEATS
    player = (leaders[:, trick] + pos[None, :]) % N_SEATS                  # (n, t)
    cards = batch.history[:, :batch.t]
    led = batch.suit_of[batch.history[:, trick * N_SEATS]]                 # (n, t)
    off = (batch.suit_of[cards] != led) & (pos[None, :] > 0)
    rows = batch.rows[:, None].expand_as(off)
    void[rows[off], player[off], led[off]] = True
    return void


def cards_left(batch: PlayBatch) -> torch.Tensor:
    """``(n, 4)`` how many cards each seat still holds. Public: 13 minus cards played."""
    seats = torch.arange(N_SEATS, device=batch.device)
    return ((batch.owner[:, None, :] == seats[None, :, None]) & batch.unplayed[:, None, :]
            ).sum(-1)


def seat_probs(belief: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
    """Belief logits over *relative* seats -> ``(n, 52, 4)`` probabilities over absolute."""
    probs = torch.softmax(belief, -1)
    absolute = torch.empty_like(probs)
    for rel in range(N_SEATS):
        seat = (turn[:, None] + rel) % N_SEATS
        absolute.scatter_(2, seat[:, :, None].expand(-1, N_CARDS, 1), probs[:, :, rel:rel + 1])
    return absolute


@dataclass
class Position:
    """One row's decision point, holding only what the seat on turn may see."""

    row: int
    turn: int
    leader: int                      # who led the current trick
    trump: int
    trick: list[tuple[int, int]]     # (seat, card) already on the table this trick
    holding: dict[int, list[int]]    # seat -> cards still held, for the seats we can see
    hidden: list[int]                # cards we cannot see, still unplayed
    caps: dict[int, int]             # hidden seat -> how many of them it holds
    suit_seats: list[list[int]]      # suit -> hidden seats not known void in it
    probs: dict[int, dict[int, float]] | None    # card -> hidden seat -> weight


def read_positions(batch: PlayBatch, contracts: Contracts, dummy_visible: bool,
                   belief: torch.Tensor | None, rows: list[int]) -> dict[int, Position]:
    """Extract a :class:`Position` per named row. ``belief`` None means card counting."""
    turn = batch.to_play()
    dummy = (contracts.declarer + 2) % N_SEATS
    partner = torch.where(turn == contracts.declarer, dummy,
                          torch.where(turn == dummy, contracts.declarer, dummy))
    counts = cards_left(batch)
    void = revealed_voids(batch)
    leaders = trick_leaders(batch)[:, batch.trick_no]
    trick_cards = batch.trick_cards()
    probs = seat_probs(belief, turn) if belief is not None else None

    out = {}
    for row in rows:
        me, mate = int(turn[row]), int(partner[row])
        seen_seats = [me] + ([mate] if dummy_visible else [])
        hidden_seats = [s for s in range(N_SEATS) if s not in seen_seats]
        holding, hidden = {}, []
        for card in range(N_CARDS):
            if not bool(batch.unplayed[row, card]):
                continue
            owner = int(batch.owner[row, card])
            if owner in seen_seats:
                holding.setdefault(owner, []).append(card)
            else:
                hidden.append(card)
        for seat in seen_seats:
            holding.setdefault(seat, [])
        caps = {s: int(counts[row, s]) for s in hidden_seats}
        suit_seats = [[s for s in hidden_seats if not bool(void[row, s, suit])]
                      for suit in range(N_SUITS)]
        weight = (None if probs is None else
                  {c: {s: float(probs[row, c, s]) for s in hidden_seats} for c in hidden})
        out[row] = Position(
            row=row, turn=me, leader=int(leaders[row]), trump=int(contracts.trump[row]),
            trick=[(int((leaders[row] + i) % N_SEATS), int(trick_cards[row, i]))
                   for i in range(batch.pos)],
            holding=holding, hidden=hidden, caps=caps, suit_seats=suit_seats, probs=weight,
        )
    return out


# -- drawing a layout -------------------------------------------------------


def _subsets(seats: list[int]) -> list[list[int]]:
    out = []
    for mask in range(1, 1 << len(seats)):
        out.append([s for i, s in enumerate(seats) if mask >> i & 1])
    return out


def _feasible(caps: dict[int, int], left: list[int], suit_seats: list[list[int]],
              groups: list[list[int]]) -> bool:
    """Can the cards still unassigned fill every remaining seat exactly?

    Gale-Hoffman on the tiny transportation problem "suits -> seats": a set of seats can
    be filled only if the suits that may reach it hold at least that many cards. Four
    suits and at most three unseen seats, so the whole test is seven small sums.
    """
    for group in groups:
        need = sum(caps[s] for s in group)
        supply = sum(left[suit] for suit in range(N_SUITS)
                     if any(s in group for s in suit_seats[suit]))
        if need > supply:
            return False
    return True


class LayoutSampler:
    """Draw complete legal layouts of the hidden cards.

    Why the scheme is sound. The constraints are exactly three: a card that has been
    seen is never re-dealt, each unseen seat receives precisely the number of cards it
    still holds, and no seat receives a suit it has already shown out of. Cards are
    dealt one at a time; before each draw a seat is struck off if it is full, if it is
    void in that suit, or if giving it the card would leave the rest of the deal
    impossible (the Gale-Hoffman test above). So the sampler can never paint itself into
    a corner, never needs to reject and restart, and every layout consistent with the
    three constraints keeps positive probability.

    Where it is biased. The belief head gives *independent per-card* marginals, and the
    true posterior conditioned on the counts is not their product. Dealing sequentially
    with renormalisation samples from a chain of conditionals built out of unconditioned
    marginals, which is not the same distribution: it tilts towards layouts whose
    confident cards happen to come early in the order. The order is shuffled per draw so
    no card is systematically favoured, but the tilt does not vanish. Correcting it would
    need importance weights, or fitting the marginals to the count constraint first.
    Two further approximations, both outside this class: the marginals themselves are
    only as calibrated as the head, and nothing here reasons about *why* an opponent
    chose the card they did beyond what the head has already learned.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def draw(self, position: Position) -> dict[int, int]:
        """One layout: ``card -> seat`` for every hidden card."""
        rng = self.rng
        caps = dict(position.caps)
        seats = list(caps)
        groups = _subsets(seats)
        # A void only bites when it hides a seat, so skip the feasibility test otherwise.
        constrained = any(len(position.suit_seats[s]) < len(seats) for s in range(N_SUITS))
        left = [0] * N_SUITS
        for card in position.hidden:
            left[card // 13] += 1

        order = list(position.hidden)
        rng.shuffle(order)
        # Deal the cards with the fewest homes first: it keeps the feasibility mask from
        # having to rescue a position that a free choice has already spoiled.
        order.sort(key=lambda c: len(position.suit_seats[c // 13]))

        layout = {}
        for card in order:
            suit = card // 13
            left[suit] -= 1
            choices, weights, total = [], [], 0.0
            for seat in position.suit_seats[suit]:
                if caps[seat] == 0:
                    continue
                if constrained:
                    caps[seat] -= 1
                    ok = _feasible(caps, left, position.suit_seats, groups)
                    caps[seat] += 1
                    if not ok:
                        continue
                # No belief: weight by how many cards the seat still needs, which with no
                # void in play is exactly the uniform draw over legal layouts. That is
                # what counting the cards on its own tells you and nothing more.
                w = float(caps[seat]) if position.probs is None else position.probs[card][seat]
                choices.append(seat)
                weights.append(w)
                total += w
            if not choices:
                raise RuntimeError("no legal seat for a hidden card: constraints are wrong")
            if total <= 0.0:                       # belief said zero everywhere; fall back
                weights = [1.0] * len(choices)
                total = float(len(choices))
            pick = rng.random() * total
            chosen = choices[-1]
            for seat, w in zip(choices, weights):
                pick -= w
                if pick <= 0.0:
                    chosen = seat
                    break
            layout[card] = chosen
            caps[chosen] -= 1
        return layout


def true_layout(batch: PlayBatch, position: Position) -> dict[int, int]:
    return {card: int(batch.owner[position.row, card]) for card in position.hidden}


def layout_is_legal(position: Position, layout: dict[int, int]) -> bool:
    """Does a layout satisfy the counts and the voids? The tests live on this."""
    if set(layout) != set(position.hidden):
        return False
    held = {seat: 0 for seat in position.caps}
    for card, seat in layout.items():
        if seat not in position.caps or seat not in position.suit_seats[card // 13]:
            return False
        held[seat] += 1
    return held == position.caps


# -- solving a sampled layout ----------------------------------------------


def _pbn(hands: list[list[int]]) -> str:
    out = []
    for seat in range(N_SEATS):
        cards = set(hands[seat])
        out.append(".".join(
            "".join(RANKS[r] for r in range(13) if suit * 13 + r in cards)
            for suit in range(N_SUITS)))
    return "N:" + " ".join(out)


def deal_for(position: Position, layout: dict[int, int]):
    """An :mod:`endplay` deal at this decision point, with the hidden cards as drawn.

    Rebuilt from the start of the current trick — the cards on the table are put back in
    the hands that played them and replayed — because that is the only state endplay will
    accept mid-trick. Earlier tricks do not matter: the solver scores the tricks that are
    left, and those cards are simply gone.
    """
    hands: list[list[int]] = [[] for _ in range(N_SEATS)]
    for seat, cards in position.holding.items():
        hands[seat].extend(cards)
    for card, seat in layout.items():
        hands[seat].append(card)
    for seat, card in position.trick:
        hands[seat].append(card)
    deal = Deal(_pbn(hands), complete_deal=False)
    deal.trump = DENOMS[position.trump]
    deal.first = PLAYERS[position.leader]
    for _, card in position.trick:
        deal.play(to_card(card))
    return deal


MISPAIRED = [0, 0]                # deals solved, and tables the batch solver got wrong


def solved_values(deals: list) -> list[dict[int, int]]:
    """Tricks for the side on turn, per card, one dict per deal.

    A batch holding a deal *and* that same deal one card later occasionally comes back
    with the neighbour's table, listing cards that are not legal in the board it was
    asked about — measured at roughly 1 in 10000 solves by
    ``experiments/play/depth2.py``, which is the only caller that batches positions of
    different depths. PIMC never does (all its samples share one decision point) and
    ``best_cards`` never does (a ``PlayBatch`` advances in lockstep), so today every
    caller is safe — but safe by a precondition none of them states. Each table is
    checked against the deal's own legal moves and the odd one re-solved alone, so a
    future caller that does batch a tree cannot inherit the bug silently.
    """
    out = []
    for start in range(0, len(deals), SOLVE_CHUNK):
        part = deals[start:start + SOLVE_CHUNK]
        for deal, solved in zip(part, solve_all_boards(part)):
            table = {from_card(card): tricks for card, tricks in solved}
            MISPAIRED[0] += 1
            if set(table) != {from_card(card) for card in deal.legal_moves()}:
                MISPAIRED[1] += 1
                table = {from_card(card): tricks for card, tricks in solve_board(deal)}
            out.append(table)
    return out


# -- the player -------------------------------------------------------------


class PIMCPlayer:
    """The net, with the declaring side's cards chosen by perfect-information search.

    Declarer only, on purpose. A defender's search would have to model partner, who is
    also guessing and whose cards carry signals; declarer sees dummy and plays both hands,
    so the sampled layout is a fair picture of what is left to find out.

    Defence is discarded rather than refused: the harnesses ask both players for every
    row and keep the one whose seat is on turn, so a defensive row's answer is thrown
    away anyway and we simply do not pay for it.
    """

    @classmethod
    def from_net(cls, net: PlayNet, samples: int, **kwargs) -> "PIMCPlayer":
        """Search with a net someone else already loaded.

        The play server holds one :class:`PlayNet` per checkpoint and serves a live
        table from it. Loading a second copy just to search would double the weights
        in memory for no gain, and the box it runs on is small.
        """
        return cls(None, samples, net=net, **kwargs)

    DEFENCE_MODES = ("off", "lead", "all", "only")

    def __init__(self, path: str | None, samples: int, seed: int = 0,
                 counting: bool = False, budget_ms: float | None = None,
                 net: PlayNet | None = None, defence: str = "off",
                 defence_from_trick: int = 0):
        """``budget_ms`` caps the wall clock of one decision, for serving a live table.

        A solve costs ~300x more at trick one than at trick seven, and 34x more on a
        small cloud box than on a laptop, so a fixed sample count is either too slow
        early or wasteful late. With a budget the samples are drawn in chunks until the
        clock runs out, which self-tunes to the machine. Leave it None for measurement:
        the offline numbers have to be reproducible, and a clock is not.
        """
        if not HAVE_ENDPLAY:
            raise RuntimeError("PIMC needs the `dds` extra (endplay)")
        if path is None:                          # see :meth:`from_net`
            self.net, name = net, "pimc"
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
            self.net = PlayNet(**state["config"])
            self.net.load_state_dict(state["net"])
            self.net.eval()
            name = Path(path).parent.name
        # Every knob that changes the cards must show in the name: it is the only
        # identity line most runs print, and `--defence-from-trick 0` vs `2` and
        # `--search-counting` on vs off are comparisons this project actually makes.
        self.name = f"{name}+pimc{samples}"
        if counting:
            self.name += "+counting"
        if defence != "off":
            self.name += f"+def-{defence}"
            if defence_from_trick:
                self.name += f"@t{defence_from_trick}"
        if budget_ms is not None:
            self.name += f"+{budget_ms:.0f}ms"
        self.samples = samples
        self.budget_ms = budget_ms
        self.counting = counting
        if defence not in self.DEFENCE_MODES:
            raise ValueError(f"defence must be one of {self.DEFENCE_MODES}")
        # "lead" is the cheap, well-founded case: at the opening lead nothing has been
        # played, so there are no partner signals to misread -- the only difficulty is
        # that 39 cards are unseen, which is exactly what the belief head is for. "all"
        # searches every defensive turn too, where partner is guessing as well and the
        # sampled layout is a weaker picture. "only" searches the defenders and leaves
        # declarer to the net; it exists so distillation can generate defensive targets
        # without paying for declarer solves it is going to throw away.
        self.defence = defence
        # A solve costs ~290ms at trick one on the droplet and ~1ms by trick seven, so
        # nearly all of defence search's price is the first few tricks. Skipping them
        # keeps most of the decisions and almost none of the cost.
        self.defence_from_trick = defence_from_trick
        self.sampler = LayoutSampler(seed)
        self.auction = None
        self.solves = 0
        self.seats = None            # set by the harness when it knows which chairs are ours
        # Last decision's per-card averages, row -> card -> mean tricks. The search throws
        # these away once it has named a card; distillation is the one caller that wants
        # the whole table, so it is kept rather than recomputed.
        self.values: dict[int, dict[int, float]] = {}

    def start(self, contracts: Contracts) -> None:
        with torch.no_grad():
            self.auction = self.net.auction(contracts.calls, contracts.n_calls,
                                            contracts.dealer)

    def choose(self, batch: PlayBatch, contracts: Contracts, legal: torch.Tensor,
               step: int) -> torch.Tensor:
        with torch.no_grad():
            view = encode(batch, contracts, step > 0, self.auction)
            out = self.net(view["features"], legal)
        card = out["log_probs"].argmax(-1)
        self.values = {}
        if self.samples <= 0:
            return card

        turn = batch.to_play()
        dummy = (contracts.declarer + 2) % N_SEATS
        declaring = (turn == contracts.declarer) | (turn == dummy)
        if self.defence == "all" and batch.trick_no >= self.defence_from_trick:
            declaring = torch.ones_like(declaring)
        elif self.defence == "lead" and step == 0:
            declaring = torch.ones_like(declaring)
        elif self.defence == "only":
            declaring = (~declaring if batch.trick_no >= self.defence_from_trick
                         else torch.zeros_like(declaring))
        if self.seats is not None:
            mine = torch.zeros_like(declaring)
            for seat in self.seats:
                mine |= turn == seat
            declaring &= mine
        rows = torch.nonzero(declaring & (legal.sum(1) > 1)).squeeze(1).tolist()
        if not rows:
            return card

        belief = None if self.counting else out["belief"]
        positions = read_positions(batch, contracts, step > 0, belief, rows)
        totals: dict[int, dict[int, int]] = {row: {} for row in rows}

        # Without a budget every sample is drawn up front, which keeps the solver call
        # one batch and the result independent of how fast the machine is. With one, the
        # same draws are taken in chunks and the clock decides where to stop.
        chunk = self.samples if self.budget_ms is None else max(1, self.samples // 8)
        deadline = None if self.budget_ms is None else time.monotonic() + self.budget_ms / 1000
        drawn = 0
        while drawn < self.samples:
            take = min(chunk, self.samples - drawn)
            deals, owners = [], []
            for row in rows:
                position = positions[row]
                for _ in range(take):
                    deals.append(deal_for(position, self.sampler.draw(position)))
                    owners.append(row)
            self.solves += len(deals)
            for row, values in zip(owners, solved_values(deals)):
                bag = totals[row]
                for index, tricks in values.items():
                    bag[index] = bag.get(index, 0) + tricks
            drawn += take
            if deadline is not None and time.monotonic() >= deadline:
                break
        prior = out["log_probs"]
        self.values = {row: {index: tricks / drawn for index, tricks in totals[row].items()}
                       for row in rows if totals[row]}
        for row in rows:
            bag = totals[row]
            if not bag:
                continue
            # Ties go to the net: among cards the solver cannot separate, its own
            # preference is the only thing left that knows about a fallible opponent.
            card[row] = max(bag, key=lambda index: (bag[index], float(prior[row, index])))
        return card
