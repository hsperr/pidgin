"""Duplicate IMP matches between two card players.

The only question worth asking is "is this net stronger than that one, and by how much".
Trick counts cannot answer it: better defence hides better declarer play, and a trick is
worth a different amount in every contract.

So we play duplicate. One board, two tables, the same contract at both. The challenger
sits North-South at table one, the reference sits North-South at table two, and the *same*
fixed opponent plays East-West at both. Score each table from North-South's side, take the
difference, turn it into IMPs. The deal is identical at both tables, so most of the luck
cancels and the error bar shrinks.

    python -m bridgezero.play.match BENCH.npz --challenger runs/x/last.pt \\
        --reference random --opponent random --deals 20000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from bridgezero.bridge.play import PlayBatch
from bridgezero.bridge.scoring import contract_score, imps
from bridgezero.play.data import Contracts, load_contracts
from bridgezero.play.model import PlayNet, encode
from bridgezero.bridge.calls import STRAIN_PERM

TRUMP_TO_BID_STRAIN = STRAIN_PERM         # its own inverse, so this is the same tuple


class RandomPlayer:
    """Plays a legal card at random. The floor."""

    name = "random"

    def start(self, contracts: Contracts) -> None:
        pass

    def choose(self, batch: PlayBatch, contracts: Contracts, legal: torch.Tensor,
               step: int) -> torch.Tensor:
        return torch.multinomial(legal.float(), 1).squeeze(1)


class NetPlayer:
    """A checkpoint, playing its best card. Greedy, so a match is repeatable."""

    def __init__(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net = PlayNet(**state["config"])
        self.net.load_state_dict(state["net"])
        self.net.eval()
        self.name = Path(path).parent.name
        self.auction = None

    def start(self, contracts: Contracts) -> None:
        with torch.no_grad():
            self.auction = self.net.auction(contracts.calls, contracts.n_calls,
                                            contracts.dealer)

    def choose(self, batch: PlayBatch, contracts: Contracts, legal: torch.Tensor,
               step: int) -> torch.Tensor:
        with torch.no_grad():
            view = encode(batch, contracts, step > 0, self.auction)
            return self.net(view["features"], legal)["log_probs"].argmax(-1)


def make_player(spec: str, search: int = 0, counting: bool = False,
                defence: str = "off", defence_from_trick: int = 0):
    """``search`` layouts per decision turns on PIMC; 0 is the plain greedy net.

    ``counting`` draws those layouts from the card counts alone instead of from the
    belief head, which is the control the search has to beat to have earned the head.
    """
    if spec == "random":
        return RandomPlayer()
    if search > 0:
        from bridgezero.play.search import PIMCPlayer      # needs the optional dds extra
        return PIMCPlayer(spec, search, counting=counting, defence=defence,
                          defence_from_trick=defence_from_trick)
    return NetPlayer(spec)


@torch.no_grad()
def play_table(contracts: Contracts, north_south, east_west) -> PlayBatch:
    """One table. Seats 0 and 2 are North-South, seats 1 and 3 are East-West."""
    north_south.start(contracts)
    east_west.start(contracts)
    # Both players are asked for every row and the wrong one's answer is thrown away.
    # A searcher that knows its chairs can skip the rows it will never be asked to play.
    for player, seats in ((north_south, (0, 2)), (east_west, (1, 3))):
        if hasattr(player, "seats"):
            player.seats = seats
    batch = PlayBatch(contracts.owner.clone(), contracts.trump, contracts.declarer)
    for step in range(52):
        legal = batch.legal()
        ours = north_south.choose(batch, contracts, legal, step)
        theirs = east_west.choose(batch, contracts, legal, step)
        batch.play(torch.where(batch.to_play() % 2 == 0, ours, theirs))
    return batch


def north_south_score(contracts: Contracts, batch: PlayBatch) -> torch.Tensor:
    """Duplicate score from North-South's side, one number per board."""
    tricks = batch.declarer_tricks().tolist()
    declarer_is_ns = (contracts.declarer % 2 == 0)
    vulnerable = torch.where(declarer_is_ns, contracts.vul_ns, contracts.vul_ew)
    out = []
    for i, won in enumerate(tricks):
        raw = contract_score(int(contracts.level[i]),
                             TRUMP_TO_BID_STRAIN[int(contracts.trump[i])],
                             won, int(contracts.doubled[i]), bool(vulnerable[i]))
        out.append(raw if bool(declarer_is_ns[i]) else -raw)
    return torch.tensor(out, dtype=torch.float32)


def match(contracts: Contracts, challenger, reference, opponent) -> dict:
    """Challenger against reference, both facing the same opponent, on the same boards.

    The whole point is that the two tables differ in one thing only, so the deal's luck
    cancels. A *stochastic* opponent breaks that: run back to back, the second table
    draws from a different point in the global RNG stream and plays different cards, so
    the opponent's 26 cards stop cancelling. It is unbiased, but on 200 boards with the
    same checkpoint on both sides it leaves ~3.9 IMP of per-board noise that the design
    exists to remove. Both tables therefore start from the same RNG state.
    """
    state = torch.get_rng_state()
    ours = play_table(contracts, challenger, opponent)
    torch.set_rng_state(state)
    theirs = play_table(contracts, reference, opponent)
    mine, yours = north_south_score(contracts, ours), north_south_score(contracts, theirs)
    board = torch.tensor([float(imps(d)) for d in (mine - yours).tolist()])
    n = len(board)
    return {
        "boards": n,
        "imps_per_board": board.mean().item(),
        "standard_error": (board.std().item() / math.sqrt(n)) if n > 1 else float("nan"),
        "challenger_score": mine.mean().item(),
        "reference_score": yours.mean().item(),
        "challenger_declarer_tricks": ours.declarer_tricks().float().mean().item(),
        "reference_declarer_tricks": theirs.declarer_tricks().float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--challenger", required=True, help="checkpoint path, or 'random'")
    parser.add_argument("--reference", default="random", help="checkpoint path, or 'random'")
    parser.add_argument("--opponent", default="random", help="checkpoint path, or 'random'")
    parser.add_argument("--deals", type=int, default=20000)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-defence", default="off",
                        choices=("off", "lead", "all", "only"),
                        help="also search on defence: the opening lead only, every turn, "
                             "or 'only' for the defenders and not declarer")
    parser.add_argument("--defence-from-trick", type=int, default=0,
                        help="with --search-defence all/only, skip the dear early tricks")
    parser.add_argument("--search", type=int, default=0,
                        help="PIMC layouts per challenger decision; 0 is the plain net")
    parser.add_argument("--reference-search", type=int, default=0,
                        help="same for the reference, so searcher can be matched against "
                             "searcher; 0 keeps the reference a plain net")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    every = load_contracts(args.benchmark, args.deals)
    challenger = make_player(args.challenger, args.search, defence=args.search_defence,
                             defence_from_trick=args.defence_from_trick)
    reference = make_player(args.reference, args.reference_search)
    opponent = make_player(args.opponent)

    totals = []
    for start in range(0, len(every), args.chunk):
        part = every.subset(torch.arange(start, min(start + args.chunk, len(every))))
        result = match(part, challenger, reference, opponent)
        totals.append(result)
        print(f"{start + result['boards']}/{len(every)} boards", flush=True)

    n = sum(r["boards"] for r in totals)
    mean = sum(r["imps_per_board"] * r["boards"] for r in totals) / n
    # Pool the per-chunk errors: each is std/sqrt(m), so variance recombines by weight.
    # A chunk of one board has no standard error (nan), and one nan would make the whole
    # interval nan. That happens whenever the kept boards are 1 mod --chunk, which is luck,
    # not a choice: load_contracts drops passed-out deals. A single board contributes no
    # variance estimate, so drop it from the pool rather than poisoning it.
    var = sum((r["standard_error"] ** 2) * r["boards"] ** 2
              for r in totals if r["boards"] > 1) / n ** 2
    error = math.sqrt(var)
    print(f"\n{challenger.name} vs {reference.name}, opponent {opponent.name}")
    print(f"{n} boards   {mean:+.3f} +/- {1.96 * error:.3f} IMP per board (95%)")
    print(f"declarer tricks: challenger {sum(r['challenger_declarer_tricks'] * r['boards'] for r in totals) / n:.3f}"
          f"   reference {sum(r['reference_declarer_tricks'] * r['boards'] for r in totals) / n:.3f}")


if __name__ == "__main__":
    main()
