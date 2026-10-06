"""Learn card play from self play alone.

The net sits in all four chairs. It plays the deal out, and the only signal is how
many tricks each side took. No solver labels, no copying anybody.

Three losses:
  * policy   REINFORCE on the tricks a side still wins, with the critic as baseline.
  * value    the critic learns those same remaining tricks, with all hands face up.
  * belief   guess the seat of every hidden card. Free labels: we dealt the cards.

``--group K`` swaps the learned critic for a duplicate-bridge baseline: play the same
deal K times and score each play-out against the mean of its siblings. Bridge is scored
in IMPs for exactly this reason -- hold the cards constant and the deal's difficulty,
which is most of the variance, cancels instead of having to be predicted.

The double dummy table is used only to report a score. It never touches a gradient.

    python -m training.play.train AUCTIONS.npz --out runs/play_first --steps 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from training.bridge.play import PlayBatch
from training.play.data import Contracts, load_contracts
from training.play.model import Critic, PlayNet, encode


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pool:
    """The last few snapshots of the learner, kept as opponents.

    Pure self play lets both sides drift together: the net can look better every step
    while only learning to beat its own latest habits. A frozen snapshot cannot drift,
    so beating it means something. The snapshots play at a temperature rather than
    greedily, which gives variety for free without training them separately.
    """

    def __init__(self, size: int, width: int, depth: int):
        self.size = size
        self.slots: list[PlayNet] = []
        self.width, self.depth = width, depth

    def add(self, net: PlayNet) -> None:
        if self.size == 0:
            return
        copy = PlayNet(self.width, self.depth).to(next(net.parameters()).device)
        copy.load_state_dict(net.state_dict())
        copy.eval()
        for p in copy.parameters():
            p.requires_grad_(False)
        self.slots.append(copy)
        del self.slots[:-self.size]

    def sample(self, generator) -> PlayNet | None:
        if not self.slots:
            return None
        return self.slots[int(torch.randint(len(self.slots), (1,), generator=generator))]


def rollout(net: PlayNet, contracts: Contracts, temperature: float = 1.0,
            greedy: bool = False, grad: bool = True, learn_from: int = 0,
            warmup: str = "net", opponent: PlayNet | None = None,
            learner_side: torch.Tensor | None = None,
            opponent_temperature: float = 1.0):
    """Play every deal out. Returns the finished batch and one record per decision.

    ``learn_from`` is the first trick we learn from. Earlier tricks are still played, so
    the deal is whole and the trick counts are real, but they leave no training record.
    That is the backward curriculum: set ``learn_from`` to 10 and the net only answers the
    last three tricks, where almost every card is already on the table and the right play
    is nearly countable. Lower it as the net gets better.

    ``warmup`` picks who plays those early tricks: ``net`` (its own cards, no gradient) or
    ``random``. ``net`` keeps the late positions in the distribution the net itself makes.

    With an ``opponent``, the learner holds only the side named by ``learner_side`` (0 for
    North-South, 1 for East-West, one value per deal) and the opponent holds the other two
    seats. Only the learner's decisions carry a ``mask`` of True, so only they train. The
    deal is still played out whole, so the trick counts stay honest.
    """
    batch = PlayBatch(contracts.owner.clone(), contracts.trump, contracts.declarer)
    with torch.set_grad_enabled(grad):
        auction = net.auction(contracts.calls, contracts.n_calls, contracts.dealer)
    if opponent is not None:
        with torch.no_grad():
            their_auction = opponent.auction(contracts.calls, contracts.n_calls,
                                             contracts.dealer)
    steps = []
    for t in range(52):
        legal = batch.legal()
        if batch.trick_no < learn_from:
            with torch.no_grad():
                if warmup == "net":
                    out = net(encode(batch, contracts, t > 0, auction)["features"], legal)
                    card = torch.multinomial(out["log_probs"].exp(), 1).squeeze(1)
                else:
                    card = torch.multinomial(legal.float(), 1).squeeze(1)
            batch.play(card)
            continue
        view = encode(batch, contracts, t > 0, auction)
        with torch.set_grad_enabled(grad):
            out = net(view["features"], legal)
        log_probs = out["log_probs"]
        if greedy:
            card = log_probs.argmax(-1)
        else:
            probs = (log_probs / temperature).softmax(-1)
            card = torch.multinomial(probs, 1).squeeze(1)
        mine = torch.ones(batch.n, dtype=torch.bool, device=batch.device)
        if opponent is not None:
            mine = (view["turn"] % 2) == learner_side
            with torch.no_grad():
                theirs = opponent(encode(batch, contracts, t > 0, their_auction)["features"],
                                  legal)["log_probs"]
                their_card = torch.multinomial(
                    (theirs / opponent_temperature).softmax(-1), 1).squeeze(1)
            card = torch.where(mine, card, their_card)
        steps.append({
            "log_prob": log_probs.gather(1, card[:, None]).squeeze(1),
            "entropy": -(log_probs.exp() * log_probs.nan_to_num(neginf=0.0)).sum(1),
            "critic": view["critic"], "turn": view["turn"], "belief": out["belief"],
            "belief_target": view["belief_target"], "belief_mask": view["belief_mask"],
            "trick_no": batch.trick_no, "mask": mine,
        })
        batch.play(card)
    return batch, steps


def returns_for(batch: PlayBatch, steps: list) -> torch.Tensor:
    """Tricks the side on turn still wins, counted from the current trick onward."""
    rows = torch.arange(batch.n, device=batch.device)
    out = []
    per_trick = torch.zeros(batch.n, 13, 2, dtype=torch.long, device=batch.device)
    per_trick[rows[:, None], torch.arange(13, device=batch.device)[None, :],
              batch.trick_winner % 2] = 1
    from_here = per_trick.flip(1).cumsum(1).flip(1)       # (n, 13, 2) tricks left to win
    for step in steps:
        side = step["turn"] % 2
        out.append(from_here[rows, step["trick_no"], side].float())
    return torch.stack(out, 1)


def replicate(x: torch.Tensor, k: int) -> torch.Tensor:
    """K copies of every row, side by side, or the row itself when grouping is off."""
    return x.repeat_interleave(k) if k else x


def group_advantage(steps: list, target: torch.Tensor, k: int):
    """Duplicate-bridge baseline. Returns the advantage and the within-group spread.

    The batch holds K play-outs of each deal, laid out next to each other. Play runs in
    lockstep, so at step ``s`` every row of a group sits at the same trick and the same
    position in it, on the same cards; the rows differ only in how the policy sampled its
    way there. That is a duplicate set, and subtracting the group mean cancels the deal
    difficulty the same way an IMP score does.

    Grouping is at the decision step, not at the deal, because the return itself is per
    step. Two things make that sound:

    * ``target`` counts tricks for the *side on turn*, and the side on turn at step ``s``
      differs between play-outs as soon as a trick goes the other way. Both readings are
      restated as tricks for North-South -- one quantity, comparable across the group --
      and the sign is put back at the end.
    * the mean leaves the row itself out. A row's own return depends on the card it chose
      at this very step, and a baseline that saw the action would bias the gradient. The
      siblings never see that card, so their mean is independent of it given the deal and
      the row's own history, which is all an unbiased baseline has to be.
    """
    n_steps = len(steps)
    trick = torch.tensor([s["trick_no"] for s in steps], device=target.device)
    side = torch.stack([s["turn"] % 2 for s in steps], 1)            # 0 North-South
    left = (13 - trick).float()[None, :]                             # tricks still to play
    ours = torch.where(side == 0, target, left - target)             # always North-South
    grouped = ours.view(-1, k, n_steps)
    others = (grouped.sum(1, keepdim=True) - grouped) / (k - 1)
    sign = torch.where(side == 0, 1.0, -1.0)
    return sign * (ours - others.reshape(-1, n_steps)), grouped.std(1).mean()


@torch.no_grad()
def play_out(net: PlayNet, contracts: Contracts, control: str) -> PlayBatch:
    """Play a deal out. ``control`` says which seats the net holds; the rest play at random."""
    batch = PlayBatch(contracts.owner.clone(), contracts.trump, contracts.declarer)
    auction = net.auction(contracts.calls, contracts.n_calls, contracts.dealer)
    for t in range(52):
        legal = batch.legal()
        turn = batch.to_play()
        if control == "all":
            mine = torch.ones_like(turn, dtype=torch.bool)
        elif control == "declarer":
            mine = turn % 2 == contracts.declarer % 2
        elif control == "defence":
            mine = turn % 2 != contracts.declarer % 2
        else:
            mine = torch.zeros_like(turn, dtype=torch.bool)
        card = torch.multinomial(legal.float(), 1).squeeze(1)
        if mine.any():
            out = net(encode(batch, contracts, t > 0, auction)["features"], legal)
            card = torch.where(mine, out["log_probs"].argmax(-1), card)
        batch.play(card)
    return batch


def evaluate(net: PlayNet, contracts: Contracts) -> dict:
    """Tricks against the double dummy answer. Lower gap is better.

    ``trick_gap`` has the net on both sides, so it mixes declarer play with defence: it
    only says how much better declarer does than the net's own defenders let it. The two
    one-sided gaps are the honest ones. Each fixes the other side to random cards, so
    ``declarer_gap`` falls only when declarer play improves.
    """
    dd = contracts.dd_tricks.float()
    target = (contracts.level + 6).float()
    both = play_out(net, contracts, "all").declarer_tricks().float()
    solo = play_out(net, contracts, "declarer").declarer_tricks().float()
    defend = play_out(net, contracts, "defence").declarer_tricks().float()
    chaos = play_out(net, contracts, "none").declarer_tricks().float()
    return {
        "trick_gap": (dd - both).mean().item(),
        "declarer_gap": (solo - chaos).mean().item(),     # extra tricks the net's declarer wins
        "defence_gap": (chaos - defend).mean().item(),    # tricks the net's defence steals back
        "random_gap": (dd - chaos).mean().item(),
        "made": (both >= target).float().mean().item(),
    }


def belief_weight_at(step: int, args) -> float:
    """The belief head is a warm-up task, not a permanent one.

    Guessing where the hidden cards are teaches the trunk to read a position, and early on
    that is most of what it has to learn. But it saturates by roughly step 8000 and then
    keeps pulling the shared trunk sideways at a quarter of the policy's gradient, at a
    job it can no longer do better. So the weight decays to a floor and the policy gets
    the trunk to itself for the rest of the run. ``--belief-final`` equal to
    ``--belief-weight`` keeps the old constant behaviour.
    """
    if args.belief_final is None or args.belief_final == args.belief_weight:
        return args.belief_weight
    over = max(1, int(args.steps * args.belief_decay_by))
    part = min(1.0, step / over)
    return args.belief_weight + (args.belief_final - args.belief_weight) * part


def curriculum_start(step: int, args) -> int:
    """First trick we learn from. 0 means the whole deal.

    With ``--last-tricks 3`` the net starts on the last three tricks only, and the window
    opens backwards until it covers the whole deal at ``--grow-by`` of the way through.
    """
    if args.last_tricks >= 13:
        return 0
    grow = max(1, int(args.steps * args.grow_by))
    learned = args.last_tricks + (13 - args.last_tricks) * min(1.0, step / grow)
    return max(0, 13 - int(learned))


def train(args):
    if args.group:
        if args.group < 2:
            raise SystemExit("--group needs at least 2 play-outs to have a baseline")
        if args.batch % args.group:
            raise SystemExit(f"--batch {args.batch} is not a multiple of --group {args.group}")
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    every = load_contracts(args.auctions, args.limit)
    split = max(1, int(len(every) * 0.02))
    holdout = every.subset(torch.arange(split)).to(device)
    deals = every.subset(torch.arange(split, len(every))).to(device)
    print(f"{len(deals)} training deals, {len(holdout)} held out, device {device}")

    net, critic = PlayNet(args.width, args.depth).to(device), Critic().to(device)
    league = Pool(args.pool_size, args.width, args.depth)
    if args.init:
        # Carry on from an earlier run, and put that net in the pool as opponent zero:
        # the first thing the learner has to prove is that it can beat where it started.
        state = torch.load(args.init, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
        league.add(net)
        print(f"started from {args.init}, seeded the pool with it")
    optimiser = torch.optim.Adam(
        [{"params": net.parameters(), "lr": args.lr},
         {"params": critic.parameters(), "lr": args.lr * 3}])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, history = out_dir / "log.jsonl", []
    start = time.time()

    for step in range(1, args.steps + 1):
        learn_from = curriculum_start(step, args)
        # With --group the batch is K play-outs of batch//K deals, so the rollout costs the
        # same as before. Fewer distinct deals per step, each one scored against itself.
        picks = args.batch // args.group if args.group else args.batch
        index = torch.randint(len(deals), (picks,), generator=generator)
        contracts = deals.subset(replicate(index, args.group).to(device))
        # One opponent per step, not per deal: a second net costs a forward pass at every
        # card, so sampling it once keeps the cost flat while the mixture stays right.
        # It is therefore already constant inside a group, which is what the baseline needs.
        rival = league.sample(generator) if torch.rand(1, generator=generator) < args.league else None
        # The seat is drawn per deal, not per play-out: a group whose learner sat North-South
        # in one copy and East-West in the next would be two different players' results, and
        # their mean is a baseline for neither.
        side = replicate(torch.randint(2, (picks,), generator=generator), args.group).to(device)
        batch, steps = rollout(net, contracts, args.temperature, learn_from=learn_from,
                               warmup=args.warmup, opponent=rival, learner_side=side,
                               opponent_temperature=args.opponent_temperature)
        target = returns_for(batch, steps)                      # (n, learned steps)

        log_prob = torch.stack([s["log_prob"] for s in steps], 1)
        entropy = torch.stack([s["entropy"] for s in steps], 1)
        mask = torch.stack([s["mask"] for s in steps], 1)       # learner decisions only
        if args.group:
            # The critic is idle here, not deleted: the group is the baseline now.
            advantage, group_std = group_advantage(steps, target, args.group)
            value_loss = target.new_zeros(())
        else:
            value = critic(torch.cat([s["critic"] for s in steps], 0)).view(len(steps), -1).t()
            advantage = (target - value).detach()
            value_loss = nn.functional.mse_loss(value[mask], target[mask])
            group_std = None
        kept = advantage[mask]
        advantage = (advantage - kept.mean()) / (kept.std() + 1e-6)

        belief = torch.cat([s["belief"] for s in steps], 0)
        belief_target = torch.cat([s["belief_target"] for s in steps], 0)
        belief_mask = torch.cat([s["belief_mask"] for s in steps], 0)
        belief_mask = belief_mask & mask.t().reshape(-1, 1)     # our seats' views only
        belief_loss = nn.functional.cross_entropy(
            belief[belief_mask], belief_target[belief_mask])

        policy_loss = -(advantage * log_prob)[mask].mean()
        belief_weight = belief_weight_at(step, args)
        loss = (policy_loss + args.value_weight * value_loss
                + belief_weight * belief_loss - args.entropy_weight * entropy[mask].mean())

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        if not args.group:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimiser.step()

        if step % args.report == 0 or step == 1:
            scores = evaluate(net, holdout)
            row = {"step": step, "seconds": round(time.time() - start),
                   "policy_loss": round(policy_loss.item(), 4),
                   "value_loss": round(value_loss.item(), 4),
                   "belief_loss": round(belief_loss.item(), 4),
                   "entropy": round(entropy[mask].mean().item(), 3),
                   "league": len(league.slots), "rival": rival is not None,
                   "belief_weight": round(belief_weight, 4),
                   "learn_from": learn_from, "group": args.group,
                   "group_std": None if group_std is None else round(group_std.item(), 4), **
                   {k: round(v, 3) for k, v in scores.items()}}
            history.append(row)
            log_path.write_text("\n".join(json.dumps(r) for r in history))
            print(f"step {step:5d}  t{learn_from:2d}+  declarer {row['declarer_gap']:+.3f}  "
                  f"defence {row['defence_gap']:+.3f}  both-gap {row['trick_gap']:+.3f}  "
                  f"made {row['made']:.3f}  belief {row['belief_loss']:.3f}  "
                  f"entropy {row['entropy']:.2f}  {row['seconds']}s", flush=True)
            torch.save({"net": net.state_dict(), "config": net.config,
                        "critic": critic.state_dict()}, out_dir / "last.pt")
        if args.pool_size and step % args.pool_every == 0:
            league.add(net)
    return history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("auctions")
    parser.add_argument("--out", default="runs/play_first")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--belief-weight", type=float, default=1.0)
    parser.add_argument("--belief-final", type=float, default=None,
                        help="decay the belief weight to this; unset keeps it constant")
    parser.add_argument("--belief-decay-by", type=float, default=0.5,
                        help="fraction of the run over which the belief weight decays")
    parser.add_argument("--last-tricks", type=int, default=13,
                        help="learn only the last N tricks at first; 13 means no curriculum")
    parser.add_argument("--grow-by", type=float, default=0.6,
                        help="fraction of the run by which the window covers the whole deal")
    parser.add_argument("--warmup", choices=("net", "random"), default="net",
                        help="who plays the tricks before the learning window")
    parser.add_argument("--init", default=None,
                        help="checkpoint to continue from; it also becomes pool opponent zero")
    parser.add_argument("--pool-size", type=int, default=4,
                        help="how many past snapshots to keep as opponents; 0 is pure self play")
    parser.add_argument("--pool-every", type=int, default=2000,
                        help="take a snapshot into the pool every N steps")
    parser.add_argument("--league", type=float, default=0.5,
                        help="chance a step is played against the pool instead of itself")
    parser.add_argument("--opponent-temperature", type=float, default=1.0,
                        help="pool opponents sample at this temperature, for variety")
    parser.add_argument("--group", type=int, default=0,
                        help="play each deal K times and use the group mean as the baseline "
                             "instead of the critic; 0 keeps the critic")
    parser.add_argument("--report", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
