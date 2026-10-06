"""On-policy four-seat trajectories and greedy four-seat / silent validation.

Trajectories are grouped per (episode, side): every call of a side receives that
side's own-bid terminal return. The grouped object has the same fields as
``cooperative.actor_critic.Trajectories`` so ``trajectory_losses`` is reused
unchanged for the no-double game (its "episode" index is the dense group).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..bridge.calls import DOUBLE, PASS
from ..contract.data import TorchDeals
from ..contract.environment import AUCTION_FEATURES
from ..contract.evaluate_auction import auction_rows, run_auctions
from ..contract.prefixes import final_scores
from ..contract.targets import TARGET_SCALE, TorchScorer
from .model import SilentView, policy_log_probs
from .state import (
    MAX_CALLS,
    FourSeatBatch,
    FourSeatDoubleBatch,
    double_delta,
    own_bid_scores,
    table_ns_score,
)


@dataclass
class FourSeatTrajectories:
    states: FourSeatBatch
    actions: torch.Tensor
    episode: torch.Tensor     # dense (episode, side) group of each decision
    score: torch.Tensor       # own-bid points per group
    ceiling: torch.Tensor     # side cooperative DD ceiling per group
    terminal: FourSeatBatch
    side_score: torch.Tensor  # (episodes,2) own-bid points for both sides
    side_ceiling: torch.Tensor
    row: torch.Tensor         # episode row of each decision
    standing_x: torch.Tensor  # (episodes,) history position of the standing Double, -1 if none
    final_delta: torch.Tensor  # (episodes,) defenders' doubled-minus-undoubled result / 100
    opponent: torch.Tensor = None     # (episodes,) 0 self-play, k = frozen pool player k
    frozen_side: torch.Tensor = None  # (episodes,) side played by the frozen player, -1 none

    @property
    def returns(self) -> torch.Tensor:
        return (self.score - self.ceiling) / TARGET_SCALE

    def __len__(self) -> int:
        return len(self.actions)


def batch_class(doubles: bool):
    return FourSeatDoubleBatch if doubles else FourSeatBatch


@torch.no_grad()
def _legal(state: FourSeatBatch, hand: torch.Tensor) -> torch.Tensor:
    """``state.legal()`` with the opening rule (``competitive.OPENING_RULE``) applied."""
    from . import competitive
    return competitive.apply_opening_rule(state.legal(), hand, state.last, state.t,
                                          competitive.OPENING_RULE)


def play(actor, deals: TorchDeals, batch: FourSeatBatch, choose, chunk: int = 32768,
         on_decision=None, temperature: float = 1.0,
         double_margin: float | None = None, frozen: dict | None = None) -> FourSeatBatch:
    """Finish ``batch`` in place. ``choose(log_probs) -> actions``; silent seats Pass.

    ``log_probs`` are the legal-action log-probabilities at ``temperature``.
    ``double_margin`` (double-value nets, greedy eval only): ignore the X logit/gate;
    Double iff it is legal and ``double_value > margin`` (in /100 units), otherwise
    the best of the other 36 calls.
    ``frozen`` = {"players": {code: net}, "code": (B,), "side": (B,)}: rows whose side
    to act equals ``side`` are played greedily by ``players[code]`` (never Double),
    fed its own feature prefix (77 for cooperative nets, 147 for D5OWN4); they are
    not reported to ``on_decision``.

    ``on_decision(idx, state, chosen, masked_logits, outputs)`` sees every non-forced decision.
    """
    was_training = actor.training
    actor.eval()
    for _ in range(max(MAX_CALLS, batch.history.shape[1]) + 1):
        if bool(batch.ended.all()):
            break
        alive = ~batch.ended
        decide = alive & ~batch.forced
        action = torch.full((len(batch),), PASS, dtype=torch.long, device=batch.deal.device)
        if frozen is not None:
            frozen_rows = decide & (frozen["side"] == batch.side)
            decide = decide & ~frozen_rows
            for code, net in frozen["players"].items():
                ids = (frozen_rows & (frozen["code"] == code)).nonzero().squeeze(1)
                width = AUCTION_FEATURES + getattr(net, "extra_features", 0)
                for i in range(0, len(ids), chunk):
                    idx = ids[i:i + chunk]
                    state = batch.subset(idx)
                    hand = deals.hands[state.deal, state.actor_seat]
                    out = net(hand, state.features()[:, :width])
                    full = _legal(state, hand)
                    from .model import COMPETITIVE_STAGE, competitive_log_probs
                    if (getattr(net, "stage", None) == COMPETITIVE_STAGE
                            and getattr(actor, "stage", None) == COMPETITIVE_STAGE):
                        action[idx] = competitive_log_probs(out, full).argmax(-1)
                        continue
                    legal = full[:, :PASS + 1]
                    action[idx] = out["policy_logits"][:, :PASS + 1].masked_fill(
                        ~legal, -torch.inf).argmax(-1)
        ids = decide.nonzero().squeeze(1)
        for i in range(0, len(ids), chunk):
            idx = ids[i:i + chunk]
            state = batch.subset(idx)
            hand = deals.hands[state.deal, state.actor_seat]
            outputs = actor(hand, state.features())
            masked = policy_log_probs(outputs, _legal(state, hand), temperature)
            chosen = choose(masked)
            if double_margin is not None:
                rest = masked[:, :DOUBLE].argmax(-1)
                take = masked[:, DOUBLE].isfinite() & (outputs["double_value"] > double_margin)
                chosen = torch.where(take, torch.full_like(rest, DOUBLE), rest)
            action[idx] = chosen
            if on_decision is not None:
                on_decision(idx, state, chosen, masked, outputs)
        batch.apply(action, alive)
    if not bool(batch.ended.all()):
        raise RuntimeError("four-seat auction did not terminate")
    actor.train(was_training)
    return batch


def episode_setup(deals: TorchDeals, episodes: int, generator: torch.Generator,
                  silent_frac: float, device, pool: dict | None):
    """Seeded (deal, dealer, vul_ns, vul_ew, opponent, frozen_side, silent) per episode."""

    def rand(high: int) -> torch.Tensor:
        return torch.randint(high, (episodes,), generator=generator).to(device)

    deal, dealer, vul_ns, vul_ew = rand(deals.n), rand(4), rand(2), rand(2)
    opponent = torch.zeros_like(deal)
    if pool:
        u = torch.rand(episodes, generator=generator).to(device)
        edge = 0.0
        for code, (_, frac) in pool.items():
            opponent[(u >= edge) & (u < edge + frac)] = code
            edge += frac
        if edge > 1.0:
            raise ValueError("pool fractions exceed 1")
    frozen_side = torch.where(opponent > 0, rand(2), torch.full_like(deal, -1))
    silent_row = torch.rand(episodes, generator=generator).to(device) < silent_frac
    silent = torch.where(silent_row & (opponent == 0), rand(2), torch.full_like(deal, -1))
    return deal, dealer, vul_ns, vul_ew, opponent, frozen_side, silent


@torch.no_grad()
def collect_trajectories(actor, deals: TorchDeals, episodes: int, generator: torch.Generator,
                         scorer: TorchScorer, temperature: float = 1.0,
                         silent_frac: float = 0.25, device="cpu",
                         doubles: bool = False, pool: dict | None = None) -> FourSeatTrajectories:
    """``pool`` = {code: (frozen_net, fraction)}: that share of episodes has one random
    side played by the frozen net; ``silent_frac`` applies only inside self-play."""
    if episodes < 1 or temperature <= 0 or not 0 <= silent_frac <= 1:
        raise ValueError("invalid episodes, temperature, or silent_frac")
    device = torch.device(device)
    deal, dealer, vul_ns, vul_ew, opponent, frozen_side, silent = episode_setup(
        deals, episodes, generator, silent_frac, device, pool)
    frozen = None
    batch = batch_class(doubles).start(deal, dealer, vul_ns, vul_ew, silent)
    if pool:
        frozen = {"players": {code: net for code, (net, _) in pool.items()},
                  "code": opponent, "side": frozen_side}

    def sample(log_probs: torch.Tensor) -> torch.Tensor:
        probs = log_probs.exp()
        return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(device)

    record: list = []
    play(actor, deals, batch, sample, chunk=1 << 30, temperature=temperature, frozen=frozen,
         on_decision=lambda idx, state, chosen, *_: record.append((idx, state, chosen)))
    states = type(batch).cat([state for _, state, _ in record])
    actions = torch.cat([chosen for _, _, chosen in record])
    rows = torch.cat([idx for idx, _, _ in record])
    group = rows * 2 + states.side
    present, dense = torch.unique(group, return_inverse=True)
    score, ceiling, _ = own_bid_scores(batch, deals, scorer)
    if doubles:
        standing_x = batch.standing_double_position()
        final_delta = double_delta(batch, deals)
    else:
        standing_x = torch.full_like(deal, -1)
        final_delta = torch.zeros(episodes, device=device)
    return FourSeatTrajectories(states, actions, dense, score.reshape(-1)[present],
                                ceiling.reshape(-1)[present], batch, score, ceiling, rows,
                                standing_x, final_delta, opponent, frozen_side)


def greedy(masked: torch.Tensor) -> torch.Tensor:
    return masked.argmax(-1)


def fourseat_rows(n_deals: int, device="cpu"):
    """(deal, dealer, vul_ns, vul_ew): deal x 4 dealers x 4 vulnerability combos."""
    deal = torch.arange(n_deals, device=device).repeat_interleave(16)
    dealer = torch.arange(4, device=device).repeat_interleave(4).repeat(n_deals)
    vul = torch.arange(4, device=device).repeat(4 * n_deals)
    return deal, dealer, (vul == 1) | (vul == 3), (vul == 2) | (vul == 3)


@torch.no_grad()
def fourseat_validation(actor, deals: TorchDeals, scorer: TorchScorer,
                        doubles: bool = False, double_margin: float | None = None,
                        frozen_net=None) -> dict:
    """Greedy four-seat self-play (same net at every seat), own-bid score per side.

    With ``doubles`` also: double rate over final contracts, accuracy (doubled
    contracts that go down), mean delta per double in points, mean p(X) over
    final defending decisions (undoubled opponents' contract, Pass would end),
    and ``objective`` = per-side own-bid score plus the standing doubler's delta.
    """
    final_px = {"sum": 0.0, "count": 0}
    calib = {"q": [], "delta": [], "final": [], "doubled": []}

    def watch(idx, state, chosen, masked, outputs):
        if not doubles:
            return
        can = state.can_double()
        eligible = can & (state.pass_count == 2)
        if bool(eligible.any()):
            final_px["sum"] += float(masked[eligible, DOUBLE].exp().sum())
            final_px["count"] += int(eligible.sum())
        if "double_value" in outputs and bool(can.any()):
            rows = can.nonzero().squeeze(1)
            calib["q"].append(outputs["double_value"][rows])
            calib["delta"].append(double_delta(state.subset(rows), deals))
            calib["final"].append(eligible[rows])
            calib["doubled"].append(chosen[rows] == DOUBLE)

    start = batch_class(doubles).start(*fourseat_rows(deals.n, deals.hands.device))
    frozen = None
    if frozen_net is not None:
        # frozen side alternates with (deal + dealer): every vulnerability gets both seatings
        side = (start.deal + start.dealer) % 2
        frozen = {"players": {1: frozen_net}, "code": torch.ones_like(side), "side": side}
    batch = play(actor, deals, start, greedy, on_decision=watch, double_margin=double_margin,
                 frozen=frozen)
    if frozen is not None:
        return pool_metrics(batch, deals, scorer, frozen["side"])
    score, ceiling, table_ns = own_bid_scores(batch, deals, scorer)
    seats = batch.bid_seats()
    bid_ns = ((seats >= 0) & (seats % 2 == 0)).any(1)
    bid_ew = ((seats >= 0) & (seats % 2 == 1)).any(1)
    level = torch.where(batch.last >= 0, batch.last // 5 + 1, torch.zeros_like(batch.last))
    out = {
        "own_score": float(score.mean()),
        "own_ns": float(score[:, 0].mean()),
        "own_ew": float(score[:, 1].mean()),
        "own_regret": float((ceiling - score).mean()),
        "table_ns_undoubled": float(table_ns.mean()),
        "passout": float((batch.last < 0).float().mean()),
        "both_sides_bid": float((bid_ns & bid_ew).float().mean()),
        "calls": float(batch.t.float().mean()),
        "level_share": {str(lv): float((level == lv).float().mean()) for lv in range(8)},
    }
    out["objective"] = out["own_score"]
    if doubles:
        contracts = batch.last >= 0
        doubled = batch.doubled & contracts
        delta = double_delta(batch, deals) * TARGET_SCALE
        n_dbl = int(doubled.sum())
        out.update({
            "double_rate": float(doubled.sum() / contracts.sum().clamp(min=1)),
            "double_accuracy": float((delta[doubled] > 0).float().mean()) if n_dbl else 0.0,
            "delta_per_double": float(delta[doubled].mean()) if n_dbl else 0.0,
            "doubles": n_dbl,
            "p_double_final": final_px["sum"] / max(final_px["count"], 1),
            "objective": float((score.sum(1) + torch.where(doubled, delta, 0 * delta)).mean() / 2),
        })
        if calib["q"]:
            out["double_value"] = double_value_calibration(
                *(torch.cat(calib[k]) for k in ("q", "delta", "final", "doubled")))
    return out


@torch.no_grad()
def pool_metrics(batch: FourSeatDoubleBatch, deals: TorchDeals, scorer: TorchScorer,
                 frozen_side: torch.Tensor) -> dict:
    """Learner-side metrics against a frozen opponent (learner side = 1 - frozen_side)."""
    rows = torch.arange(len(batch), device=batch.deal.device)
    learner = 1 - frozen_side
    score, _, _ = own_bid_scores(batch, deals, scorer)
    table = table_ns_score(batch, deals)
    learner_table = torch.where(learner == 0, table, -table)
    seats = batch.bid_seats()
    owner = seats[rows, batch.last.clamp(min=0)] % 2
    opp_contract = (batch.last >= 0) & (owner == frozen_side)
    doubled = batch.doubled & opp_contract
    delta = double_delta(batch, deals) * TARGET_SCALE
    n_dbl = int(doubled.sum())
    return {
        "learner_own_score": float(score[rows, learner].mean()),
        "learner_table_score": float(learner_table.mean()),
        "learner_declares": float(((batch.last >= 0) & (owner == learner)).float().mean()),
        "double_rate": float(doubled.sum() / opp_contract.sum().clamp(min=1)),
        "double_accuracy": float((delta[doubled] > 0).float().mean()) if n_dbl else 0.0,
        "delta_per_double": float(delta[doubled].mean()) if n_dbl else 0.0,
        "doubles": n_dbl,
        "calls": float(batch.t.float().mean()),
    }


def double_value_calibration(q, delta, final, doubled, bins: int = 10) -> dict:
    """Predicted vs exact delta (/100) by predicted decile, at every legal-X greedy spot."""
    out = {"spots": int(len(q)), "final_spots": int(final.sum()),
           "mse": float((q - delta).pow(2).mean()),
           "profitable_share": float((delta > 0).float().mean()),
           "greedy_x_share": float(doubled.float().mean()),
           "greedy_x_accuracy": float((delta[doubled] > 0).float().mean()) if bool(doubled.any())
           else 0.0,
           "q_positive_share": float((q > 0).float().mean()),
           "q_positive_accuracy": float((delta[q > 0] > 0).float().mean()) if bool((q > 0).any())
           else 0.0}
    order = q.argsort()
    out["deciles"] = [
        {"pred": float(q[chunk].mean()), "real": float(delta[chunk].mean()), "n": int(len(chunk))}
        for chunk in order.tensor_split(bins) if len(chunk)]
    if bool(final.any()):
        out["final_mse"] = float((q[final] - delta[final]).pow(2).mean())
    return out


@torch.no_grad()
def silent_validation(actor, deals: TorchDeals, scorer: TorchScorer) -> dict:
    """Greedy silent-opponent auctions on ``auction_rows`` via the cooperative evaluator."""
    rows = auction_rows(deals.n, deals.hands.device)
    auctions = run_auctions(SilentView(actor), deals, rows, scorer, "policy")
    score, ceiling, _ = final_scores(auctions, deals, scorer)
    return {"score": float(score.mean()), "regret": float((ceiling - score).mean()),
            "passout": float((auctions.last < 0).float().mean()),
            "decisions": float(auctions.k.float().mean())}


def trajectory_stats(trajectories: FourSeatTrajectories) -> dict:
    terminal = trajectories.terminal
    free = terminal.silent < 0
    seats = terminal.bid_seats()
    both = (((seats >= 0) & (seats % 2 == 0)).any(1) & ((seats >= 0) & (seats % 2 == 1)).any(1))
    out = {"decisions": len(trajectories), "groups": len(trajectories.score),
           "calls": float(terminal.t.float().mean()),
           "silent_episodes": int((~free).sum()),
           "both_sides_bid_free": float(both[free].float().mean()) if bool(free.any()) else 0.0,
           "own_score_train": float(trajectories.score.float().mean())}
    if terminal.doubles:
        out["train_double_calls"] = int((trajectories.actions == DOUBLE).sum())
        out["train_standing_doubles"] = int((trajectories.standing_x >= 0).sum())
    if trajectories.opponent is not None:
        for code in trajectories.opponent.unique().tolist():
            rows = trajectories.opponent == code
            out[f"train_episodes_opp{code}"] = int(rows.sum())
            out[f"train_standing_doubles_opp{code}"] = int((trajectories.standing_x[rows] >= 0).sum())
    return out
