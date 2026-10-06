"""Redouble and the Sacrifice gate for the four-seat trainer (stage D5OWN4XC).

Mirrors the D5OWN4XD Double gate: every new head reads ``hidden.detach()``, learns an
exact "act now, then everyone passes" DD counterfactual, and its gate is trained by
BCE toward ``sigmoid(value / tau)``.

"Last bid only": X, XX and SAC are legal only where the actor's Pass would end the
auction (a standing contract followed by two passes). The direct-seat player never
doubles. "Only instead of Pass": they take probability only from the softmax36 Pass
entry: p(X or XX) = p_pass p_D, p(SAC) = p_pass (1-p_D) p_SAC, p(Pass) = p_pass
(1-p_D)(1-p_SAC); every bid keeps its softmax mass (the SAC candidate adds p(SAC)).

Redouble (action 37): legal for the declaring side over a standing X at
its pass-out seat; any later bid clears X/XX. Credit: a standing XX gets the declaring
side's (redoubled - doubled)/100; the standing X gets the defenders' (result at the final
doubling level - undoubled)/100, so X learns the risk of being redoubled. The bidder's
own-bid return stays undoubled.

Sacrifice: where the opponents hold the standing contract and Pass
would end the auction, candidates are the cheapest bid in each strain above their
contract and at level 4+. The SAC bids the candidate with the highest ``sac_value``
(ties -> cheapest). A fired SAC gets (own side's final real table result - own side's
result of the contract it bid over, at its doubling level)/100 on its gate; the
sacrifice bid is excluded from the side's own-bid score, and the decision is left out
of the Q loss.

PG: the trunk's log softmax36(call, or Pass under a gate path) gets the undoubled
own-bid advantage; the gate log-probs get the credited advantage and are detached by
default (``--gate-pg`` trains them). X/XX/SAC value and gate losses only use "clean"
spots where every earlier opponent call was that player's greedy top call
(``--all-spots`` disables this).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from ..bridge.calls import DOUBLE, PASS, REDOUBLE
from ..contract.data import TorchDeals
from ..contract.targets import TARGET_SCALE, TorchScorer
from ..simplicity import LIGHT_HCP
from .model import (
    COMPETITIVE_STAGE,
    SAC_STRAINS,
    competitive_log_probs,
    competitive_parts,
    competitive_path_log_probs,
    sac_candidates_from_last,
)
from .rollout import (
    FourSeatTrajectories,
    double_value_calibration,
    episode_setup,
    fourseat_rows,
    greedy,
    play,
    pool_metrics,
)
from ..simplicity import analyse_batch
from .state import (
    _DOUBLED,
    _REDOUBLED,
    _TABLE_STRAIN,
    _UNDOUBLED,
    FourSeatBatch,
    FourSeatDoubleBatch,
    contract_declarer,
    double_delta,
    features_from_history,
    own_bid_scores,
    side_table_scores,
    table_ns_score,
)

# Generous history width for auctions with X and XX (the no-XX bound is 216).
MAX_REDOUBLE_CALLS = 320


def competitive_features(history: torch.Tensor, dealer: torch.Tensor, vul_ns: torch.Tensor,
                         vul_ew: torch.Tensor, actor: torch.Tensor) -> torch.Tensor:
    """Double features ``(B,149)`` + ``[149]`` Pass would end the auction + ``[150]`` redoubled."""
    feats = features_from_history(history, dealer, vul_ns, vul_ew, actor, doubles=True)
    extra = torch.zeros(len(history), 2, device=history.device)
    if history.shape[1]:
        pos = torch.arange(history.shape[1], device=history.device)[None]
        none = torch.full_like(history, -1)
        valid = history >= 0
        last_bid = torch.where(valid & (history < PASS), pos, none).max(1).values
        last_active = torch.where(valid & (history != PASS), pos, none).max(1).values
        last_xx = torch.where(history == REDOUBLE, pos, none).max(1).values
        trailing = valid.sum(1) - 1 - last_active
        extra[:, 0] = ((last_bid >= 0) & (trailing == 2)).float()
        extra[:, 1] = (last_xx > last_bid).float()
    return torch.cat((feats, extra), 1)


# With ANY_SEAT_DOUBLE, X and XX are legal at every seat for every player and the auction
# continues after them (escapes and redoubles are real). SAC stays at the pass-out seat.
ANY_SEAT_DOUBLE = False


def set_any_seat_double(on: bool) -> None:
    global ANY_SEAT_DOUBLE
    ANY_SEAT_DOUBLE = bool(on)


# Opening rule (rule of 18): in 1st and 2nd seat a 1-level opening (1C..1NT) needs HCP +
# the lengths of the two longest suits >= OPENING_RULE. After two passes any opening is
# legal. 0 = off. A hard legality rule for every net at the table, like a law of the game.
OPENING_RULE = 0
ONE_LEVEL = 5                                   # actions 0..4 = 1C 1D 1H 1S 1NT


def set_opening_rule(points: int) -> None:
    global OPENING_RULE
    OPENING_RULE = int(points)


def opening_points(hands: torch.Tensor) -> torch.Tensor:
    """HCP + lengths of the two longest suits; ``hands`` ``(..., 52)``, suit * 13 + rank (A first)."""
    cards = hands.float().reshape(*hands.shape[:-1], 4, 13)
    hcp = (cards[..., :4] * torch.tensor([4.0, 3.0, 2.0, 1.0], device=hands.device)).sum((-1, -2))
    return hcp + cards.sum(-1).topk(2, -1).values.sum(-1)


def opening_blocked(hands: torch.Tensor, last: torch.Tensor, t, rule: int) -> torch.Tensor:
    """``(B,)`` rows whose 1-level bids are illegal: nobody has bid yet, fewer than two
    passes so far (1st/2nd seat), and the hand has fewer than ``rule`` opening points."""
    if not rule:
        return torch.zeros_like(last, dtype=torch.bool)
    return (last < 0) & (torch.as_tensor(t, device=last.device) < 2) & (opening_points(hands) < rule)


def apply_opening_rule(legal: torch.Tensor, hands, last, t, rule: int) -> torch.Tensor:
    """``legal`` with the 1-level bids removed where ``opening_blocked``."""
    if not rule:
        return legal
    block = opening_blocked(hands, last, t, rule)
    cols = torch.arange(legal.shape[1], device=legal.device) < ONE_LEVEL
    return legal & ~(block[:, None] & cols[None])


@dataclass
class FourSeatFinalDoubleBatch(FourSeatDoubleBatch):
    """Four-seat auction where Double is legal only at the pass-out seat (Pass would end it)."""

    redoubled: torch.Tensor = None       # standing contract redoubled (never without action 37)

    @classmethod
    def start(cls, deal, dealer, vul_ns, vul_ew, silent=None):
        base = FourSeatDoubleBatch.start(deal, dealer, vul_ns, vul_ew, silent)
        values = {f.name: getattr(base, f.name) for f in fields(FourSeatDoubleBatch)}
        values["history"] = torch.full((len(deal), MAX_REDOUBLE_CALLS), -1, dtype=torch.long,
                                       device=deal.device)
        return cls(**values, redoubled=torch.zeros_like(base.ended))

    def pass_ends(self) -> torch.Tensor:
        return (self.last >= 0) & (self.pass_count == 2)

    def can_double(self) -> torch.Tensor:
        if ANY_SEAT_DOUBLE:
            return super().can_double()
        return super().can_double() & self.pass_ends()

    def _after_call(self, r, a, seat) -> None:
        super()._after_call(r, a, seat)
        self.redoubled[r[a < PASS]] = False
        xx = r[a == REDOUBLE]
        self.redoubled[xx] = True
        self.pass_count[xx] = 0

    def features(self) -> torch.Tensor:
        length = int(self.t.max()) if len(self) else 0
        return competitive_features(self.history[:, :length], self.dealer, self.vul[:, 0],
                                    self.vul[:, 1], self.actor_seat)

    def standing_redouble_position(self) -> torch.Tensor:
        pos = torch.arange(self.history.shape[1], device=self.history.device)[None]
        none = torch.full_like(self.history, -1)
        last_bid = torch.where((self.history >= 0) & (self.history < PASS), pos, none).max(1).values
        last_xx = torch.where(self.history == REDOUBLE, pos, none).max(1).values
        return torch.where(last_xx > last_bid, last_xx, torch.full_like(last_xx, -1))


@dataclass
class FourSeatRedoubleBatch(FourSeatFinalDoubleBatch):
    """Adds Redouble (37): the declaring side over a standing X, at its pass-out seat."""

    n_actions = REDOUBLE + 1
    redoubles = True

    def can_redouble(self) -> torch.Tensor:
        return (~self.ended & ~self.forced & (self.pass_ends() | ANY_SEAT_DOUBLE) & self.doubled & ~self.redoubled
                & (self.contract_seat % 2 == self.side))

    def legal(self) -> torch.Tensor:
        return torch.cat([super().legal(), self.can_redouble()[:, None]], 1)


def competitive_batch_class(redouble: bool):
    return FourSeatRedoubleBatch if redouble else FourSeatFinalDoubleBatch


# ---------------------------------------------------------------- exact targets

def doubling_level(batch) -> torch.Tensor:
    """0 undoubled, 1 doubled, 2 redoubled."""
    level = batch.doubled.long()
    redoubled = getattr(batch, "redoubled", None)
    return level if redoubled is None else level + redoubled.long()


def standing_scores(batch, deals: TorchDeals):
    """Declarer seat and ``(B,3)`` declarer scores of the standing contract at X levels 0/1/2."""
    device = batch.deal.device
    rows = torch.arange(len(batch), device=device)
    declarer = contract_declarer(batch)
    c = batch.last.clamp(min=0)
    tricks = deals.tricks[batch.deal, declarer.clamp(min=0), _TABLE_STRAIN.to(device)[c]].long()
    vul = batch.vul[rows, declarer.clamp(min=0) % 2].long()
    scores = torch.stack([table.to(device)[vul, c, tricks]
                          for table in (_UNDOUBLED, _DOUBLED, _REDOUBLED)], 1)
    return declarer, scores


def redouble_delta(batch, deals: TorchDeals) -> torch.Tensor:
    """Declaring side's (redoubled - doubled) result of the standing contract / 100; 0 if none."""
    _, scores = standing_scores(batch, deals)
    delta = (scores[:, 2] - scores[:, 1]) / TARGET_SCALE
    return torch.where(batch.last >= 0, delta, torch.zeros_like(delta))


def final_double_delta(batch, deals: TorchDeals) -> torch.Tensor:
    """Defenders' (result at the standing doubling level - undoubled) / 100; 0 if undoubled."""
    _, scores = standing_scores(batch, deals)
    rows = torch.arange(len(batch), device=batch.deal.device)
    level = doubling_level(batch)
    delta = (scores[:, 0] - scores[rows, level]) / TARGET_SCALE
    return torch.where((batch.last >= 0) & (level > 0), delta, torch.zeros_like(delta))


def sac_targets(batch, deals: TorchDeals, legal: torch.Tensor | None = None):
    """SAC candidates ``(B,5)``, legal mask ``(B,5)``, exact delta ``(B,5)`` and baseline ``(B,)``.

    Spots: opponents hold the standing contract and Pass would end the auction. Candidates:
    cheapest bid per strain above their contract and at level 4+.
    delta = [own side's doubled result of the candidate, then all pass
             - own side's result of the opponents' contract at its doubling level] / 100.
    The candidate's declarer is the first player of the actor's side to have named that
    strain, else the actor. ``baseline`` is the second term in points (0 off spots).
    """
    device = batch.deal.device
    legal = batch.legal() if legal is None else legal
    rows = torch.arange(len(batch), device=device)
    side = batch.side
    opp_holds = (batch.last >= 0) & (batch.contract_seat % 2 != side)
    spot = opp_holds & (batch.pass_count == 2)
    cand = sac_candidates_from_last(batch.last, spot)
    valid = (cand >= 0) & legal[rows[:, None], cand.clamp(min=0)]
    _, scores = standing_scores(batch, deals)
    baseline = -scores[rows, doubling_level(batch)]
    baseline = torch.where(opp_holds, baseline, torch.zeros_like(baseline))
    seats = batch.bid_seats()
    ladder = torch.arange(35, device=device)[None]
    ours = (seats >= 0) & (seats % 2 == side[:, None])
    vul = batch.vul[rows, side].long()
    delta = torch.zeros(len(batch), SAC_STRAINS, device=device)
    for k in range(SAC_STRAINS):
        named = ours & (ladder % 5 == k)
        first = torch.where(named, ladder, torch.full_like(seats, 35)).min(1).values
        declarer = torch.where(first < 35, seats[rows, first.clamp(max=34)], batch.actor_seat)
        c = cand[:, k].clamp(min=0)
        tricks = deals.tricks[batch.deal, declarer, _TABLE_STRAIN.to(device)[c]].long()
        score = _DOUBLED.to(device)[vul, c, tricks]
        delta[:, k] = torch.where(valid[:, k], (score - baseline) / TARGET_SCALE,
                                  torch.zeros_like(score))
    return cand, valid, delta, baseline


def side_table_score(batch, deals: TorchDeals, side: torch.Tensor) -> torch.Tensor:
    """Real table score (doubling level included) from ``side``'s point of view."""
    table = table_ns_score(batch, deals)
    return torch.where(side == 0, table, -table)


def greedy_sac_fired(outputs: dict, legal: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Greedy play: the chosen call is the SAC candidate and the SAC path carries more mass."""
    part = competitive_parts(outputs, legal)
    rows = torch.arange(len(chosen), device=chosen.device)
    hit = part["spot"] & (chosen == part["chosen"])
    return hit & (part["sac_lp"] >= part["rest"][rows, chosen.clamp(max=PASS)])


# ---------------------------------------------------------------- trajectories

@dataclass
class CompetitiveTrajectories(FourSeatTrajectories):
    sac_fired: torch.Tensor = None     # (decisions,) the SAC gate chose this call
    sac_credit: torch.Tensor = None    # (decisions,) /100, 0 unless fired
    clean: torch.Tensor = None         # (decisions,) every earlier opponent call was greedy-top
    standing_xx: torch.Tensor = None   # (episodes,) history position of the standing XX, -1 none
    xx_delta: torch.Tensor = None      # (episodes,) declaring side's (XX - X) / 100 if XX stands
    table_score: torch.Tensor = None   # per group: real table points for that side (doubling included)


@torch.no_grad()
def collect_competitive_trajectories(actor, deals: TorchDeals, episodes: int,
                                     generator: torch.Generator, scorer: TorchScorer,
                                     temperature: float = 1.0, silent_frac: float = 0.25,
                                     device="cpu", pool: dict | None = None
                                     ) -> CompetitiveTrajectories:
    """``collect_trajectories`` for D5OWN4XC nets (same episode mix and pool semantics).

    Calls are sampled from the marginal policy; when the sampled call is the SAC
    candidate, whether the SAC gate fired is drawn from its exact posterior
    ``p(SAC path) / (softmax36(call) + p(SAC path))``, so (gate, call) is a joint sample.
    """
    if episodes < 1 or temperature <= 0 or not 0 <= silent_frac <= 1:
        raise ValueError("invalid episodes, temperature, or silent_frac")
    device = torch.device(device)
    deal, dealer, vul_ns, vul_ew, opponent, frozen_side, silent = episode_setup(
        deals, episodes, generator, silent_frac, device, pool)
    frozen = None
    batch = competitive_batch_class(actor.redouble).start(deal, dealer, vul_ns, vul_ew, silent)
    if pool:
        frozen = {"players": {code: net for code, (net, _) in pool.items()},
                  "code": opponent, "side": frozen_side}
    top = torch.ones(episodes, batch.history.shape[1], dtype=torch.bool, device=device)

    def sample(log_probs: torch.Tensor) -> torch.Tensor:
        probs = log_probs.exp()
        return torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(device)

    record: list = []

    def on_decision(idx, state, chosen, masked, outputs):
        top[idx, state.t] = chosen == masked.argmax(-1)
        fired = torch.zeros_like(chosen, dtype=torch.bool)
        if "sac_gate" in outputs:
            part = competitive_parts(outputs, state.legal(), temperature)
            hit = part["spot"] & (chosen == part["chosen"])
            if bool(hit.any()):
                rows = torch.arange(len(chosen), device=device)
                soft = part["rest"][rows, chosen.clamp(max=PASS)]
                fire = part["sac_lp"] - torch.logaddexp(soft, part["sac_lp"])
                u = torch.rand(len(chosen), generator=generator).to(device)
                fired = hit & (u < fire.exp())
        record.append((idx, state, chosen, fired))

    play(actor, deals, batch, sample, chunk=1 << 30, temperature=temperature, frozen=frozen,
         on_decision=on_decision)
    states = type(batch).cat([state for _, state, _, _ in record])
    actions = torch.cat([chosen for _, _, chosen, _ in record])
    rows = torch.cat([idx for idx, _, _, _ in record])
    fired = torch.cat([f for _, _, _, f in record])
    return competitive_trajectories(deals, scorer, batch, states, actions, rows, fired, top,
                                    opponent, frozen_side)


def competitive_trajectories(deals: TorchDeals, scorer: TorchScorer, batch, states, actions,
                             rows, fired, top, opponent, frozen_side) -> CompetitiveTrajectories:
    """Trajectories from a finished ``batch`` and its recorded decisions (round order).

    ``top`` ``(episodes, history)``: False where the call at that position was a sampled
    non-greedy call (True elsewhere); ``fired`` marks decisions where the SAC gate chose.
    Shared by the default and the fast (``fast_rollout``) collectors.
    """
    device = batch.deal.device
    episodes = len(batch)
    deal = batch.deal
    group = rows * 2 + states.side
    present, dense = torch.unique(group, return_inverse=True)

    pos = torch.arange(top.shape[1], device=device)[None]
    opp_call = (states.dealer[:, None] + pos) % 2 != states.side[:, None]
    clean = ~(~top[rows] & opp_call & (pos < states.t[:, None])).any(1)

    exclude = torch.zeros(episodes, 35, dtype=torch.bool, device=device)
    exclude[rows[fired], actions[fired]] = True
    score, ceiling, _ = own_bid_scores(batch, deals, scorer, exclude=exclude)
    sac_credit = torch.zeros(len(actions), device=device)
    if bool(fired.any()):
        f_idx = fired.nonzero().squeeze(1)
        _, _, _, baseline = sac_targets(states.subset(f_idx), deals)
        final = side_table_score(batch, deals, torch.zeros_like(deal))[rows[f_idx]]
        final = torch.where(states.side[f_idx] == 0, final, -final)
        sac_credit[f_idx] = (final - baseline) / TARGET_SCALE
    standing_xx = batch.standing_redouble_position()
    xx_delta = torch.where(batch.redoubled, redouble_delta(batch, deals),
                           torch.zeros(episodes, device=device))
    table = side_table_scores(batch, deals)
    return CompetitiveTrajectories(
        states, actions, dense, score.reshape(-1)[present], ceiling.reshape(-1)[present], batch,
        score, ceiling, rows, batch.standing_double_position(), final_double_delta(batch, deals),
        opponent, frozen_side, sac_fired=fired, sac_credit=sac_credit, clean=clean,
        standing_xx=standing_xx, xx_delta=xx_delta, table_score=table.reshape(-1)[present])


# ---------------------------------------------------------------- losses

def code_word_parts(states, actions: torch.Tensor, deals: TorchDeals) -> dict:
    """Per-decision masks behind ``code_word_mask``, plus cue bids, jumps and 4NT.

    Same rule as ``tools/simplicity.py``: a suit bid without 4+ cards in the suit (3+ to
    raise a suit partner bid), a double of a contract at level 3 or below, a redouble, or
    a 2♣ opening with fewer than 5 clubs or 20+ HCP.
    """
    seat = states.actor_seat
    hand = deals.hands[states.deal, seat].reshape(-1, 4, 13)       # suits S H D C, ranks A..2
    lengths = hand.sum(2)
    hcp = (hand[:, :, :4] * torch.tensor([4, 3, 2, 1], device=hand.device)).sum((1, 2))
    history = states.history
    pos = torch.arange(history.shape[1], device=history.device)[None]
    made = (pos < states.t[:, None]) & (history >= 0)
    bid = made & (history < PASS)
    is_bid = actions < PASS
    strain = actions.clamp(max=PASS - 1) % 5                       # C D H S NT
    suit = is_bid & (strain < 4)
    held = lengths.gather(1, (3 - strain.clamp(max=3))[:, None]).squeeze(1)
    bidder = (states.dealer[:, None] + pos) % 4
    same_strain = bid & (history % 5 == strain[:, None])
    partner_bid = (same_strain & (bidder == ((seat + 2) % 4)[:, None])).any(1)
    own_side_bid = (same_strain & (bidder % 2 == (seat % 2)[:, None])).any(1)
    last_bid = torch.where(bid, history, torch.full_like(history, -1)).max(1).values
    level = torch.where(last_bid >= 0, last_bid // 5 + 1, torch.zeros_like(last_bid))
    min_level = torch.where(strain > last_bid % 5, level.clamp(min=1), level + 1)
    min_level = torch.where(last_bid >= 0, min_level, torch.ones_like(level))
    return {
        "suit_bid": suit,
        "unnatural": suit & ~((held >= 4) | (partner_bid & (held >= 3))),
        "low_double": (actions == DOUBLE) & (level <= 3),
        "redouble": actions == REDOUBLE,
        "strong_2c": (actions == 5) & ~bid.any(1) & ((lengths[:, 3] < 5) | (hcp >= 20)),
        "cue": suit & same_strain.any(1) & ~own_side_bid,
        "bid": is_bid,
        "jump": is_bid & (actions // 5 + 1 > min_level),
        "four_nt": actions == 19,
        # an opening (nobody has bid yet) on LIGHT_HCP or fewer, except a natural preempt
        # (2-level or higher in a 6+ card suit): the destructive openings code words miss
        "light_open": is_bid & ~bid.any(1) & (hcp <= LIGHT_HCP)
                      & ~(suit & (actions >= 5) & (held >= 6)),
    }


def code_word_mask(states, actions: torch.Tensor, deals: TorchDeals) -> torch.Tensor:
    """``(decisions,)`` True where the call is a code word partner cannot read at face value."""
    return code_words_of(code_word_parts(states, actions, deals))


def code_words_of(part: dict) -> torch.Tensor:
    """The code-word mask from ``code_word_parts`` output."""
    return part["unnatural"] | part["low_double"] | part["redouble"] | part["strong_2c"]


def _gate_losses(extra: dict, name: str, value, target, gap, can, use, tau: float,
                 policy_cf: bool) -> None:
    """Value MSE + gate BCE toward sigmoid(value/tau) on ``use``; diagnostics on ``can``."""
    zero = gap.sum() * 0.0
    extra[f"{name}_value_loss"] = F.mse_loss(value[use], target[use]) if bool(use.any()) else zero
    extra[f"{name}_ce"] = (F.binary_cross_entropy_with_logits(
        gap[use], torch.sigmoid(value.detach() / tau)[use]) if bool(use.any()) and policy_cf else zero)
    if bool(can.any()):
        extra[f"{name}_value_mse"] = F.mse_loss(value[can], target[can]).detach()
        extra[f"{name}_profitable_share"] = (target[can] > 0).float().mean()
        extra[f"{name}_p"] = torch.sigmoid(gap[can]).mean().detach()
    extra[f"{name}_decisions"] = can.sum().float()
    extra[f"{name}_clean_decisions"] = use.sum().float()


def competitive_trajectory_losses(actor, critic, deals: TorchDeals, traj: CompetitiveTrajectories,
                                  entropy_weight: float = 0.01, policy_temperature: float = 1.0,
                                  double_tau: float = 0.5, xx_tau: float = 0.1,
                                  sac_tau: float = 0.5, policy_cf: bool = True,
                                  gate_pg: bool = False, all_spots: bool = False,
                                  table_weight: float = 0.0,
                                  code_word_penalty: float = 0.0,
                                  light_open_penalty: float = 0.0) -> dict:
    """PG (trunk: own-bid advantage; gates: credited advantage) + X/XX/SAC value and gate losses.

    ``table_weight`` (lambda in [0, 1]) mixes the real table result into the team return:
    ``((1 - lambda) * own-bid + lambda * table - ceiling) / 100``, where table counts the
    opponents' contracts and doubled/redoubled results for this side. The X/XX/SAC credits
    are scaled by ``1 - lambda`` because the table term already contains their outcomes.
    The critic and ``contract_q`` regress the mixed return.

    ``code_word_penalty`` (/100 points) is subtracted from the policy advantage of every
    call ``code_word_mask`` flags, charged to that call alone (not to partner's calls,
    the critic, or ``contract_q``). It trades a little score for a more natural system.
    ``light_open_penalty`` works the same way on ``code_word_parts``' "light_open" calls.
    """
    if not 0.0 <= table_weight <= 1.0:
        raise ValueError("table_weight must be in [0, 1]")
    if table_weight and traj.table_score is None:
        raise ValueError("table_weight needs trajectories with table scores")
    if policy_temperature <= 0:
        raise ValueError("policy temperature must be positive")
    T = policy_temperature
    states = traj.states
    feats = states.features()
    seat = states.actor_seat
    hand = deals.hands[states.deal, seat]
    out = actor(hand, feats)
    legal = apply_opening_rule(states.legal(), hand, states.last, states.t, OPENING_RULE)
    detach = not gate_pg
    log_policy = competitive_log_probs(out, legal, T, detach_gates=detach)
    policy = log_policy.exp()
    entropy = -(policy * log_policy.masked_fill(~legal, 0.0)).sum(-1)
    trunk_logp, gate_logp = competitive_path_log_probs(out, legal, traj.actions, traj.sac_fired,
                                                       T, detach_gates=detach)
    row = torch.arange(len(states), device=hand.device)

    pair = torch.stack((hand, deals.hands[states.deal, (seat + 2) % 4]), dim=1)
    value = critic(pair, feats)
    if table_weight:
        group_returns = ((1.0 - table_weight) * traj.score + table_weight * traj.table_score
                         - traj.ceiling) / TARGET_SCALE
    else:
        group_returns = traj.returns
    returns = group_returns[traj.episode]
    zeros = torch.zeros_like(returns)
    fired = traj.sac_fired
    stand_x = (traj.actions == DOUBLE) & (states.t == traj.standing_x[traj.row])
    stand_xx = (traj.actions == REDOUBLE) & (states.t == traj.standing_xx[traj.row])
    keep = 1.0 - table_weight
    credit_x = torch.where(stand_x, keep * traj.final_delta[traj.row], zeros)
    credit_xx = torch.where(stand_xx, keep * traj.xx_delta[traj.row], zeros)
    credit_sac = torch.where(fired, keep * traj.sac_credit, zeros)
    action_return = returns + credit_x + credit_xx + credit_sac
    base_advantage = returns - value.detach()
    advantage = action_return - value.detach()
    parts = code_word_parts(states, traj.actions, deals)
    code_words = code_words_of(parts)
    light_open = parts["light_open"]
    cost = code_word_penalty * code_words + light_open_penalty * light_open
    base_advantage = base_advantage - cost
    advantage = advantage - cost
    groups = len(traj.score)
    policy_by_group = torch.zeros(groups, device=hand.device).scatter_add_(
        0, traj.episode, trunk_logp * base_advantage + gate_logp * advantage)
    entropy_by_group = torch.zeros(groups, device=hand.device).scatter_add_(
        0, traj.episode, entropy)
    q_rows = ~fired
    q_loss = F.mse_loss(out["contract_q"][row, traj.actions][q_rows], action_return[q_rows])

    zero = out["policy_logits"].sum() * 0.0
    usable = torch.ones_like(fired) if all_spots else traj.clean
    extra: dict = {"clean_share": traj.clean.float().mean()}

    can = legal[:, DOUBLE]
    _gate_losses(extra, "double", out["double_value"], double_delta(states, deals),
                 out["double_gate"] / T, can, can & usable, double_tau, policy_cf)

    if "redouble_gate" in out:
        can_xx = legal[:, REDOUBLE]
        _gate_losses(extra, "redouble", out["redouble_value"], redouble_delta(states, deals),
                     out["redouble_gate"] / T, can_xx, can_xx & usable, xx_tau, policy_cf)
        extra["redouble_calls"] = (traj.actions == REDOUBLE).sum().float()
        extra["redouble_credit_mean"] = credit_xx[stand_xx].mean() if bool(stand_xx.any()) else zero

    if "sac_gate" in out:
        extra.update(sac_value_loss=zero, sac_ce=zero)
        cand, valid, delta, _ = sac_targets(states, deals, legal)
        if not torch.equal(torch.where(valid, cand, -1), torch.where(valid, out["sac_candidates"], -1)):
            raise AssertionError("model SAC candidates disagree with the auction state")
        spot = valid.any(1)
        use = spot & usable
        v = out["sac_value"]
        gap = out["sac_gate"] / T
        if bool(use.any()):
            mask = valid & use[:, None]
            extra["sac_value_loss"] = (v - delta).pow(2)[mask].mean()
            if policy_cf:
                best = v.detach().masked_fill(~valid, -torch.inf).max(1).values
                extra["sac_ce"] = F.binary_cross_entropy_with_logits(
                    gap[use], torch.sigmoid(best[use] / sac_tau))
        if bool(spot.any()):
            extra["sac_value_mse"] = (v - delta).pow(2)[valid].mean().detach()
            best_real = delta.masked_fill(~valid, -torch.inf).max(1).values
            extra["sac_profitable_share"] = (best_real[spot] > 0).float().mean()
            extra["sac_p"] = torch.sigmoid(gap[spot]).mean().detach()
        extra["sac_decisions"] = spot.sum().float()
        extra["sac_clean_decisions"] = use.sum().float()
        extra["sac_fired"] = fired.sum().float()
        extra["sac_credit_mean"] = credit_sac[fired].mean() if bool(fired.any()) else zero
        extra["sac_level5_share"] = ((traj.actions[fired] >= 20).float().mean()
                                     if bool(fired.any()) else zero)

    rel = deals.rel_tricks(states.deal, seat)
    doubles = traj.actions == DOUBLE
    return {
        "policy_loss": -policy_by_group.mean(),
        "entropy": entropy_by_group.mean(),
        "policy_objective": -policy_by_group.mean() - entropy_weight * entropy_by_group.mean(),
        "critic_loss": F.mse_loss(value, returns),
        "q_loss": q_loss,
        "trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "double_credit_mean": credit_x[doubles].mean() if bool(doubles.any()) else zero,
        **extra,
        "return_mean": group_returns.mean(),
        **({"table_return_mean": ((traj.table_score - traj.ceiling) / TARGET_SCALE).mean()}
           if traj.table_score is not None else {}),
        "advantage_mean": advantage.mean(),
        "code_word_share": code_words.float().mean(),
        "light_open_share": light_open.float().mean(),
        "advantage_std": advantage.std(),
    }


def competitive_trajectory_stats(traj: CompetitiveTrajectories) -> dict:
    return {"train_sac_fired": int(traj.sac_fired.sum()),
            "train_standing_redoubles": int((traj.standing_xx >= 0).sum()),
            "train_clean_share": float(traj.clean.float().mean())}


# ---------------------------------------------------------------- validation

@torch.no_grad()
def competitive_validation(actor, deals: TorchDeals, scorer: TorchScorer, frozen_net=None) -> dict:
    """``fourseat_validation`` for D5OWN4XC nets: greedy four-seat play plus XX and SAC metrics.

    A greedy SAC "fires" when the chosen call is the SAC candidate and the SAC path carries
    more probability than its softmax path. ``objective`` = per-side own-bid score (sacrifice
    bids excluded) plus the standing X/XX credits and SAC credits, averaged over sides.
    ``spots_per_1000`` counts decisions (per 1,000 boards) where X, XX or SAC was offered.
    With ``frozen_net`` the learner-side ``pool_metrics`` (doubling level scored) are returned.
    """
    redouble, sacrifice = actor.redouble, actor.sacrifice
    rec = {k: [] for k in ("x_q", "x_delta", "x_final", "x_chosen", "xx_q", "xx_delta",
                           "sac_v", "sac_d", "sac_best", "f_idx", "f_call", "f_base", "f_side")}
    final_px = {"sum": 0.0, "count": 0}

    def watch(idx, state, chosen, masked, outputs):
        legal = state.legal()
        can = legal[:, DOUBLE]
        if bool(can.any()):
            final_px["sum"] += float(masked[can, DOUBLE].exp().sum())
            final_px["count"] += int(can.sum())
            r = can.nonzero().squeeze(1)
            rec["x_q"].append(outputs["double_value"][r])
            rec["x_delta"].append(double_delta(state.subset(r), deals))
            rec["x_final"].append(torch.ones_like(r, dtype=torch.bool))
            rec["x_chosen"].append(chosen[r] == DOUBLE)
        if redouble and bool(legal[:, REDOUBLE].any()):
            r = legal[:, REDOUBLE]
            rec["xx_q"].append(outputs["redouble_value"][r])
            rec["xx_delta"].append(redouble_delta(state, deals)[r])
        if sacrifice:
            _, valid, delta, baseline = sac_targets(state, deals, legal)
            spot = valid.any(1)
            if bool(spot.any()):
                rec["sac_v"].append(outputs["sac_value"][valid])
                rec["sac_d"].append(delta[valid])
                rec["sac_best"].append(delta.masked_fill(~valid, -torch.inf).max(1).values[spot])
            fired = greedy_sac_fired(outputs, legal, chosen)
            if bool(fired.any()):
                rec["f_idx"].append(idx[fired])
                rec["f_call"].append(chosen[fired])
                rec["f_base"].append(baseline[fired])
                rec["f_side"].append(state.side[fired])

    device = deals.hands.device
    start = competitive_batch_class(redouble).start(*fourseat_rows(deals.n, device))
    frozen = None
    if frozen_net is not None:
        side = (start.deal + start.dealer) % 2
        frozen = {"players": {1: frozen_net}, "code": torch.ones_like(side), "side": side}
    batch = play(actor, deals, start, greedy, on_decision=watch, frozen=frozen)
    cat = {k: torch.cat(v) if v else None for k, v in rec.items()}
    n = len(batch)
    rows = torch.arange(n, device=device)
    exclude = torch.zeros(n, 35, dtype=torch.bool, device=device)
    n_sac = 0 if cat["f_idx"] is None else len(cat["f_idx"])
    sac_credit = torch.zeros(0, device=device)
    if n_sac:
        exclude[cat["f_idx"], cat["f_call"]] = True
        final = side_table_score(batch, deals, cat["f_side"].new_zeros(n))[cat["f_idx"]]
        final = torch.where(cat["f_side"] == 0, final, -final)
        sac_credit = final - cat["f_base"]
    score, ceiling, table_undoubled = own_bid_scores(batch, deals, scorer, exclude=exclude)
    contracts = batch.last >= 0
    doubled = batch.doubled & contracts
    redoubled = batch.redoubled & contracts
    x_delta = final_double_delta(batch, deals) * TARGET_SCALE
    xx_delta = redouble_delta(batch, deals) * TARGET_SCALE
    level = torch.where(contracts, batch.last // 5 + 1, torch.zeros_like(batch.last))

    def count(key):
        return 0 if cat[key] is None else len(cat[key])

    extra: dict = {"sacs": n_sac, "sac_rate": n_sac / max(n, 1),
                   "spots_per_1000": {"x": 1000 * count("x_q") / max(n, 1),
                                      "xx": 1000 * count("xx_q") / max(n, 1),
                                      "sac": 1000 * count("sac_best") / max(n, 1)}}
    if n_sac:
        extra.update(sac_credit_per_sac=float(sac_credit.mean()),
                     sac_positive_share=float((sac_credit > 0).float().mean()),
                     sac_level5_share=float((cat["f_call"] >= 20).float().mean()))
    else:
        extra.update(sac_credit_per_sac=0.0, sac_positive_share=0.0, sac_level5_share=0.0)
    if cat["sac_v"] is not None:
        extra["sac_value"] = {"spots": count("sac_best"),
                              "mse": float((cat["sac_v"] - cat["sac_d"]).pow(2).mean()),
                              "profitable_share": float((cat["sac_best"] > 0).float().mean())}
    n_xx = int(redoubled.sum())
    extra.update(redoubles=n_xx,
                 redouble_rate=float(redoubled.sum() / doubled.sum().clamp(min=1)),
                 delta_per_redouble=float(xx_delta[redoubled].mean()) if n_xx else 0.0,
                 redouble_accuracy=float((xx_delta[redoubled] > 0).float().mean()) if n_xx else 0.0)
    if cat["xx_q"] is not None:
        extra["redouble_value"] = {"spots": count("xx_q"),
                                   "mse": float((cat["xx_q"] - cat["xx_delta"]).pow(2).mean()),
                                   "profitable_share": float((cat["xx_delta"] > 0).float().mean())}
    extra["level5_share"] = float((level >= 5).float().mean())
    # training/simplicity.py numbers (the one implementation), greedy auctions
    extra["simplicity"] = {k: v for k, v in analyse_batch(batch, deals).items() if k != "auctions"}

    if frozen is not None:
        out = pool_metrics(batch, deals, scorer, frozen["side"])
        learner = 1 - frozen["side"]
        owner = batch.bid_seats()[rows, batch.last.clamp(min=0)] % 2
        opp_contract = contracts & (owner == frozen["side"])
        x_mine = doubled & opp_contract
        xx_mine = redoubled & (owner == learner)
        n_x = int(x_mine.sum())
        table = side_table_score(batch, deals, learner)
        out.update({
            "learner_own_score": float(score[rows, learner].mean()),
            "learner_table_score": float(table.mean()),
            "double_accuracy": float((x_delta[x_mine] > 0).float().mean()) if n_x else 0.0,
            "delta_per_double": float(x_delta[x_mine].mean()) if n_x else 0.0,
            **extra, "redoubles": int(xx_mine.sum()),
            "delta_per_redouble": float(xx_delta[xx_mine].mean()) if bool(xx_mine.any()) else 0.0,
        })
        return out

    seats = batch.bid_seats()
    bid_ns = ((seats >= 0) & (seats % 2 == 0)).any(1)
    bid_ew = ((seats >= 0) & (seats % 2 == 1)).any(1)
    n_dbl = int(doubled.sum())
    x_credit = torch.where(doubled, x_delta, torch.zeros_like(x_delta))
    xx_credit = torch.where(redoubled, xx_delta, torch.zeros_like(xx_delta))
    sac_board = torch.zeros(n, device=device)
    if n_sac:
        sac_board.scatter_add_(0, cat["f_idx"], sac_credit)
    out = {
        "own_score": float(score.mean()),
        "own_ns": float(score[:, 0].mean()),
        "own_ew": float(score[:, 1].mean()),
        "own_regret": float((ceiling - score).mean()),
        "table_ns_undoubled": float(table_undoubled.mean()),
        "table_ns": float(table_ns_score(batch, deals).mean()),
        "passout": float((~contracts).float().mean()),
        "both_sides_bid": float((bid_ns & bid_ew).float().mean()),
        "calls": float(batch.t.float().mean()),
        "level_share": {str(lv): float((level == lv).float().mean()) for lv in range(8)},
        "double_rate": float(doubled.sum() / contracts.sum().clamp(min=1)),
        "double_accuracy": float((x_delta[doubled] > 0).float().mean()) if n_dbl else 0.0,
        "delta_per_double": float(x_delta[doubled].mean()) if n_dbl else 0.0,
        "doubles": n_dbl,
        "p_double_final": final_px["sum"] / max(final_px["count"], 1),
        "objective": float((score.sum(1) + x_credit + xx_credit + sac_board).mean() / 2),
        **extra,
    }
    if cat["x_q"] is not None:
        out["double_value"] = double_value_calibration(
            cat["x_q"], cat["x_delta"], cat["x_final"], cat["x_chosen"])
    return out
