"""Opponent-aware auction networks and critics, warm-started from a cooperative model.

The auction input layer is ``base(features[:77]) + extra(features[77:])``.
``base`` is copied from the cooperative checkpoint and ``extra`` starts at exactly
zero, so while opponents are silent (and nobody doubled) the network is
bit-identical to its initialization (adding an exact zero changes no float).

Two kinds:

- ``FourSeatNet`` (stage D5OWN4): 147 features, 36 actions.
- ``FourSeatDoubleValueNet`` (stage D5OWN4XV): D5OWN4X plus a zero-init scalar
  ``double_value`` head.
- ``FourSeatDoubleGateNet`` (stage D5OWN4XD): D5OWN4XV whose X gate, double
  value and X Q read detached trunk features.
- ``FourSeatDoubleNet`` (stage D5OWN4X): 149 features (+ standing-doubled bits),
  37 actions (+ Double). The Double policy row starts with zero weights and a
  negative bias; its Q row starts at zero. The first 36 outputs are identical to
  the warm-start source.
- ``FourSeatCompetitiveNet`` (stage D5OWN4XC): D5OWN4XD plus optional Redouble and a
  Sacrifice gate on detached heads (see ``competitive.py``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from ..bridge.calls import DOUBLE, PASS, REDOUBLE
from ..contract.environment import AUCTION_FEATURES
from ..contract.model import REWARD_UNITS, AuctionContractNet, load_checkpoint
from ..cooperative.actor_critic import CentralCritic
from .state import DOUBLE_FEATURES, OPPONENT_FEATURES

FOURSEAT_FORMAT = "bridgezero-fourseat-0.1"
STAGE = "D5OWN4"
DOUBLE_STAGE = "D5OWN4X"
DOUBLE_VALUE_STAGE = "D5OWN4XV"
DOUBLE_GATE_STAGE = "D5OWN4XD"
COMPETITIVE_STAGE = "D5OWN4XC"
SAC_STRAINS = 5
OBJECTIVE = ("four-seat shared-network self-play, no X/XX; per-side own-bid return "
             "(own highest bid undoubled - side cooperative DD ceiling)/100, gamma=1; "
             "actor_critic + trick_ce + chosen_action_q_mse; silent_frac episodes keep one side "
             "forced to Pass")
DOUBLE_OBJECTIVE = (
    "four-seat shared-network self-play with Double (no XX); per-side own-bid return "
    "(undoubled, bidder never charged for being doubled); a standing Double's action advantage "
    "also gets delta=(defenders doubled - undoubled result)/100; plus exact counterfactual "
    "-p2(X|{Pass,X})*delta_if_doubled_and_all_pass at final defending decisions; "
    "actor_critic + trick_ce + chosen_action_q_mse")
DOUBLE_VALUE_OBJECTIVE = (
    "four-seat shared-network self-play with Double (no XX); per-side undoubled own-bid return "
    "(bidder never charged); standing Double's action advantage gets delta; double_value head "
    "MSE to exact delta_if_X_now_then_all_pass at every legal-X spot; policy two-action CE "
    "toward softmax({Pass:0, X:double_value.detach()}/tau); actor_critic + trick_ce + "
    "chosen_action_q_mse")
DOUBLE_GATE_OBJECTIVE = DOUBLE_VALUE_OBJECTIVE.replace(
    "policy two-action CE toward", "X-vs-rest gate BCE toward") + (
    "; Double gate, double_value and X's Q read hidden.detach() (doubling never writes to the "
    "trunk); p(X)=sigmoid(gate), p(call)=(1-p(X))*softmax36(call)")
COMPETITIVE_OBJECTIVE = DOUBLE_GATE_OBJECTIVE.split("; Double gate")[0] + (
    "; D5OWN4XC: X, XX and SAC are offered only where the actor's Pass would end the auction "
    "(pass-out seat; the direct-seat player never doubles, redoubles or sacrifices) and only "
    "take probability from Pass: p(X or XX)=p_pass*sigmoid(gD), p(SAC)=p_pass*(1-sigmoid(gD))*"
    "sigmoid(gSAC), p(Pass)=p_pass*(1-sigmoid(gD))*(1-sigmoid(gSAC)), other calls softmax36. "
    "All gate/value/XX-Q heads read hidden.detach(). XX: legal for the declaring side over a "
    "standing X; redouble_value MSE to exact declaring-side (redoubled - doubled)/100 of "
    "XX-then-all-pass; XX gate BCE toward sigmoid(redouble_value/xx_tau). A standing XX's gate "
    "gets (redoubled - doubled)/100 and the standing X gets defenders' (result at the final "
    "doubling level - undoubled)/100. SAC: candidates are the cheapest bid per strain above the "
    "opponents' contract and at level 4+; sac_value (5 strains) MSE to exact [own side's doubled "
    "result of the candidate then all pass - own side's result of their contract at its "
    "level]/100; SAC gate BCE toward sigmoid(max legal sac_value/sac_tau); the SAC bids the "
    "highest-valued candidate (ties cheapest); a fired SAC's gate gets (own side's final real "
    "table result - its result of the contract it bid over)/100 and the sacrifice bid is "
    "excluded from its side's own-bid score and the Q loss. PG: trunk log softmax36(call or "
    "Pass) with the undoubled own-bid advantage, gate log-probs with the credited advantage; "
    "gate logits detached from the PG unless gate_pg. X/XX/SAC value and gate losses only at "
    "clean spots (every earlier opponent call greedy-top) unless all_spots")
FEATURE_NOTE = "77 own-partnership features + 35 LHO bids + 35 RHO bids (actor-relative)"
DOUBLE_FEATURE_NOTE = FEATURE_NOTE + " + standing-doubled + doubled-by-my-side"
COMPETITIVE_FEATURE_NOTE = DOUBLE_FEATURE_NOTE + " + Pass-would-end-auction + standing-redoubled"


class OpponentAugmentedInput(nn.Module):
    def __init__(self, width: int, extra: int = OPPONENT_FEATURES):
        super().__init__()
        self.base = nn.Linear(AUCTION_FEATURES, width)
        self.opponent = nn.Linear(extra, width, bias=False)
        nn.init.zeros_(self.opponent.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.base(features[:, :AUCTION_FEATURES].contiguous())
                + self.opponent(features[:, AUCTION_FEATURES:].contiguous()))


class FourSeatNet(AuctionContractNet):
    """``AuctionContractNet`` whose auction input also sees opponent bids."""

    kind = "fourseat_auction"
    stage = STAGE
    extra_features = OPPONENT_FEATURES
    n_actions = PASS + 1

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None):
        super().__init__(width, suit_width, depth, policy_logit_bound)
        self.auction_net = nn.Sequential(OpponentAugmentedInput(width, self.extra_features),
                                         nn.GELU())
        if self.n_actions != PASS + 1:
            self.q_head = nn.Linear(width, self.n_actions)
            self.policy_head = nn.Linear(width, self.n_actions)
            for head in (self.q_head, self.policy_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)


class FourSeatDoubleNet(FourSeatNet):
    kind = "fourseat_double_auction"
    stage = DOUBLE_STAGE
    extra_features = OPPONENT_FEATURES + DOUBLE_FEATURES
    n_actions = DOUBLE + 1


class FourSeatDoubleValueNet(FourSeatDoubleNet):
    """Double net plus ``double_value``: E[defenders' delta if X now and all pass | obs] / 100."""

    kind = "fourseat_double_value_auction"
    stage = DOUBLE_VALUE_STAGE
    double_value = True

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None):
        super().__init__(width, suit_width, depth, policy_logit_bound)
        self.double_value_head = nn.Linear(width, 1)
        nn.init.zeros_(self.double_value_head.weight)
        nn.init.zeros_(self.double_value_head.bias)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(hand, auction)
        return {
            "trick_logits": self.trick_head(hidden).view(len(hand), 2, 5, 14),
            "contract_q": self.q_head(hidden),
            "policy_logits": self.policy_logits(hidden),
            "double_value": self.double_value_head(hidden).squeeze(-1),
        }


class FourSeatDoubleGateNet(FourSeatDoubleValueNet):
    """Double read-only on the trunk: gate, double value and X's Q see ``hidden.detach()``.

    ``policy_logits[:, 36]`` is the X gate logit, not a flat-softmax logit; use
    ``policy_log_probs`` for probabilities. The first 36 logits come from the trunk.
    """

    kind = "fourseat_double_gate_auction"
    stage = DOUBLE_GATE_STAGE
    gated_double = True

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None):
        super().__init__(width, suit_width, depth, policy_logit_bound)
        self.double_gate_head = nn.Linear(width, 1)
        nn.init.zeros_(self.double_gate_head.weight)
        nn.init.zeros_(self.double_gate_head.bias)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(hand, auction)
        frozen = hidden.detach()
        gate = self.double_gate_head(frozen)
        q = self.q_head(hidden)[:, :DOUBLE]
        q_x = nn.functional.linear(frozen, self.q_head.weight[DOUBLE:], self.q_head.bias[DOUBLE:])
        return {
            "trick_logits": self.trick_head(hidden).view(len(hand), 2, 5, 14),
            "contract_q": torch.cat((q, q_x), 1),
            # Bound only the 36 softmax calls: column DOUBLE of policy_head is unused here
            # and must not stretch the bound's range.
            "policy_logits": torch.cat(
                (self.bound_policy_logits(self.policy_head(hidden)[:, :DOUBLE]), gate), 1),
            "double_gate": gate.squeeze(-1),
            "double_value": self.double_value_head(frozen).squeeze(-1),
        }


def policy_log_probs(outputs: dict, legal: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Log-probabilities over the legal actions (-inf when illegal).

    Flat nets: masked log-softmax. Gated nets: when X is legal, ``log sigmoid(gate)``
    for X and ``log sigmoid(-gate) + log_softmax36`` for the other calls.
    """
    if "sac_gate" in outputs or "redouble_gate" in outputs:
        return competitive_log_probs(outputs, legal, temperature)
    n = legal.shape[1]
    logits = outputs["policy_logits"][:, :n] / temperature
    if "double_gate" not in outputs:
        return torch.log_softmax(logits.masked_fill(~legal, -torch.inf), -1)
    rest = torch.log_softmax(logits[:, :DOUBLE].masked_fill(~legal[:, :DOUBLE], -torch.inf), -1)
    gate = outputs["double_gate"] / temperature
    can = legal[:, DOUBLE]
    rest = torch.where(can[:, None], rest + nn.functional.logsigmoid(-gate)[:, None], rest)
    x = torch.where(can, nn.functional.logsigmoid(gate), torch.full_like(gate, -torch.inf))
    return torch.cat((rest, x[:, None]), 1)


class FourSeatCritic(CentralCritic):
    """Training-only ``V(pair hands, four-seat auction features)``."""

    extra_features = OPPONENT_FEATURES

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3):
        super().__init__(width, suit_width, depth)
        self.auction_net = nn.Sequential(OpponentAugmentedInput(width, self.extra_features),
                                         nn.GELU())


class FourSeatDoubleCritic(FourSeatCritic):
    extra_features = OPPONENT_FEATURES + DOUBLE_FEATURES


FINAL_FEATURE = AUCTION_FEATURES + OPPONENT_FEATURES + DOUBLE_FEATURES   # [149] Pass would end the auction
REDOUBLED_FEATURE = FINAL_FEATURE + 1                                    # [150] standing contract redoubled
COMPETITIVE_EXTRA = 2
SAC_MIN_CONTRACT = 15                                                    # 4C: sacrifices are level 4+


class CompetitiveInput(OpponentAugmentedInput):
    """``OpponentAugmentedInput`` plus a separate zero-init layer for the pass-ends / XX bits.

    A separate layer (not a wider ``opponent`` matrix) keeps the other features' products
    bit-identical to the source net: the new term adds an exact zero at warm start.
    """

    def __init__(self, width: int):
        super().__init__(width, OPPONENT_FEATURES + DOUBLE_FEATURES)
        self.competitive = nn.Linear(COMPETITIVE_EXTRA, width, bias=False)
        nn.init.zeros_(self.competitive.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        cut = FINAL_FEATURE
        return (self.base(features[:, :AUCTION_FEATURES].contiguous())
                + self.opponent(features[:, AUCTION_FEATURES:cut].contiguous())
                + self.competitive(features[:, cut:cut + COMPETITIVE_EXTRA].contiguous()))


def sac_candidates_from_last(last: torch.Tensor, spot: torch.Tensor) -> torch.Tensor:
    """``(B,5)`` cheapest bid per strain (C,D,H,S,NT) above contract ``last`` and at level 4+.

    -1 off ``spot`` or when no such bid exists (above 7NT).
    """
    strain = torch.arange(SAC_STRAINS, device=last.device)[None]
    c = last.clamp(min=0)[:, None]
    base = c - c % 5
    cand = torch.where(strain > c % 5, base + strain, base + 5 + strain)
    cand = torch.maximum(cand, SAC_MIN_CONTRACT + strain)
    ok = spot[:, None] & (last[:, None] >= 0) & (cand < 35)
    return torch.where(ok, cand, torch.full_like(cand, -1))


def sac_candidates_from_features(features: torch.Tensor) -> torch.Tensor:
    """SAC candidates where the opponents hold the standing contract and Pass would end it.

    The highest bid bit in the LHO/RHO blocks above every own/partner bid bit means the
    opponents hold the contract; ``[149]`` says Pass ends the auction. Legality (forced)
    is checked by the caller.
    """
    ladder = torch.arange(35, device=features.device)[None]
    none = torch.full((len(features), 35), -1, dtype=torch.long, device=features.device)
    own = (features[:, 0:35] + features[:, 35:70]) > 0
    opp = (features[:, AUCTION_FEATURES:AUCTION_FEATURES + 35]
           + features[:, AUCTION_FEATURES + 35:AUCTION_FEATURES + 70]) > 0
    top_own = torch.where(own, ladder, none).max(1).values
    top_opp = torch.where(opp, ladder, none).max(1).values
    final = features[:, FINAL_FEATURE] > 0
    return sac_candidates_from_last(top_opp, (top_opp > top_own) & final)


class FourSeatCompetitiveNet(FourSeatDoubleGateNet):
    """D5OWN4XD plus optional Redouble (action 37) and a Sacrifice gate, all on detached heads.

    Inputs: the 149 double features + ``[149]`` Pass would end the auction + ``[150]``
    standing redoubled, through a separate zero-init layer. ``redouble``: XX gate / value /
    Q heads. ``sacrifice``: ``sac_gate_head`` (1) and ``sac_value_head`` (5, strains C..NT);
    outputs also carry ``sac_candidates``. Policy and Q heads keep 37 rows, so the first 36
    logits (and any logit bound) are exactly the source's. Use ``policy_log_probs``.
    """

    kind = "fourseat_competitive_auction"
    stage = COMPETITIVE_STAGE
    extra_features = OPPONENT_FEATURES + DOUBLE_FEATURES + COMPETITIVE_EXTRA

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3,
                 policy_logit_bound: float | None = None, redouble: bool = True,
                 sacrifice: bool = True):
        if not (redouble or sacrifice):
            raise ValueError("a competitive net needs redouble and/or sacrifice")
        # plain attribute read by FourSeatNet.__init__; setting it before Module init is fine
        self.n_actions = REDOUBLE + 1 if redouble else DOUBLE + 1
        super().__init__(width, suit_width, depth, policy_logit_bound)
        self.redouble, self.sacrifice = bool(redouble), bool(sacrifice)
        self.config.update(redouble=self.redouble, sacrifice=self.sacrifice)
        self.auction_net = nn.Sequential(CompetitiveInput(width), nn.GELU())
        heads = {"q_head": DOUBLE + 1, "policy_head": DOUBLE + 1}
        if self.redouble:
            heads.update(redouble_gate_head=1, redouble_value_head=1, redouble_q_head=1)
        if self.sacrifice:
            heads.update(sac_gate_head=1, sac_value_head=SAC_STRAINS)
        for name, rows in heads.items():
            head = nn.Linear(width, rows)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            setattr(self, name, head)

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(hand, auction)
        frozen = hidden.detach()
        gate = self.double_gate_head(frozen)
        q = self.q_head(hidden)[:, :DOUBLE]
        q_x = nn.functional.linear(frozen, self.q_head.weight[DOUBLE:], self.q_head.bias[DOUBLE:])
        # as in FourSeatDoubleGateNet: bound only the 36 softmax calls
        q_parts, logits = [q, q_x], [self.bound_policy_logits(self.policy_head(hidden)[:, :DOUBLE]),
                                     gate]
        out = {"trick_logits": self.trick_head(hidden).view(len(hand), 2, 5, 14),
               "double_gate": gate.squeeze(-1),
               "double_value": self.double_value_head(frozen).squeeze(-1)}
        if self.redouble:
            xx_gate = self.redouble_gate_head(frozen)
            q_parts.append(self.redouble_q_head(frozen))
            logits.append(xx_gate)
            out["redouble_gate"] = xx_gate.squeeze(-1)
            out["redouble_value"] = self.redouble_value_head(frozen).squeeze(-1)
        if self.sacrifice:
            out["sac_gate"] = self.sac_gate_head(frozen).squeeze(-1)
            out["sac_value"] = self.sac_value_head(frozen)
            out["sac_candidates"] = sac_candidates_from_features(auction)
        out["contract_q"] = torch.cat(q_parts, 1)
        out["policy_logits"] = torch.cat(logits, 1)
        return out


class FourSeatCompetitiveCritic(FourSeatDoubleCritic):
    """Double critic that also reads the pass-ends / standing-XX bits (zero-init layer)."""

    extra_features = OPPONENT_FEATURES + DOUBLE_FEATURES + COMPETITIVE_EXTRA

    def __init__(self, width: int = 384, suit_width: int = 64, depth: int = 3):
        super().__init__(width, suit_width, depth)
        self.auction_net = nn.Sequential(CompetitiveInput(width), nn.GELU())


def competitive_parts(outputs: dict, legal: torch.Tensor, temperature: float = 1.0,
                      detach_gates: bool = False) -> dict:
    """Pieces of the D5OWN4XC policy composition at ``temperature``.

    ``rest``: log softmax36 over bids + Pass. ``d_gate``: the X or XX gate (never both legal;
    both are legal only where Pass ends the auction). ``spot``: a legal level-4+ SAC candidate
    exists (Pass ends the auction, opponents hold the contract); ``chosen``: the candidate
    with the highest ``sac_value`` (ties -> cheapest), -1 off spots. ``sac_lp``: log-prob of
    the SAC path, ``log p_pass + log(1-p_D) + log p_SAC``. ``detach_gates`` detaches all
    three gate logits.
    """
    logsig = nn.functional.logsigmoid
    logits = outputs["policy_logits"][:, :DOUBLE] / temperature
    rest = torch.log_softmax(logits.masked_fill(~legal[:, :DOUBLE], -torch.inf), -1)
    can_x = legal[:, DOUBLE]
    can_xx = legal[:, REDOUBLE] if legal.shape[1] > REDOUBLE else torch.zeros_like(can_x)
    gate = outputs["double_gate"] / temperature
    if "redouble_gate" in outputs:
        gate = torch.where(can_xx, outputs["redouble_gate"] / temperature, gate)
    if detach_gates:
        gate = gate.detach()
    rows = torch.arange(len(legal), device=legal.device)
    chosen = torch.full_like(rows, -1)
    spot = torch.zeros_like(can_x)
    s_gate = torch.zeros_like(gate)
    if "sac_gate" in outputs:
        cand = outputs["sac_candidates"]
        valid = (cand >= 0) & legal[rows[:, None], cand.clamp(min=0)]
        spot = valid.any(1)
        value = outputs["sac_value"].detach().masked_fill(~valid, -torch.inf)
        tie = valid & (value == value.max(1, keepdim=True).values)
        pick = torch.where(tie, cand, torch.full_like(cand, 99)).min(1).values
        chosen = torch.where(spot, pick, chosen)
        s_gate = outputs["sac_gate"] / temperature
        if detach_gates:
            s_gate = s_gate.detach()
    zero = torch.zeros_like(gate)
    can_d = can_x | can_xx
    not_d = torch.where(can_d, logsig(-gate), zero)
    not_s = torch.where(spot, logsig(-s_gate), zero)
    pass_lp = rest[:, PASS]
    return {"rest": rest, "can_x": can_x, "can_xx": can_xx, "can_d": can_d, "d_gate": gate,
            "spot": spot, "chosen": chosen, "s_gate": s_gate, "not_d": not_d, "not_s": not_s,
            "pass_lp": pass_lp, "d_lp": pass_lp + logsig(gate),
            "sac_lp": pass_lp + not_d + logsig(s_gate)}


def competitive_log_probs(outputs: dict, legal: torch.Tensor, temperature: float = 1.0,
                          detach_gates: bool = False, static: bool = False) -> torch.Tensor:
    """Marginal log p(call) of the D5OWN4XC composition (-inf when illegal).

    X/XX/SAC only take mass from Pass: p(X or XX) = p_pass p_D, p(SAC) = p_pass (1-p_D) p_SAC,
    p(Pass) = p_pass (1-p_D)(1-p_SAC), and the SAC candidate also keeps its softmax mass.
    Rows without a legal X/XX/SAC are exactly log softmax36, as in D5OWN4XD without X.
    ``static``: always take the mixing branch (same values; no host sync, CUDA-graph safe).
    """
    part = competitive_parts(outputs, legal, temperature, detach_gates)
    rest, can_d, spot = part["rest"], part["can_d"], part["spot"]
    calls = rest
    special = can_d | spot
    if static or bool(special.any()):
        hit = (torch.arange(DOUBLE, device=legal.device)[None] == part["chosen"][:, None]) & spot[:, None]
        floor = torch.full_like(rest, -1e30)       # finite: logaddexp backward is NaN at (-inf,-inf)
        mixed = torch.logaddexp(torch.where(hit, rest, floor),
                                torch.where(hit, part["sac_lp"][:, None], floor))
        calls = torch.where(hit, mixed, rest)
        new_pass = torch.where(special, part["pass_lp"] + part["not_d"] + part["not_s"],
                               part["pass_lp"])
        calls = torch.cat((calls[:, :PASS], new_pass[:, None]), 1)
    never = torch.full_like(part["pass_lp"], -torch.inf)
    d = torch.where(can_d, part["d_lp"], never)
    columns = [calls, torch.where(part["can_x"], d, never)[:, None]]
    if legal.shape[1] > REDOUBLE:
        columns.append(torch.where(part["can_xx"], d, never)[:, None])
    return torch.cat(columns, 1)


def competitive_path_log_probs(outputs: dict, legal: torch.Tensor, actions: torch.Tensor,
                               sac_fired: torch.Tensor, temperature: float = 1.0,
                               detach_gates: bool = True):
    """``(trunk_logp, gate_logp)`` of the sampled latent path, for the PG.

    The trunk picks a softmax36 call; X/XX, a fired SAC and Pass all go through its Pass
    entry, after which the gates decide. Trunk part: log softmax36(call or Pass). Gate part:
    X/XX log p_D; fired SAC log(1-p_D) + log p_SAC; Pass log(1-p_D) + log(1-p_SAC) (each only
    where legal); 0 for a bid chosen by the softmax. exp(trunk + gate) summed over latent
    paths is the marginal probability.
    """
    part = competitive_parts(outputs, legal, temperature, detach_gates)
    logsig = nn.functional.logsigmoid
    rows = torch.arange(len(actions), device=actions.device)
    is_d = actions >= DOUBLE
    is_pass = actions == PASS
    via_pass = is_d | sac_fired | is_pass
    trunk = torch.where(via_pass, part["pass_lp"], part["rest"][rows, actions.clamp(max=PASS)])
    zero = torch.zeros_like(part["pass_lp"])
    gate = torch.where(is_d, logsig(part["d_gate"]),
                       torch.where(sac_fired, part["not_d"] + logsig(part["s_gate"]),
                                   torch.where(is_pass, part["not_d"] + part["not_s"], zero)))
    return trunk, gate


KINDS = {FourSeatNet.kind: (FourSeatNet, FourSeatCritic),
         FourSeatDoubleNet.kind: (FourSeatDoubleNet, FourSeatDoubleCritic),
         FourSeatDoubleValueNet.kind: (FourSeatDoubleValueNet, FourSeatDoubleCritic),
         FourSeatDoubleGateNet.kind: (FourSeatDoubleGateNet, FourSeatDoubleCritic),
         FourSeatCompetitiveNet.kind: (FourSeatCompetitiveNet, FourSeatCompetitiveCritic)}
STAGE_TEXT = {STAGE: (OBJECTIVE, FEATURE_NOTE), DOUBLE_STAGE: (DOUBLE_OBJECTIVE, DOUBLE_FEATURE_NOTE),
              DOUBLE_VALUE_STAGE: (DOUBLE_VALUE_OBJECTIVE, DOUBLE_FEATURE_NOTE),
              DOUBLE_GATE_STAGE: (DOUBLE_GATE_OBJECTIVE, DOUBLE_FEATURE_NOTE),
              COMPETITIVE_STAGE: (COMPETITIVE_OBJECTIVE, COMPETITIVE_FEATURE_NOTE)}


class SilentView(nn.Module):
    """Feed 77-feature cooperative inputs to a four-seat net (all extra bits zero).

    Outputs are cut to the 36 cooperative actions: Double is never legal in a
    silent-opponent auction, so this is exact. Lets ``evaluate_auctions`` score
    a four-seat net on the silent-opponent rows.
    """

    def __init__(self, net: FourSeatNet):
        super().__init__()
        self.net = net

    def forward(self, hand: torch.Tensor, auction: torch.Tensor) -> dict:
        pad = auction.new_zeros(len(auction), self.net.extra_features)
        out = self.net(hand, torch.cat((auction, pad), 1))
        return {**out, "policy_logits": out["policy_logits"][:, :PASS + 1],
                "contract_q": out["contract_q"][:, :PASS + 1]}


def _lift_state(source: dict, target: nn.Module) -> dict:
    state = target.state_dict()
    for name, value in source.items():
        key = name.replace("auction_net.0.", "auction_net.0.base.")
        if key not in state:
            raise ValueError(f"cannot map cooperative parameter {name}")
        if state[key].shape == value.shape:
            state[key] = value.clone()
        elif key.split(".")[0] in ("policy_head", "q_head") and state[key].shape[1:] == value.shape[1:]:
            state[key] = torch.zeros_like(state[key])
            state[key][:len(value)] = value
        else:
            raise ValueError(f"shape mismatch for {name}")
    zero = "auction_net.0.opponent.weight"
    state[zero] = torch.zeros_like(state[zero])
    new_heads = [k for k in state if k.startswith(("double_value_head.", "double_gate_head."))]
    for key in new_heads:
        state[key] = torch.zeros_like(state[key])
    if len(source) + 1 + len(new_heads) != len(state):
        raise ValueError("cooperative checkpoint does not cover every four-seat parameter")
    return state


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def warm_start(init_path: str | Path, device="cpu", doubles: bool = False,
               double_bias: float = -3.0, double_value: bool = False,
               double_gate: bool = False):
    """Four-seat actor and critic initialized from a cooperative checkpoint."""
    base, ck = load_checkpoint(init_path, device)
    if ck.get("model_kind") != "auction" or "critic" not in ck:
        raise ValueError("warm start needs an AuctionContractNet checkpoint with a critic")
    kind = (FourSeatDoubleGateNet.kind if double_gate else
            FourSeatDoubleValueNet.kind if double_value else
            FourSeatDoubleNet.kind if doubles else FourSeatNet.kind)
    doubles = doubles or double_value or double_gate
    net_cls, critic_cls = KINDS[kind]
    actor = net_cls(**ck["model_config"]).to(device)
    actor.load_state_dict(_lift_state(base.state_dict(), actor))
    if doubles:
        with torch.no_grad():
            actor.policy_head.bias[DOUBLE] = double_bias
            if double_gate:
                actor.double_gate_head.bias.fill_(double_bias)
    critic = critic_cls(**ck["critic_config"]).to(device)
    critic.load_state_dict(_lift_state(ck["critic"], critic))
    meta = {"init_path": str(init_path), "init_sha256": sha256(init_path),
            "init_stage": ck.get("stage"), "init_step": ck.get("step")}
    return actor, critic, meta


def save_fourseat_checkpoint(path: str | Path, actor: FourSeatNet, critic: FourSeatCritic,
                             **extra) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    objective, features = STAGE_TEXT[actor.stage]
    torch.save({"format": FOURSEAT_FORMAT, "stage": actor.stage, "objective": objective,
                "reward_units": REWARD_UNITS, "features": features,
                "model_kind": actor.kind, "model_config": actor.config,
                "net": actor.state_dict(), "critic_config": critic.config,
                "critic": critic.state_dict(), **extra}, path)


def load_fourseat_checkpoint(path: str | Path, device="cpu", with_critic: bool = False):
    """``(actor, checkpoint)`` or ``(actor, critic, checkpoint)`` from a D5OWN4/D5OWN4X file."""
    ck = torch.load(path, map_location=device, weights_only=False)
    kind = ck.get("model_kind")
    if ck.get("format") != FOURSEAT_FORMAT or kind not in KINDS:
        raise ValueError(f"{path} is not a {FOURSEAT_FORMAT} checkpoint")
    net_cls, critic_cls = KINDS[kind]
    if ck.get("stage") != net_cls.stage:
        raise ValueError(f"checkpoint stage {ck.get('stage')} != {net_cls.stage}")
    if ck.get("reward_units") != REWARD_UNITS:
        raise ValueError("checkpoint reward units differ from this code")
    actor = net_cls(**ck["model_config"]).to(device)
    actor.load_state_dict(ck["net"])
    if not with_critic:
        return actor, ck
    critic = critic_cls(**ck["critic_config"]).to(device)
    critic.load_state_dict(ck["critic"])
    return actor, critic, ck


def _pad_fourseat_state(source: dict, target: nn.Module,
                        new_heads: tuple = ("double_value_head.", "double_gate_head.")) -> dict:
    """Copy a D5OWN4 state into a larger double net: pad new columns/rows with zeros."""
    state = target.state_dict()
    for name, value in source.items():
        if name not in state or state[name].dim() != value.dim():
            raise ValueError(f"cannot map four-seat parameter {name}")
        if state[name].shape == value.shape:
            state[name] = value.clone()
        else:
            padded = torch.zeros_like(state[name])
            padded[tuple(slice(0, size) for size in value.shape)] = value
            state[name] = padded
    for name in state:
        if name not in source:
            if not name.startswith(new_heads):
                raise ValueError(f"no source for {name}")
            state[name] = torch.zeros_like(state[name])
    return state


def warm_start_from_fourseat(init_path: str | Path, device="cpu", double_bias: float = -6.0):
    """D5OWN4XD actor/critic from a D5OWN4 checkpoint (e.g. E18).

    Opponent-bid weights, heads and trunk are copied; the two doubled-state
    columns, the X policy/Q rows and the double heads start at zero, the X gate
    bias at ``double_bias`` < 0. Greedy calls are therefore identical to the source
    (X needs gate > 0; the other 36 log-probs only shift by log(1 - p_X)).
    """
    source, source_critic, ck = load_fourseat_checkpoint(init_path, device, with_critic=True)
    meta = {"init_path": str(init_path), "init_sha256": sha256(init_path),
            "init_stage": ck.get("stage"), "init_step": ck.get("step")}
    if ck.get("model_kind") == FourSeatDoubleGateNet.kind:
        # continue a D5OWN4XD model (e.g. E20b) unchanged: exact weights, no re-initialisation
        return source, source_critic, meta
    if ck.get("model_kind") != FourSeatNet.kind:
        raise ValueError("warm_start_from_fourseat needs a D5OWN4 or D5OWN4XD checkpoint")
    if double_bias >= 0:
        raise ValueError("double_bias must be negative so greedy play never doubles at start")
    actor = FourSeatDoubleGateNet(**ck["model_config"]).to(device)
    actor.load_state_dict(_pad_fourseat_state(source.state_dict(), actor))
    critic = FourSeatDoubleCritic(**ck["critic_config"]).to(device)
    critic.load_state_dict(_pad_fourseat_state(source_critic.state_dict(), critic))
    with torch.no_grad():
        actor.policy_head.bias[DOUBLE] = double_bias
        actor.double_gate_head.bias.fill_(double_bias)
    meta = {"init_path": str(init_path), "init_sha256": sha256(init_path),
            "init_stage": ck.get("stage"), "init_step": ck.get("step")}
    return actor, critic, meta


COMPETITIVE_NEW = ("redouble_gate_head.", "redouble_value_head.", "redouble_q_head.",
                   "sac_gate_head.", "sac_value_head.", "auction_net.0.competitive.")


def warm_start_competitive(init_path: str | Path, device="cpu", fourseat: bool = True,
                           redouble: bool = True, sacrifice: bool = True,
                           double_bias: float = -6.0, xx_bias: float = -6.0,
                           sac_bias: float = -6.0):
    """D5OWN4XC actor/critic.

    ``init_path``: a D5OWN4XC checkpoint with the same options (continued unchanged: every
    weight including the X/XX/SAC heads; extra keys such as ``belief_head`` are ignored here;
    detected by ``model_kind`` whatever ``fourseat`` says), a D5OWN4XD one (e.g. E20b; every
    weight copied), a D5OWN4 one (via ``warm_start_from_fourseat``) or, with
    ``fourseat=False``, a cooperative one. New parameters start at zero and the XX/SAC gate
    biases at ``xx_bias``/``sac_bias`` < 0, so at step 0 calls away from XX/SAC spots have
    exactly the source log-probabilities.
    """
    ck = torch.load(init_path, map_location=device, weights_only=False)
    if ck.get("model_kind") == FourSeatCompetitiveNet.kind:
        actor, critic, ck = load_fourseat_checkpoint(init_path, device, with_critic=True)
        if (actor.redouble, actor.sacrifice) != (bool(redouble), bool(sacrifice)):
            raise ValueError("D5OWN4XC checkpoint has different redouble/sacrifice options")
        return actor, critic, {"init_path": str(init_path), "init_sha256": sha256(init_path),
                               "init_stage": ck.get("stage"), "init_step": ck.get("step")}
    if fourseat:
        source, source_critic, meta = warm_start_from_fourseat(init_path, device, double_bias)
    else:
        source, source_critic, meta = warm_start(init_path, device, double_bias=double_bias,
                                                 double_gate=True)
    if xx_bias >= 0 or sac_bias >= 0:
        raise ValueError("xx_bias and sac_bias must be negative so XX/SAC start near zero")
    actor = FourSeatCompetitiveNet(**source.config, redouble=redouble, sacrifice=sacrifice).to(device)
    actor.load_state_dict(_pad_fourseat_state(source.state_dict(), actor, COMPETITIVE_NEW))
    critic = FourSeatCompetitiveCritic(**source_critic.config).to(device)
    critic.load_state_dict(_pad_fourseat_state(source_critic.state_dict(), critic, COMPETITIVE_NEW))
    with torch.no_grad():
        if actor.redouble:
            actor.redouble_gate_head.bias.fill_(xx_bias)
        if actor.sacrifice:
            actor.sac_gate_head.bias.fill_(sac_bias)
    return actor, critic, meta
