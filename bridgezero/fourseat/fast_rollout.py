"""Fixed-shape four-seat rollout (``--fast-rollout``), CUDA-graph captured on GPU.

Same game, same policy, same episode setup draws as ``rollout.collect_trajectories``
(or ``competitive.collect_competitive_trajectories`` for D5OWN4XC nets); only the
per-call sampling differs. The old path calls ``torch.multinomial`` on CPU for the
live rows each round. Here every round advances all episodes at a fixed batch shape
(ended rows are masked, not dropped) and samples by inverse CDF from uniforms drawn
up front on the CPU ``generator``: ``(rounds, episodes)`` per rollout, plus a second
``(rounds, episodes)`` block for the SAC gate's fire draw when the net has a Sacrifice
gate. The action distribution is identical, the random stream is not, so runs are not
bit-identical to the default path (they are reproducible from the seed, on CPU and GPU
alike, up to float rounding at CDF edges).

Features are kept incrementally (absolute per-seat bid bits, early-pass flags, the
standing doubler's side, and for D5OWN4XC the pass-ends / redoubled bits) instead of
being rebuilt from the full history each call. Decision states for the losses are
rebuilt afterwards from the final history: a row decides at round ``r`` with exactly
``r`` calls made, and history is append-only.

D5OWN4XC (``--redouble`` / ``--sacrifice``): X/XX only at the pass-out seat, the Redouble
state, the gate composition (``competitive_log_probs(static=True)``: branch-free), SAC
candidate selection (inside the net's forward) and the latent path record (whether the
SAC gate fired, whether the call was the greedy top call) all run inside the round.
Trajectory assembly (own-bid exclusion of fired SACs, credits, clean spots) runs once
after the rollout through ``competitive.competitive_trajectories``, shared with the
default path.

On CUDA one round (features, masks, actor forward, frozen pool players, sampling,
state update, recording) is a single ``CUDAGraph.replay``; termination is checked
every ``check_every`` rounds, so a rollout costs a handful of host-device syncs.
The graph binds the actor's parameter storage, which in-place optimizer steps and
``load_state_dict`` keep.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..bridge.calls import DOUBLE, PASS, REDOUBLE
from ..contract.data import TorchDeals
from ..contract.environment import AUCTION_FEATURES
from ..contract.targets import TorchScorer
from .competitive import MAX_REDOUBLE_CALLS, competitive_batch_class, competitive_trajectories
from .model import COMPETITIVE_STAGE, competitive_log_probs, competitive_parts, policy_log_probs
from .rollout import FourSeatTrajectories, batch_class, episode_setup
from .state import MAX_CALLS, double_delta, own_bid_scores

ROUNDS = MAX_CALLS + 1
COMPETITIVE_ROUNDS = MAX_REDOUBLE_CALLS + 1       # the default loop's bound for 320-wide histories
# gather order of absolute seats relative to the actor: self, partner, LHO, RHO
_REL_ORDER = (0, 2, 1, 3)


def inverse_cdf_sample(log_probs: torch.Tensor, u: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """One action per row with ``P(a) = exp(log_probs[a])`` from uniforms ``u`` in [0, 1).

    Illegal actions have probability 0 so they are never the first CDF step above
    ``u * total``; the clamp only guards ``u * total`` rounding up to the total.
    """
    cdf = log_probs.exp().cumsum(-1)
    k = (cdf <= u[:, None] * cdf[:, -1:]).sum(-1)
    idx = torch.arange(legal.shape[1], device=legal.device)
    last_legal = torch.where(legal, idx, torch.zeros_like(idx)).amax(-1)
    return torch.minimum(k, last_legal)


def sac_fire(outputs: dict, legal: torch.Tensor, chosen: torch.Tensor, u: torch.Tensor,
             temperature: float = 1.0) -> torch.Tensor:
    """Whether the SAC gate chose ``chosen``: posterior ``p(SAC path) / p(call)`` vs ``u``."""
    part = competitive_parts(outputs, legal, temperature)
    hit = part["spot"] & (chosen == part["chosen"])
    soft = part["rest"].gather(1, chosen.clamp(max=PASS)[:, None]).squeeze(1)
    fire = part["sac_lp"] - torch.logaddexp(soft, part["sac_lp"])
    return hit & (u < fire.exp())


class FastCollector:
    """Reusable fixed-shape collector for one actor, episode count, game and pool."""

    def __init__(self, actor, episodes: int, device, doubles: bool = False,
                 temperature: float = 1.0, pool: dict | None = None,
                 cuda_graph: bool | None = None, check_every: int = 6, compile: bool = False):
        if episodes < 1 or temperature <= 0:
            raise ValueError("invalid episodes or temperature")
        self.competitive = getattr(actor, "stage", None) == COMPETITIVE_STAGE
        self.redouble = self.competitive and bool(actor.redouble)
        self.sacrifice = self.competitive and bool(actor.sacrifice)
        doubles = doubles or self.competitive
        self.actor, self.B, self.doubles = actor, episodes, doubles
        self.temperature, self.pool = temperature, pool or {}
        from . import competitive as _competitive
        self.any_seat = self.competitive and _competitive.ANY_SEAT_DOUBLE
        self.device = torch.device(device)
        self.cuda_graph = self.device.type == "cuda" if cuda_graph is None else cuda_graph
        self.check_every = check_every
        self.n_actions = (REDOUBLE + 1 if self.redouble else
                          DOUBLE + 1 if doubles else PASS + 1)
        self.H = MAX_REDOUBLE_CALLS if self.competitive else MAX_CALLS
        self.rounds = COMPETITIVE_ROUNDS if self.competitive else ROUNDS
        self.graph = None
        self.round = torch.compile(self._round) if compile else self._round
        B, R, dev = episodes, self.rounds, self.device
        long = dict(dtype=torch.long, device=dev)
        boolean = dict(dtype=torch.bool, device=dev)
        # per-episode inputs
        self.deal = torch.zeros(B, **long)
        self.dealer = torch.zeros(B, **long)
        self.vul = torch.zeros(B, 2, **boolean)
        self.silent = torch.zeros(B, **long)
        self.code = torch.zeros(B, **long)
        self.frozen_side = torch.zeros(B, **long)
        self.hands = torch.zeros(B, 4, 52, device=dev)
        self.u = torch.zeros(R, B, device=dev)
        self.u_fire = torch.zeros(R, B, device=dev)
        # auction state
        self.t = torch.zeros(B, **long)
        self.last = torch.zeros(B, **long)
        self.pass_count = torch.zeros(B, **long)
        self.ended = torch.zeros(B, **boolean)
        self.contract_seat = torch.zeros(B, **long)
        self.doubled = torch.zeros(B, **boolean)
        self.redoubled = torch.zeros(B, **boolean)
        self.doubler_side = torch.zeros(B, **long)
        self.history = torch.zeros(B, self.H, **long)
        self.bits = torch.zeros(B, 4 * 35, device=dev)
        self.early = torch.zeros(B, 4, **boolean)
        self.side_bid = torch.zeros(B, 2, **boolean)
        # E45: frozen brl (pgx-observation) opponents, see experiments/brl/brl_player.py
        self.pgx = any(getattr(net, "is_brl", False) for net, _ in self.pool.values())
        if self.pgx:
            self.pgx_open = torch.zeros(B, 4, **boolean)        # passed before any bid, abs seat
            self.pgx_x = torch.zeros(B, 4 * 35, device=dev)     # abs seat doubled bid b
            self.pgx_xx = torch.zeros(B, 4 * 35, device=dev)    # abs seat redoubled bid b
            our_to_os = [(3 - c // 13) + (12 - c % 13) * 4 for c in range(52)]
            self.pgx_hand_idx = torch.tensor(our_to_os, device=dev)
            self.pgx_to_ours = torch.tensor([PASS, DOUBLE, REDOUBLE] + list(range(35)), device=dev)
        self.bad = torch.zeros((), **boolean)
        self.r = torch.zeros(1, **long)
        # per-round records (state before the call)
        self.rec_decide = torch.zeros(R, B, **boolean)
        self.rec_action = torch.zeros(R, B, **long)
        self.rec_last = torch.zeros(R, B, **long)
        self.rec_pass = torch.zeros(R, B, **long)
        self.rec_cseat = torch.zeros(R, B, **long)
        self.rec_doubled = torch.zeros(R, B, **boolean)
        self.rec_redoubled = torch.zeros(R, B, **boolean)
        self.rec_top = torch.zeros(R, B, **boolean)     # sampled call == greedy top call
        self.rec_fired = torch.zeros(R, B, **boolean)   # the SAC gate chose the call
        # constants
        self.ladder = torch.arange(35, device=dev)
        self.rel_order = torch.tensor(_REL_ORDER, device=dev)
        self.pass_only = F.one_hot(torch.tensor(PASS, device=dev), self.n_actions).bool()

    # ------------------------------------------------------------------ round
    def _round(self) -> None:
        B = self.B
        seat = (self.dealer + self.t) % 4
        side = seat % 2
        alive = ~self.ended
        forced = alive & (side == self.silent)
        pass_ends = (self.last >= 0) & (self.pass_count == 2)
        contracts = (self.ladder[None] > self.last[:, None]) & ~forced[:, None]
        parts = [contracts, torch.ones_like(contracts[:, :1])]
        if self.doubles:
            can = (alive & ~forced & (self.last >= 0) & ~self.doubled
                   & (self.contract_seat % 2 != side))
            if self.competitive and not self.any_seat:   # X only where Pass would end the auction
                can = can & pass_ends
            parts.append(can[:, None])
        if self.redouble:
            can_xx = (alive & ~forced & (pass_ends | self.any_seat) & self.doubled & ~self.redoubled
                      & (self.contract_seat % 2 == side))
            parts.append(can_xx[:, None])
        legal = torch.cat(parts, 1) & alive[:, None]

        order = (seat[:, None] + self.rel_order[None]) % 4                  # (B,4)
        bits = self.bits.view(B, 4, 35).gather(1, order[:, :, None].expand(B, 4, 35))
        feats = [bits[:, 0], bits[:, 1], self.early.gather(1, order[:, :2]).float(),
                 F.one_hot((self.dealer - seat) % 4, 4).float(),
                 self.vul.gather(1, side[:, None]).float(), bits[:, 2], bits[:, 3]]
        if self.doubles:
            feats += [self.doubled[:, None].float(),
                      (self.doubled & (self.doubler_side == side))[:, None].float()]
        if self.competitive:
            feats += [pass_ends[:, None].float(), self.redoubled[:, None].float()]
        feats = torch.cat(feats, 1)
        hand = self.hands.gather(1, seat[:, None, None].expand(B, 1, 52)).squeeze(1)

        decide = alive & ~forced
        action = torch.full_like(seat, PASS)
        ok_legal = legal
        if self.pool:
            frozen_rows = decide & (self.frozen_side == side)
            decide = decide & ~frozen_rows
            for code, (net, _) in self.pool.items():
                if getattr(net, "is_brl", False):
                    greedy, brl_legal = self._brl_act(net, hand, seat, side, alive)
                    rows = frozen_rows & (self.code == code)
                    action = torch.where(rows, greedy, action)
                    ok_legal = torch.where(rows[:, None], brl_legal, ok_legal)
                    continue
                width = AUCTION_FEATURES + getattr(net, "extra_features", 0)
                out = net(hand, feats[:, :width])
                if getattr(net, "stage", None) == COMPETITIVE_STAGE and self.competitive:
                    # frozen competitive nets play their full policy: X / XX / SAC included
                    greedy = competitive_log_probs(out, legal, 1.0, static=True).argmax(-1)
                else:
                    greedy = out["policy_logits"][:, :PASS + 1].masked_fill(
                        ~legal[:, :PASS + 1], -torch.inf).argmax(-1)
                action = torch.where(frozen_rows & (self.code == code), greedy, action)
        sample_legal = torch.where(decide[:, None], legal, self.pass_only[None])
        outputs = self.actor(hand, feats)
        if self.competitive:
            log_probs = competitive_log_probs(outputs, sample_legal, self.temperature, static=True)
        else:
            log_probs = policy_log_probs(outputs, sample_legal, self.temperature)
        r = self.r
        u = self.u.index_select(0, r).squeeze(0)
        chosen = inverse_cdf_sample(log_probs, u, sample_legal)
        action = torch.where(decide, chosen, action)
        if self.competitive:
            self.rec_top.index_copy_(0, r, (chosen == log_probs.argmax(-1))[None])
        if self.sacrifice:
            u_fire = self.u_fire.index_select(0, r).squeeze(0)
            fired = decide & sac_fire(outputs, sample_legal, chosen, u_fire, self.temperature)
            self.rec_fired.index_copy_(0, r, fired[None])

        self.rec_decide.index_copy_(0, r, decide[None])
        self.rec_action.index_copy_(0, r, action[None])
        self.rec_last.index_copy_(0, r, self.last[None])
        self.rec_pass.index_copy_(0, r, self.pass_count[None])
        self.rec_cseat.index_copy_(0, r, self.contract_seat[None])
        self.rec_doubled.index_copy_(0, r, self.doubled[None])
        if self.competitive:
            self.rec_redoubled.index_copy_(0, r, self.redoubled[None])
        ok = ok_legal.gather(1, action[:, None]).squeeze(1)
        self.bad.logical_or_(((alive & ~ok) | (alive & (self.t >= self.H))).any())

        # apply (alive rows only)
        bid = alive & (action < PASS)
        isx = alive & (action == DOUBLE)
        isxx = alive & (action == REDOUBLE)
        isp = alive & (action == PASS)
        pos = self.t.clamp(max=self.H - 1)[:, None]
        self.history.scatter_(1, pos, torch.where(alive[:, None], action[:, None],
                                                  self.history.gather(1, pos)))
        slot = (seat * 35 + action.clamp(max=34))[:, None]
        self.bits.scatter_(1, slot, torch.where(bid[:, None], 1.0, self.bits.gather(1, slot)))
        own_side_bid = self.side_bid.gather(1, side[:, None]).squeeze(1)
        self.early.scatter_(1, seat[:, None], self.early.gather(1, seat[:, None])
                            | (isp & ~own_side_bid)[:, None])
        self.side_bid.scatter_(1, side[:, None], (own_side_bid | bid)[:, None])
        if self.pgx:
            self.pgx_open.scatter_(1, seat[:, None], self.pgx_open.gather(1, seat[:, None])
                                   | (isp & (self.last < 0))[:, None])
            xslot = (seat * 35 + self.last.clamp(min=0))[:, None]
            self.pgx_x.scatter_(1, xslot, torch.where(isx[:, None], 1.0, self.pgx_x.gather(1, xslot)))
            self.pgx_xx.scatter_(1, xslot, torch.where(isxx[:, None], 1.0, self.pgx_xx.gather(1, xslot)))
        last = torch.where(bid, action, self.last)
        passes = torch.where(bid | isx | isxx, 0,
                             torch.where(isp, self.pass_count + 1, self.pass_count))
        self.contract_seat.copy_(torch.where(bid, seat, self.contract_seat))
        self.doubled.copy_(torch.where(bid, False, self.doubled | isx))
        self.redoubled.copy_(torch.where(bid, False, self.redoubled | isxx))
        self.doubler_side.copy_(torch.where(isx, side, self.doubler_side))
        self.ended.logical_or_(isp & (((last < 0) & (passes >= 4)) | ((last >= 0) & (passes >= 3))))
        self.last.copy_(last)
        self.pass_count.copy_(passes)
        self.t.add_(alive.long())
        self.r.add_(1)

    def pgx_observation(self, hand: torch.Tensor, seat: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        """(B,480) pgx.bridge_bidding observation for the player at ``seat`` (actor-relative)."""
        B = self.B
        rel = (seat[:, None] + torch.arange(4, device=seat.device)[None]) % 4   # 0 me, 1 LHO, 2 pd, 3 RHO
        me_vul = self.vul.gather(1, side[:, None]).squeeze(1)
        them_vul = self.vul.gather(1, (1 - side)[:, None]).squeeze(1)
        vul = torch.stack((~me_vul, me_vul, ~them_vul, them_vul), 1).float()
        opening = self.pgx_open.gather(1, rel).float()

        def per_bid(flat):                                    # (B,4*35) abs -> (B,35,4) rel
            return flat.view(B, 4, 35).gather(1, rel[:, :, None].expand(B, 4, 35)).transpose(1, 2)

        history = torch.cat((per_bid(self.bits), per_bid(self.pgx_x), per_bid(self.pgx_xx)), 2)
        cards = torch.zeros(B, 52, device=hand.device)
        cards[:, self.pgx_hand_idx] = hand
        return torch.cat((vul, opening, history.reshape(B, 420), cards), 1)

    def _brl_act(self, net, hand, seat, side, alive):
        """Greedy legal call of a brl net (ours ids) and its full pgx legality in our column order."""
        logits = net(self.pgx_observation(hand, seat, side))
        owner = torch.where(self.contract_seat >= 0, self.contract_seat % 2, -1)
        x_ok = (self.last >= 0) & ~self.doubled & (owner != side)
        xx_ok = self.doubled & ~self.redoubled & (owner == side)
        bids = self.ladder[None] > self.last[:, None]
        pgx_legal = torch.cat((torch.ones_like(x_ok)[:, None], x_ok[:, None], xx_ok[:, None], bids), 1)
        pick = logits.masked_fill(~pgx_legal, -torch.inf).argmax(-1)
        ours = torch.cat((bids, torch.ones_like(x_ok)[:, None], x_ok[:, None], xx_ok[:, None]), 1)
        return self.pgx_to_ours[pick], ours[:, :self.n_actions] & alive[:, None]

    def _reset(self) -> None:
        for buf, value in ((self.t, 0), (self.last, -1), (self.pass_count, 0),
                           (self.ended, False), (self.contract_seat, -1), (self.doubled, False),
                           (self.redoubled, False), (self.doubler_side, -1), (self.history, -1),
                           (self.bits, 0.0), (self.early, False), (self.side_bid, False),
                           (self.bad, False), (self.r, 0), (self.rec_decide, False),
                           (self.rec_top, False), (self.rec_fired, False)):
            buf.fill_(value)
        if self.pgx:
            for buf in (self.pgx_open, self.pgx_x, self.pgx_xx):
                buf.fill_(0)

    def _capture(self) -> None:
        self._reset()
        stream = torch.cuda.Stream(self.device)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.round()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.round()

    # ---------------------------------------------------------------- collect
    @torch.no_grad()
    def collect(self, deals: TorchDeals, generator: torch.Generator, scorer: TorchScorer,
                silent_frac: float = 0.25) -> FourSeatTrajectories:
        if not 0 <= silent_frac <= 1:
            raise ValueError("invalid silent_frac")
        B, R, dev = self.B, self.rounds, self.device
        was_training = self.actor.training
        self.actor.eval()
        deal, dealer, vul_ns, vul_ew, opponent, frozen_side, silent = episode_setup(
            deals, B, generator, silent_frac, dev, self.pool)
        u = torch.rand((R, B), generator=generator)
        u_fire = torch.rand((R, B), generator=generator) if self.sacrifice else None
        if self.cuda_graph and self.graph is None:
            self._capture()
        self._reset()
        self.deal.copy_(deal)
        self.dealer.copy_(dealer)
        self.vul.copy_(torch.stack((vul_ns.bool(), vul_ew.bool()), 1))
        self.silent.copy_(silent)
        self.code.copy_(opponent)
        self.frozen_side.copy_(frozen_side)
        self.hands.copy_(deals.hands[deal])
        self.u.copy_(u)
        if u_fire is not None:
            self.u_fire.copy_(u_fire)
        step = self.graph.replay if self.graph is not None else self.round
        rounds = 0
        while rounds < R:
            for _ in range(min(self.check_every, R - rounds)):
                step()
                rounds += 1
            if bool(self.ended.all()):
                break
        if bool(self.bad):
            raise ValueError("illegal four-seat call or auction longer than its history")
        if not bool(self.ended.all()):
            raise RuntimeError("four-seat auction did not terminate")
        self.actor.train(was_training)
        return self._trajectories(deals, scorer, rounds, deal, dealer, silent, opponent,
                                  frozen_side)

    def _trajectories(self, deals, scorer, rounds, deal, dealer, silent, opponent, frozen_side):
        cls = competitive_batch_class(self.redouble) if self.competitive else batch_class(self.doubles)
        decide = self.rec_decide[:rounds]
        t, rows = decide.nonzero(as_tuple=True)          # round-major, row-ascending: old order
        vul = self.vul.clone()
        history = self.history.clone()
        pos = torch.arange(self.H, device=self.device)[None]
        extra = {}
        term_extra = {}
        if self.doubles:
            extra = dict(contract_seat=self.rec_cseat[:rounds][decide],
                         doubled=self.rec_doubled[:rounds][decide])
            term_extra = dict(contract_seat=self.contract_seat.clone(),
                              doubled=self.doubled.clone())
        if self.competitive:
            extra["redoubled"] = self.rec_redoubled[:rounds][decide]
            term_extra["redoubled"] = self.redoubled.clone()
        states = cls(deal[rows], dealer[rows], vul[rows], silent[rows],
                     torch.where(pos < t[:, None], history[rows], torch.full_like(pos, -1)),
                     t, self.rec_last[:rounds][decide], self.rec_pass[:rounds][decide],
                     torch.zeros_like(rows, dtype=torch.bool), **extra)
        terminal = cls(deal, dealer.long(), vul, silent.long(), history, self.t.clone(),
                       self.last.clone(), self.pass_count.clone(), self.ended.clone(),
                       **term_extra)
        actions = self.rec_action[:rounds][decide]
        if self.competitive:
            top = torch.ones(self.B, self.H, dtype=torch.bool, device=self.device)
            top[rows, t] = self.rec_top[:rounds][decide]
            fired = self.rec_fired[:rounds][decide]
            return competitive_trajectories(deals, scorer, terminal, states, actions, rows, fired,
                                            top, opponent, frozen_side)
        group = rows * 2 + states.side
        present, dense = torch.unique(group, return_inverse=True)
        score, ceiling, _ = own_bid_scores(terminal, deals, scorer)
        if self.doubles:
            standing_x = terminal.standing_double_position()
            final_delta = double_delta(terminal, deals)
        else:
            standing_x = torch.full_like(deal, -1)
            final_delta = torch.zeros(self.B, device=self.device)
        return FourSeatTrajectories(states, actions, dense, score.reshape(-1)[present],
                                    ceiling.reshape(-1)[present], terminal, score, ceiling, rows,
                                    standing_x, final_delta, opponent, frozen_side)
