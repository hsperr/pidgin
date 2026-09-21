"""ONE NET, FOUR SEATS, and the auction runs until everyone passes.

exp10four.py gave each turn its own head, the way exp9.py does. At a horizon
of 8 that is seven heads over seats N E S W N E S W, and turn 7 -- the forced
Pass -- lands on West. Count the decisions: N 2, E 2, S 2, W 1. NS gets four
calls and EW three, so the +100 the dealing side scored there is partly a
language result and partly a rules bug. Worse, a fixed horizon hands the
board to whoever speaks last.

Three changes.

## The auction ends when everyone passes

Three passes after a bid or a double end it; four passes at the start pass
the board out for 0. Nothing else. Bids only ascend, so the auction is
bounded whatever anyone does -- 35 bids plus the passes between them -- and
--max-turns is a safety cap the code needs, not a rule anybody bids to.
Almost no auction reaches it.

## One network, sitting in every seat

The head already sees the auction ROTATED to its own seat: slot 0 is me, 1
LHO, 2 partner, 3 RHO, and it predicts MY SIDE's score rather than NS's. So
a seat is nothing but a position in the rotation, and one set of weights
answers for all four. Every deal then trains the net from four seats and
both partnerships, and there is no NS-versus-EW asymmetry left to measure.
It also fits an auction with no fixed length, which per-turn heads cannot.

exp9's docstring is right that a single net chasing its own next-position
guess oscillates. The fix is not seven networks, it is a target that holds
still: a FROZEN copy picks every call in every rollout, and the score is
read off the double-dummy table. --refresh copies the live net into it.
Within a refresh window nothing underneath you moves, which is the property
the per-turn heads were bought for. If the score wobbles anyway, exp10four
with a long horizon is the fallback.

## Par is now the real thing

exp10four reported a one-sided par: the best NS could do with EW silent.
That is a ceiling nobody can reach in a competitive auction and it is not
zero sum. Here par is the double-dummy PAR with doubles -- a minimax over
the 35 rungs, both sides free to outbid or to double and let it stand --
so `ns - par` is a real gap, and splitting it by sign says which side gave
the points away.

## The par baseline

--par-baseline regresses `score - par` instead of `score`. Par is a
constant of the deal, so it cannot change the order of two calls at a
state and therefore cannot change what the net plays -- it is a baseline,
exactly like the one in a policy gradient. What it removes is noise.
Par has a standard deviation of about 490 points per deal, and every bit
of that is variance no call is responsible for; RESULTS.md section 11
blames precisely this for making the trainer so sensitive to batch size
("the target is a wild sample of the conditional mean"). It also answers
the objection that a side is being rewarded for the opponents' blunders:
it still is, correctly, but now the number says how far from par the board
finished rather than how rich the deal happened to be.

Double stays; redouble is still out. --no-double is an ablation and a bad
idea: without it a sacrifice is cheaper than any game, so bidding over the
opponents is free and the auction escalates toward 7NT.

## Both pairs learn the same amount

Each seat's target is its OWN side's final score, and the opponents' calls in
every rollout come from the frozen net. So the two partnerships are
independent learners: each optimises as if it were alone, against an opponent
that happens to improve as the net improves. No zero-sum machinery, no
always-pass curriculum. That much was already true here.

What was not true is that they learned equally. Even turns are NS and odd
turns EW, turn 0 is always trained, and the other slots were drawn from one
pool of turns 1..11 -- which handed NS 2.8 trained decisions per step against
EW's 2.2. Half the slots now come from each side. Keep --turns-per-step even.

## --window: a practice auction somebody could hold

The random half of state sampling used to draw flat over all 35 rungs, so
North opened 6D one time in thirty-five and by turn 8 both of its own calls
were nonsense. Section 12 already measured the two-player version of this,
and four seats multiply it: three other seats each carry an epsilon floor, and
unlike the cooperative game an absurd opponent bid changes WHO declares, so
the same call's target swings by a thousand points. --window 6 draws a random
call from Pass, Double and the next six rungs only. A greedy call is never
windowed.

Why the two-player trainer never showed this: horizon 4 has three calls, so at
most one own call and one partner call sit above any trained state; and at
turn 1 exp9 ENUMERATES all 35 openings and weights each by how likely A is to
bid it, so a 6D opening carries 0.6% of the weight instead of 1/35 of the
samples.

## --own-bid: four seats, and nobody is competing

"Each side maximises its own score minus its own par" is the same model as
the zero-sum game -- EW's score IS minus NS's, and par is a per-deal
constant, so the argmax never moves. The rule that is genuinely different:

    Score each side on ITS OWN highest bid, as if that side declared it,
    and ignore who actually won the auction.

Outbidding the opponents now wins nothing, because you are not scored on
taking the board away from them. What interference still costs is real:
EW's 4S means NS cannot stop below 5, so NS's reachable contracts shrink.
The pressure is auction room, not points off the opponents.

It also kills the variance that makes the competitive runs flail. There an
absurd opponent bid changes WHO declares and the same call's target swings
by a thousand points and flips sign; here a side's target depends only on
its own bids and the room it had, so nothing flips. Doubles are meaningless
under it -- you cannot double yourself and the opponents' double is not
read -- so --own-bid turns them off. The baseline becomes the side's own
cooperative par. Yardsticks on 100k boards: WBridge5 97.8, two-player 111.3.

## --pass-cost: what silence costs, added to --own-bid

--own-bid works because its reward is ATTRIBUTABLE: a side's score depends
only on its own calls, so an informative bid shows up in its own number on
every board. Grade the same bid through three other seats and one shared
contract and the net either goes silent (measured: 88.1% of boards passed
out from a cold zero-sum start) or escalates into a doubling ladder. What
--own-bid gets wrong is not the bidder's branch, it is the silent one:
saying nothing while the opponents make 4S costs exactly 0.

--pass-cost LAMBDA charges for that, and changes nothing else:

    A side that BID is scored on its own highest bid, undoubled, exactly as
    --own-bid does today. A side that NEVER BID is charged LAMBDA times the
    opponents' ACTUAL final score, from their point of view, doubled if it
    was doubled. A passed-out board is still 0 for both.

Never par: par is privileged information about the deal and this project
keeps it out of the players' reward. The opponents' real result is not --
it is the thing that happened at the table.

This buys the sacrifice incentive for free: with a weak hand, -100 in your
own contract beats letting them have +420. The fiction it KEEPS on purpose
is that being outbid still pays you for your own bid, because that fiction
is why phase 1 is stable; LAMBDA exists so the price can be annealed. 0 is
off and bit-identical to plain --own-bid, 1 is the full price. --own-bid
masks Double out, so no double can occur under it today; the pass term
reads the doubled score through `ns_score` anyway, so it is already correct
if doubles are ever turned on alongside it.

## --winner-only: only the contract that won gets played

--own-bid pays both sides for their own highest bid. That is false about
bridge -- one contract survives an auction, the other side's best idea is
never played -- and it is why Double has no meaning there and stays masked.
--winner-only keeps everything else about own-bid and fixes just that:

    The side that WON the auction scores its real contract, doubled if it
    was doubled. The side that lost it scores --loser-mode, not minus the
    winner's score.

The loser term is the whole design. Zero-sum makes Pass cost 420 on a board
where own-bid makes it cost 0, and that shock is what the competitive runs
died of (RESULTS 18), so the loser is paid on a dial instead:

    zero            (default) nothing. Letting the opponents play is free,
                    so nothing is learned about defence, and nothing about
                    Pass is disturbed either.
    dbl             the double, and only the double: (undoubled score) -
                    (doubled score), zero when nobody doubled, positive when
                    the double beat a contract that failed. This is the one
                    that gives the Double head a gradient at all.
    potential       the winner's cooperative par minus what it actually
                    made: the points the board owed them and they missed.
                    Teaches pushing the opponents past what they can make.
    potential-clip  the same, floored at 0. Then a double that feeds a
                    making contract is merely worthless rather than
                    punished -- and the difference between the two numbers
                    is exactly how much damage the greedy doubles do.
    zerosum         minus the winner's score, i.e. the real game. It is here
                    so the new code can be checked against ns_score, not
                    because it is a curriculum.

--loser-scale multiplies that term, so the dial is continuous as well as
discrete. Doubles are ON unless --no-double: with a loser term there is
something for the Double head to learn, and a warm start from an own-bid
checkpoint brings an untrained one -- see --dbl-bias.

## --huber, --clip-target, --dbl-head: the loss, not the reward

W1 (`--winner-only --loser-mode dbl --dbl-bias -5`, warm from a +113 phase-1
net) doubled 99.8% of contracts by step 250, mse 420 against a normal ~20.
The obvious reading -- "the losing side's only income is the double, so of
course it doubles" -- is wrong, and `emergent/screen0.py` measures it wrong:
over ~1M rolled-out calls from that same net, P(Double is the argmax where
Double is legal) is between 0.7% and 3.1% under EVERY scoring rule, own-bid
included. **No target asks for a doubling collapse. The training loop makes
one.** The suspected loop: the net doubles slightly more, so more rolled-out
auctions end doubled; a doubled 13-down contract is -3500 where undoubled it
is -350; squared error chases those; and the frozen rollout copy is refreshed
into the same habit every 100 steps. The same screen put the per-state target
spread at 1345 points with Double legal against own-bid's 620, sd 465 against
236.

So three flags that attack the loop rather than the reward. All default off,
and an unflagged run is bit-identical to what it was.

    --huber D       pay for an error with Huber instead of squared error,
                    elbow at D. Same masked, reach-weighted, same-normalised
                    loss on the same quantity -- only the per-entry penalty
                    changes. D is in the loss's units, POINTS/100, because
                    the target is divided by 100; 3 is a 300-point elbow.
    --clip-target P clamp the baselined target to +-P POINTS before the loss
                    sees it. Bounds WHAT is regressed where Huber bounds what
                    an error COSTS. Independent, and they combine.
    --dbl-head      ticket 006. Double gets its own parameters instead of
                    sharing the output layer with the bids, and the gradient
                    is stopped where that head meets the trunk. One argmax
                    over all L+2 calls, as before: separate WEIGHTS, not a
                    second decision.

## --dbl-ends: a Double is the last call. NOT BRIDGE.

Every arm with Double legal escalates into a doubling ladder -- bid, double,
bid higher, double, bid higher. W1d (`--dbl-head`, which DID protect the
language: bits 2.04 against phase 1's 1.73) still finished at 89.9% of
contracts doubled, 76.4% slams, and auctions 23.0 calls long against phase
1's 12.5. The head fixed Double corrupting the language; it did not stop the
escalation, because the bidding side can always escape a double by bidding
one more rung.

--dbl-ends takes that escape away: a Double ENDS the auction where it is
made, and the contract is the standing rung, doubled. The bidding side cannot
run, so every bid has to be one it is willing to have doubled, and the
pressure against overbidding comes from the OPPONENTS rather than from a
tuning constant. Rollouts get shorter too, which is why the -3500 doubled
disasters get rarer and the targets calmer.

Nothing else changes. The finished state scores through `ns_score` /
`own_score` / `winner_score` untouched, `declarer_of` still reads the first
seat of the bidding side to name the strain, the legal mask for Double is
unchanged (only the side that did not make the standing bid, no redouble),
and a passed-out board is unaffected.

**Running from a double is legal in the real game, so this is a training
wheel and it must come off.** It is fine-tuning only, it is off by default,
an unflagged run is bit-identical, and any result measured under it has to be
re-checked without it. The rule is saved in the checkpoint config, so
`wb5.py`, `showfour.py` and `screen0.py` replay under it automatically and
`bbo.py` -- where the opponent is a real robot that will bid over a double --
refuses the checkpoint outright.

## --init: warm start

--own-bid is a pretraining stage, so the competitive run needs to start
from its weights. --init loads the net out of a checkpoint this script
saved; the optimizer starts fresh.

## --ew-pass: the gate that has to pass first

East and West may only Pass. NS then bids alone against silence, which is
exactly exp9.py --open-pass, and the score must land near its 110 on the
35-rung ladder. If it does not, the four-seat plumbing is wrong and every
competitive number measured afterwards is worthless. Read `ns` against 110
during the gate and ignore `gap`: the printed par is the COMPETITIVE minimax
par, which is not the right denominator for a game the opponents sit out.
It is a test, not a pretraining stage -- nothing needs it as a warm start.
"""
import argparse, copy, json, os, random, time

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent.exp1 import FULL_CONTRACTS
from emergent.exp3q import free_cache, _bits
from emergent.fullinfo import SuitEncoder
from emergent.scoring import contract_strain_dd
from emergent.scoring4 import build_score_table4, contract_levels, contract_strains
from emergent.exp10four import (St, declarer_of, load_deals4, ns_score)
from emergent.exp10four import ends_mask as _ends4
from emergent.exp10four import legal_mask as _legal4

NEG_INF = -1e9
SEATS = "NESW"


def apply_call_(st, call, seat, L, dbl_ends=False):
    """apply_call, but writing into the state instead of copying it.

    The two history grids are (N, 35, 4) floats. At 113k rollout rows that
    is 63 MB each, and exp10four.apply_call clones both on every turn of
    every rollout -- gigabytes of memcpy per step, which is what the
    trainer was actually spending its time on rather than arithmetic.

    Only safe where the caller owns the state exclusively: a rollout fork,
    a freshly sampled prefix, an evaluation auction. Never on a state that
    something else still holds a reference to.

    `dbl_ends` is ticket 005's training wheel and NOT bridge: a Double ends
    the auction where it is made, so the bidding side cannot run from it.
    Everything else about the finished row is unchanged -- the standing rung
    is still `last`, `lastseat` still fixes the side, `dblflag` is still 1 --
    so it scores through exactly the same code as an auction that reached
    the same contract by three passes.
    """
    n, dev = st.n, st.last.device
    rows = torch.arange(n, device=dev)
    act = st.alive
    is_bid = act & (call < L)
    is_pass = act & (call == L)
    is_dbl = act & (call == L + 1)
    ci = call.clamp(max=L - 1)
    lc = st.last.clamp(min=0)
    st.bid[rows, ci, seat] = torch.where(is_bid, torch.ones(n, device=dev),
                                        st.bid[rows, ci, seat])
    st.dbl[rows, lc, seat] = torch.where(is_dbl, torch.ones(n, device=dev),
                                         st.dbl[rows, lc, seat])
    st.last = torch.where(is_bid, ci, st.last)
    st.lastseat = torch.where(is_bid, torch.full_like(st.lastseat, seat),
                              st.lastseat)
    st.dblflag = torch.where(is_bid, torch.zeros_like(st.dblflag),
                             torch.where(is_dbl, torch.ones_like(st.dblflag),
                                         st.dblflag))
    st.npass = torch.where(is_pass, st.npass + 1,
                           torch.where(act, torch.zeros_like(st.npass),
                                       st.npass))
    over = is_pass & (((st.npass >= 3) & (st.last >= 0)) | (st.npass >= 4))
    if dbl_ends:
        over = over | is_dbl
    st.alive = st.alive & ~over
    return st


def ends_mask(st, L, dbl_ends=False):
    """(N, L+2). Which calls would END the auction here.

    exp10four's rule -- only a Pass can -- plus `dbl_ends`, under which a
    Double always does. `legal_mask` still decides WHETHER Double may be
    called; this only says what happens if it is. `sampled_states` reads
    this to strike out prefixes that could not have been reached, so it has
    to agree with `apply_call_` exactly or a dead state gets a live target.
    """
    m = _ends4(st, L)
    if dbl_ends:
        m[:, L + 1] = True
    return m


def legal_mask(st, seat, L, doubles=True, ew_pass=False):
    """The calls `seat` may make. `ew_pass` gags East and West completely.

    Forcing the MASK to Pass-only is all it takes to sit a silent pair at the
    table: every argmax and every sample in this file reads its options from
    here, so nothing else has to know about it.
    """
    m = _legal4(st, seat, L)
    if not doubles:
        m = m.clone()
        m[:, L + 1] = False
    if ew_pass and seat % 2 == 1:
        m = torch.zeros_like(m)
        m[:, L] = True
    return m


def window_mask(st, L, window):
    """(N, L+2). Bids within `window` rungs above the standing bid, plus Pass
    and Double.

    Only the RANDOM part of state sampling is windowed, never the greedy pick.
    A practice auction is meant to be one somebody could hold: drawn flat over
    all 35 rungs, North opens 6D one time in thirty-five, and by turn 8 both of
    its calls are nonsense. exp9.py already had this idea as --prefix-beta,
    which weighted random prefixes toward the cheap end because that is where
    auctions live. This is the same prior, expressed as a hard window: with
    window=6 and nothing bid yet the openings are 1C..2C, which is the range a
    real opening comes from.

    window <= 0 or >= L turns it off, which is the old behaviour.
    """
    dev = st.last.device
    if window <= 0 or window >= L:
        return torch.ones(st.n, L + 2, dtype=torch.bool, device=dev)
    rungs = torch.arange(L, device=dev)[None, :]
    near = (rungs > st.last[:, None]) & (rungs <= st.last[:, None] + window)
    return torch.cat([near, torch.ones(st.n, 2, dtype=torch.bool, device=dev)], 1)


# ------------------------------------------------- the non-competing score

def own_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
              seat_map=None):
    """(N,) what `side` scores on ITS OWN highest bid, as if it declared it.

    Who won the auction is not read at all, so both sides can be scored on
    the same board and neither is paid for taking it away from the other.
    Bids ascend, so a side's highest bid is simply the last one it made, and
    its declarer is the first of its two seats to have named that strain --
    the same rule as `declarer_of`, applied to the given side instead of the
    winning one. Undoubled, because a double is not part of this game.

    `seat_map` is the rotated evaluation's table-seat -> dd-seat map, as in
    `ns_score`. A side that never bid scores 0, which is also the passed-out
    board.
    """
    n, dev = st.n, st.last.device
    b1, b2 = st.bid[:, :, side], st.bid[:, :, side + 2]
    mine = (b1 + b2) > 0                                     # (n, L)
    has = mine.any(1)
    rungs = torch.arange(L, device=dev)[None, :].expand(n, L)
    c = torch.where(mine, rungs, torch.full_like(rungs, -1)).max(1).values
    c = c.clamp(min=0)
    cand = mine & (strain_si[None, :] == strain_si[c][:, None])
    k0 = torch.where(cand, rungs, torch.full_like(rungs, L)).min(1).values
    decl = torch.where(b1.gather(1, k0.clamp(max=L - 1)[:, None]).squeeze(1) > 0,
                       torch.full_like(st.last, side),
                       torch.full_like(st.last, side + 2))
    dd_seat = decl if seat_map is None else seat_map[decl]
    tr = tricks_all[st.deal, dd_seat, strain_dd[c]].long()
    raw = tbl4[c, 0, tr]
    return torch.where(has, raw, torch.zeros_like(raw))


def pass_cost_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                    pass_cost, seat_map=None):
    """(N,) `own_score` for a side that bid; a price for one that never did.

    Phase 1 works because its reward is ATTRIBUTABLE: a side's score depends
    only on its own calls, so an informative bid shows up in its own number
    on every board. Grade the same bid through three other seats and one
    shared contract and the net either goes silent (measured: 88.1% of
    boards passed out from a cold zero-sum start) or escalates into a
    doubling ladder. This rule keeps the attributable branch for anybody who
    bid, and charges silence an honest, deal-dependent price:

      bid at all   `own_score(side)`, unchanged -- its own highest bid, as
                   if it declared it, undoubled, same declarer rule. Being
                   outbid still pays you for your own bid.
      never bid    `-pass_cost x` the opponents' ACTUAL final score, from
                   their point of view, doubled if it was doubled. Never
                   par: par is privileged information about the deal and
                   this project keeps it out of the players' reward.
      passed out   0 for both sides, as today. Nobody scored anything.

    The sacrifice incentive comes for free: holding a weak hand, -100 in
    your own contract beats letting them have +420. The fiction it keeps --
    that being outbid still pays you -- is deliberate, because that fiction
    is why phase 1 is stable; `pass_cost` is a lambda precisely so it can be
    annealed away later.

    `pass_cost` 0 is `own_score` row for row. `seat_map` is the rotated
    evaluation's table-seat -> dd-seat map, as in `ns_score`. The opponents'
    term reads `st.dblflag` through `ns_score`, so it is already right if
    doubles are ever legal alongside this; a side scored on its own bid
    stays undoubled either way.
    """
    own = own_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                    seat_map)
    bid = (st.bid[:, :, side].sum(1) + st.bid[:, :, side + 2].sum(1)) > 0
    # ns_score is signed for NS, so +1 for side 0 and -1 for side 1 turns
    # "the opponents did well" into a big NEGATIVE number for this side.
    sgn = 1.0 if side == 0 else -1.0
    opp = sgn * pass_cost * ns_score(st, tricks_all, tbl4, strain_dd,
                                     strain_si, L, seat_map)
    return torch.where(bid, own, opp)


def call_kind(st, call, L, dbl_ends=False):
    """(N,) what KIND of decision each call is, read BEFORE it is applied.

    0 a bid, 1 a pass that leaves the auction alive, 2 a pass that ENDS it,
    3 a double. `--route-blame` grades a row by this and nothing else, so it
    has to be computed on the pre-call state: once `apply_call_` has run,
    `npass` no longer says whether this call was the one that closed the
    auction.
    """
    ends = ends_mask(st, L, dbl_ends).gather(1, call[:, None]).squeeze(1)
    k = torch.zeros_like(call)
    is_pass = call == L
    k = torch.where(is_pass & ends, torch.full_like(k, 2),
                    torch.where(is_pass, torch.full_like(k, 1), k))
    k = torch.where(call == L + 1, torch.full_like(k, 3), k)
    return k


def own_score_dbl(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                  seat_map=None):
    """`own_score`, but priced DOUBLED when the board really doubled us.

    `own_score`'s fiction is that your highest bid is your contract, always
    undoubled. Under `--route-blame` a bid is graded on that fiction, so a
    sacrifice looks free: bid 5H over their 4S and you are charged 150 for
    three down when the table would have charged you 500. That is the one
    hole in routing the blame, and it is cheap to close -- if your side in
    fact won the auction AND the contract standing at the end is in fact
    doubled, read the doubled column. If your side was outbid, your highest
    bid was never doubled and the undoubled column is the honest one.
    """
    n, dev = st.n, st.last.device
    b1, b2 = st.bid[:, :, side], st.bid[:, :, side + 2]
    mine = (b1 + b2) > 0
    has = mine.any(1)
    rungs = torch.arange(L, device=dev)[None, :].expand(n, L)
    c = torch.where(mine, rungs, torch.full_like(rungs, -1)).max(1).values
    c = c.clamp(min=0)
    cand = mine & (strain_si[None, :] == strain_si[c][:, None])
    k0 = torch.where(cand, rungs, torch.full_like(rungs, L)).min(1).values
    decl = torch.where(b1.gather(1, k0.clamp(max=L - 1)[:, None]).squeeze(1) > 0,
                       torch.full_like(st.last, side),
                       torch.full_like(st.last, side + 2))
    dd_seat = decl if seat_map is None else seat_map[decl]
    tr = tricks_all[st.deal, dd_seat, strain_dd[c]].long()
    # ours only if the final standing rung is our own highest bid and the
    # last bidder was on our side; otherwise they own the contract.
    ours = (st.last == c) & (((st.lastseat - side) % 2) == 0) & (st.last >= 0)
    dbl = torch.where(ours, st.dblflag, torch.zeros_like(st.dblflag))
    raw = tbl4[c, dbl, tr]
    return torch.where(has, raw, torch.zeros_like(raw))


def route_blame_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                      kind, scale=1.0, own_dbl=True, seat_map=None):
    """(N,) the target, ROUTED BY WHICH DECISION THE CALL WAS.

    The user's rule, 2026-09-11. An opening bid did not choose the doubled
    disaster three calls later; the player who accepted that contract did.
    So an early, constructive call is graded the way phase 1 grades it -- on
    its own side's contract, which is attributable and is the only reward in
    this project that has ever grown a language. The call that CLOSES the
    auction, and the call that doubles, are graded on the real zero-sum
    score, because those two are the calls that actually chose it.

      bid, or a pass that leaves the auction alive   own contract
      a pass that ends the auction                   real zero-sum score
      a double                                       real zero-sum score

    There is a second reason beyond fairness, and it is the stronger one.
    The zero-sum number is hardest to fit at turn 0, where the contract, the
    strain and the declarer are all unknown, and easiest at the last call,
    where only the trick count is open. Routing it to the terminal decisions
    puts the noisy regression exactly where the state has already pinned it
    down.

    The cost, which is real and must be watched: the value head is no longer
    self-consistent. Turn 0 says +420 on the fiction while the board pays
    -200 after they double, so bidding is systematically overrated and the
    auction may creep upward. `scale` is the dial -- 0 is phase 1 row for
    row, 1 is the full zero-sum price on the terminal calls -- so it can be
    annealed. Watch the 5-level and slam shares against WBridge5's 8.47% and
    7.74%, and watch `own`.
    """
    own_fn = own_score_dbl if own_dbl else own_score
    own = own_fn(st, side, tricks_all, tbl4, strain_dd, strain_si, L, seat_map)
    sgn = 1.0 if side == 0 else -1.0
    zs = sgn * ns_score(st, tricks_all, tbl4, strain_dd, strain_si, L, seat_map)
    terminal = (kind == 2) | (kind == 3)
    return torch.where(terminal, own + scale * (zs - own), own)


def ns_score_undoubled(st, tricks_all, tbl4, strain_dd, strain_si, L,
                       seat_map=None):
    """`ns_score` with the double struck off: what the board would have paid
    had nobody doubled. The decomposition needs it to isolate what the double
    itself was worth, which is the whole of `Q_double`'s target."""
    has = st.last >= 0
    c = st.last.clamp(min=0)
    decl = declarer_of(st, strain_si, L)
    dd_seat = decl if seat_map is None else seat_map[decl]
    tr = tricks_all[st.deal, dd_seat, strain_dd[c]].long()
    raw = tbl4[c, torch.zeros_like(st.dblflag), tr]
    signed = torch.where((decl % 2) == 0, raw, -raw)
    return torch.where(has, signed, torch.zeros_like(signed))


def triple_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                 seat_map=None):
    """(N, 3): the real zero-sum score, split into three things a call does.

    Ticket 008. One rollout, three cheap reads, and the split is EXACT --
    the three columns sum to the real signed score of the board, row for
    row, so nothing is invented and nothing is lost:

      own     what our own highest bid is worth, undoubled. This is
              `own_score`, i.e. phase 1, the only reward in this project
              that has ever grown a language.
      denial  the real undoubled score minus that. Everything the opponents'
              contract does to us, and everything a sacrifice saves: the
              zero-sum part, isolated.
      double  the real score minus the real UNDOUBLED score. Exactly what
              the double was worth and nothing else.

    Each column is far less noisy than the total, and each gets its own head
    and its own weight, so the curriculum becomes two coefficients rather
    than a sequence of different games.
    """
    own = own_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                    seat_map)
    sgn = 1.0 if side == 0 else -1.0
    zs = sgn * ns_score(st, tricks_all, tbl4, strain_dd, strain_si, L,
                        seat_map)
    zs_u = sgn * ns_score_undoubled(st, tricks_all, tbl4, strain_dd,
                                    strain_si, L, seat_map)
    return torch.stack([own, zs_u - own, zs - zs_u], -1)


LOSER_MODES = ("zero", "dbl", "potential", "potential-clip", "zerosum")


def winner_score(st, side, tricks_all, tbl4, strain_dd, strain_si, L,
                 loser_mode, loser_scale, own_par_ns, own_par_ew,
                 seat_map=None):
    """(N,) what `side` scores when only the winning contract is played.

    An auction leaves ONE contract standing, so own_score's fiction -- both
    sides paid for their own highest bid -- is the thing to drop. The winner
    gets its real score, doubled if it was doubled. What the loser gets is
    the whole experiment, because it is the only place a Double can be worth
    anything to the side that made it:

      zero            nothing. Letting the opponents play costs 0, so Pass
                      keeps the meaning it had under --own-bid.
      dbl             undoubled - real, which is 0 unless somebody doubled
                      and positive exactly when the double beat a failing
                      contract. That is what Double is worth and nothing
                      else.
      potential       the winner's own cooperative par minus what it
                      actually scored: what the board was worth to them and
                      did not get. Teaches pushing them past their par.
      potential-clip  the same, floored at 0, so feeding a making contract
                      is free rather than punished.
      zerosum         -real, the real game. Here as a check: it must
                      reproduce ns_score row for row.

    `loser_scale` multiplies the loser term only. The score is signed for
    `side` and not for NS, so the two sides can be read off the same board.
    `seat_map` is the rotated evaluation's table-seat -> dd-seat map, as in
    `ns_score`; the cooperative pars are per dd-side, so declarer's DD seat,
    not its table seat, picks which one is the winner's. A passed-out board
    is 0 for both.
    """
    has = st.last >= 0
    c = st.last.clamp(min=0)
    decl = declarer_of(st, strain_si, L)
    dd_seat = decl if seat_map is None else seat_map[decl]
    tr = tricks_all[st.deal, dd_seat, strain_dd[c]].long()
    real = tbl4[c, st.dblflag, tr]
    if loser_mode == "zero":
        lose = torch.zeros_like(real)
    elif loser_mode == "dbl":
        lose = tbl4[c, 0, tr] - real
    elif loser_mode == "zerosum":
        lose = -real
    else:
        par_w = torch.where((dd_seat % 2) == 0, own_par_ns[st.deal],
                            own_par_ew[st.deal])
        lose = par_w - real
        if loser_mode == "potential-clip":
            lose = lose.clamp(min=0)
    r = torch.where((decl % 2) == side, real, loser_scale * lose)
    return torch.where(has, r, torch.zeros_like(r))


def coop_par_side(tricks_all, tbl4, strain_dd, side, chunk=100000):
    """(N,) the best undoubled score `side` could reach with the ladder to
    itself, from either of its two seats: the baseline that goes with
    `own_score`. exp10four.coop_par is this for NS. Chunked so the (n, 35)
    scratch stays small over a million deals.
    """
    dev = tbl4.device
    L = tbl4.shape[0]
    rung = torch.arange(L, device=dev)[None, :]
    out = []
    for i in range(0, tricks_all.shape[0], chunk):
        tr = tricks_all[i:i + chunk].long()
        best = torch.zeros(tr.shape[0], device=dev)
        for seat in (side, side + 2):
            s = tbl4[rung, 0, tr[:, seat, :][:, strain_dd]]   # (n, L)
            best = torch.maximum(best, s.max(1).values)
        out.append(best.clamp(min=0))
    return torch.cat(out)


# ------------------------------------------------------------------- par

def par_all(tricks_all, tbl4, strain_dd, chunk=100000):
    """par_table over the whole file, in slices, so the (n, 35) scratch
    tensors stay small."""
    return torch.cat([par_table(tricks_all[i:i + chunk], tbl4, strain_dd)
                      for i in range(0, tricks_all.shape[0], chunk)])


def par_table(tricks_all, tbl4, strain_dd):
    """(N,) the double-dummy PAR of each deal, as an NS score.

    A minimax over the ladder. Because bids only ascend, the whole auction
    collapses to 35 states: "rung k stands, owned by this side". The side
    to move either lets it stand -- choosing whether to double, which it
    only does when that helps -- or outbids it, which moves to a strictly
    higher rung, so the recursion is a single sweep from 7NT downwards.

        A[k]  rung k stands for NS, EW to move (EW minimises the NS score)
        B[k]  rung k stands for EW, NS to move (NS maximises)

        A[k] = min( stand_ns[k], min over k' > k of B[k'] )
        B[k] = max( stand_ew[k], max over k' > k of A[k'] )

    and the empty auction is worth max( min(0, min B), max A ): open, or
    pass and let the opponents open into a board that is already 0 if they
    pass too. Declarer for a side is the better of its two seats, which is
    what a par calculation always assumes.
    """
    dev = tbl4.device
    n, L = tricks_all.shape[0], tbl4.shape[0]
    tr = tricks_all[:, :, strain_dd].long()                 # (n, 4, L)
    rung = torch.arange(L, device=dev)

    def side_score(seats, d):
        vals = [tbl4[rung[None, :], d, tr[:, s, :]] for s in seats]
        return torch.maximum(*vals)                          # (n, L)

    ns0, ns1 = side_score((0, 2), 0), side_score((0, 2), 1)
    ew0, ew1 = side_score((1, 3), 0), side_score((1, 3), 1)
    stand_ns = torch.minimum(ns0, ns1)          # EW picks the double or not
    stand_ew = -torch.minimum(ew0, ew1)         # NS picks, as an NS score

    A = torch.empty(n, L, device=dev)
    B = torch.empty(n, L, device=dev)
    minB = torch.full((n,), float("inf"), device=dev)
    maxA = torch.full((n,), float("-inf"), device=dev)
    for k in range(L - 1, -1, -1):
        A[:, k] = torch.minimum(stand_ns[:, k], minB)
        B[:, k] = torch.maximum(stand_ew[:, k], maxA)
        minB = torch.minimum(minB, B[:, k])
        maxA = torch.maximum(maxA, A[:, k])
    return torch.maximum(torch.minimum(torch.zeros(n, device=dev), minB), maxA)


# ------------------------------------------------------------------- net

def clip_target(tgt, points):
    """Clamp the baselined target to +-`points`. 0 leaves it alone.

    In POINTS, before the loss's /100, so the number reads next to the
    1345-point per-state spread the level-0 screen measured with Double
    legal (own-bid: 620). This bounds WHAT is regressed. `q_err` bounds
    what an error COSTS. They are independent and they combine.
    """
    return tgt if points <= 0 else tgt.clamp(-points, points)


def q_err(d, huber=0.0):
    """Per-entry penalty for a prediction error `d`, in target units.

    Squared error by default, which is what every run so far used. With
    `huber` > 0, Huber with that elbow -- scaled by 2 because torch's Huber
    is 0.5*x^2 inside the elbow where this loss is x^2, so a large delta IS
    the squared error it replaces, to float error. The mask, the reach
    weighting and the normalisation live in the caller and do not change.
    """
    if huber <= 0:
        return d ** 2
    return 2.0 * F.huber_loss(d, torch.zeros_like(d), reduction="none",
                              delta=huber)


class SeatNet(nn.Module):
    """(my hand, the auction rotated to me) -> my side's score for each call.

    No seat identity: the only thing that tells one chair from another is
    which slot of the rotated history is mine, plus the turn index, which
    says whether I am the dealer or the fourth to speak.
    """
    def __init__(self, L, hidden=512, d_hand=64, d_rung=48, layers=3,
                 dbl_head=False, triple=False, triple_attached=False,
                 w_denial=1.0, w_double=1.0):
        super().__init__()
        self.L, self.d = L, d_rung
        self.hand_enc = SuitEncoder(d_hand)
        self.rung = nn.Embedding(L, d_rung)
        self.no_bid = nn.Parameter(torch.zeros(d_rung))
        self.pass_key = nn.Parameter(torch.randn(d_rung) * 0.1)
        self.dbl_key = nn.Parameter(torch.randn(d_rung) * 0.1)
        d_in = self.hand_enc.out_dim + 9 * d_rung + 6
        mods, d = [], d_in
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.proj = nn.Linear(d, d_rung)
        self.bias = nn.Parameter(torch.zeros(L + 2))
        # --dbl-head (ticket 006). Built LAST so that a net without it draws
        # exactly the random numbers it drew before, and an unflagged run
        # stays bit-identical.
        self.dbl_head = None
        if dbl_head:
            self.dbl_head = nn.Sequential(nn.Linear(d, d_rung), nn.ReLU(),
                                          nn.Linear(d_rung, 1))
        # --triple-q (ticket 008). Two more read-outs of the same trunk, one
        # per extra term of the decomposition. Built LAST, after --dbl-head,
        # for the same reason: a net without them draws exactly the random
        # numbers it drew before.
        self.den_head = self.dblq_head = None
        self.triple_attached = triple_attached
        self.w_denial, self.w_double = w_denial, w_double
        if triple:
            mk = lambda: nn.Sequential(nn.Linear(d, d_rung), nn.ReLU(),
                                       nn.Linear(d_rung, L + 2))
            self.den_head, self.dblq_head = mk(), mk()

    def _rung_map(self, seat):
        """(4L, 4d): the rotation of `_sums` folded into the rung table.

        `_sums` used to build its (n, L, 4) rotated view by advanced-indexing
        the seat axis and then permuting it, which copies the whole history
        grid twice per call -- 70 MB a call at rollout width, and the two
        grids are read forty times an auction. The same numbers come out of
        one matmul against a block matrix that carries the rotation, with the
        history read exactly as it already sits in memory: block (k, c) holds
        the rung embedding when seat c is relative slot j, and zero
        elsewhere, so column block j sums the rungs bid by seat (seat+j)%4.
        Adding an exact zero is exact, so this is the same arithmetic.
        """
        W = self.rung.weight
        big = W.new_zeros(self.L * 4, 4 * self.d)
        for j in range(4):
            big[(seat + j) % 4::4, j * self.d:(j + 1) * self.d] = W
        return big

    def _sums(self, h, seat, rmap=None):
        n = h.shape[0]
        if rmap is None:
            rmap = self._rung_map(seat)
        return h.reshape(n, self.L * 4) @ rmap

    def _trunk(self, hand, st, seat, t, hv=None):
        """`hv` is a precomputed `hand_enc(hand)`; then `hand` is not read."""
        dev = st.last.device
        n = st.n
        if hv is None:
            hv = self.hand_enc(hand)
        lastv = torch.where(st.last[:, None] >= 0,
                            self.rung(st.last.clamp(min=0)),
                            self.no_bid[None].expand(n, self.d))
        mine = ((st.lastseat >= 0) & ((st.lastseat % 2) == (seat % 2))).float()
        scal = torch.stack([torch.full((n,), t / 10.0, device=dev),
                            st.last.float() / self.L,
                            st.npass.float() / 3.0,
                            st.dblflag.float(), mine,
                            (st.last >= 0).float()], -1)
        rmap = self._rung_map(seat)
        h = self.body(torch.cat([hv, self._sums(st.bid, seat, rmap),
                                 self._sums(st.dbl, seat, rmap), lastv, scal], -1))
        return h

    def _base_q(self, h):
        """(N, L+2): phase 1's read-out. Under --triple-q this is Q_own."""
        if self.dbl_head is None:
            keys = torch.cat([self.rung.weight, self.pass_key[None],
                              self.dbl_key[None]], 0)
            return self.proj(h) @ keys.T + self.bias
        # --dbl-head: the bids and Pass still read the trunk through the rung
        # embeddings, so the language is untouched; Double reads it through
        # its own two layers, and `detach` stops the gradient there. Double
        # therefore learns FROM the language and can never move it -- which
        # is the exact failure mode of the shared output layer, where a rung
        # is its own input and output weights. Still ONE tensor and ONE
        # argmax over all L+2 calls: this is separate WEIGHTS, not a second
        # decision rule. `dbl_key` and `bias[L+1]` go unused; the head's own
        # output bias is what --dbl-bias sets.
        keys = torch.cat([self.rung.weight, self.pass_key[None]], 0)
        q = self.proj(h) @ keys.T + self.bias[:self.L + 1]
        return torch.cat([q, self.dbl_head(h.detach())], -1)

    def parts(self, hand, st, seat, t, hv=None):
        """(Q_own, Q_denial, Q_double), each (N, L+2). The loss reads this.

        The two competitive heads sit on a DETACHED trunk unless
        --triple-attached is passed, so the only gradient that ever reaches
        the language is phase 1's. That is the same argument as --dbl-head,
        applied to the term that actually destroyed the language in RESULTS
        18 and 19: denial. The interesting failure to watch for is a trunk
        that never learns to represent the opponents well enough for the
        detached heads to fit -- which would be a real answer, not a bug.
        """
        h = self._trunk(hand, st, seat, t, hv)
        base = self._base_q(h)
        if self.den_head is None:
            z = torch.zeros_like(base)
            return base, z, z
        hd = h if self.triple_attached else h.detach()
        return base, self.den_head(hd), self.dblq_head(hd)

    def forward(self, hand, st, seat, t, hv=None):
        """The number every argmax reads: one call, one value.

        Under --triple-q that is the WEIGHTED SUM of the three terms, not a
        gate and not a second decision rule, so no two parts can disagree
        about what to bid.
        """
        if self.den_head is None:
            return self._base_q(self._trunk(hand, st, seat, t, hv))
        own, den, dbl = self.parts(hand, st, seat, t, hv)
        return own + self.w_denial * den + self.w_double * dbl


class HandCache:
    """`hand_enc` for one batch of deals, in all four seats, computed once.

    The hand encoding depends on (deal, seat) and on nothing in the auction,
    but `q_at` used to recompute it for every rollout row on every turn: at
    rollout width that is the same 512 hands encoded a hundred thousand times
    a step. Here the batch's 512 x 4 encodings are computed once against the
    frozen net and gathered by `st.deal`, which is the same arithmetic on the
    same weights -- a gather instead of an MLP.

    `lut` maps a global deal index to its row in the batch; deals outside the
    batch map to row 0 and must never be asked for. Only valid while the net
    it was built from is unchanged, so it belongs to one training step.
    """
    __slots__ = ("enc", "lut")

    def __init__(self, net, hands, deals, n_deals):
        dev = deals.device
        self.lut = torch.zeros(n_deals, dtype=torch.long, device=dev)
        self.lut[deals] = torch.arange(deals.shape[0], device=dev)
        self.enc = [net.hand_enc(hands[s][deals].float()) for s in range(4)]

    def get(self, seat, deal):
        return self.enc[seat][self.lut[deal]]


def q_at(net, hands, st, t, cache=None):
    seat = t % 4
    if cache is not None:
        return net(None, st, seat, t, hv=cache.get(seat, st.deal))
    return net(hands[seat][st.deal].float(), st, seat, t)


def pick_turns(rng, turns_per_step, train_turns, ew_pass=False):
    """Which turns to train this step. Half the slots from each side.

    Even turns are NS, odd turns EW, and turn 0 -- the opening -- is always
    trained. Drawing the remaining slots from one pool of 1..train_turns-1
    handed NS 2.8 trained decisions per step against EW's 2.2, because that
    pool holds six EW turns to five NS ones and turn 0 adds another NS. One
    side getting 29% more gradient is not a language result, and it is the
    same class of bug as exp10four's horizon giving the dealing side an extra
    call. So: k slots per side, the opening counting as one of NS's.

    Keep turns_per_step even. With --ew-pass, EW has no decisions to train.
    """
    k = max(1, turns_per_step // 2)
    ns_pool = [u for u in range(2, train_turns) if u % 2 == 0]
    ew_pool = [u for u in range(1, train_turns) if u % 2 == 1]
    turns = [0] + rng.sample(ns_pool, min(k - 1, len(ns_pool)))
    if not ew_pass:
        turns += rng.sample(ew_pool, min(k, len(ew_pool)))
    return turns


def sign_of(t):
    """Turn the NS score the table gives into the acting side's score."""
    return 1.0 if (t % 4) % 2 == 0 else -1.0


# -------------------------------------------------------------- rollouts

STATE_FIELDS = ("bid", "dbl", "last", "lastseat", "dblflag", "npass", "alive")

# How much a compaction has to shrink the batch before `greedy_finish` pays
# for it: 2 means "only when at least half the rows are dead". Dropping rows
# saves a forward pass each and costs a handful of gathers, so on a device
# where an operator is cheap to dispatch (CPU, CUDA) 1 is fine and measures
# faster; on MPS, where every operator costs a fraction of a millisecond, a
# low threshold spends more on the bookkeeping than the rows are worth.
COMPACT_AT = 2


def _bucket(k):
    """Round a row count up to one of a few dozen repeating sizes.

    Both the row compaction and the legal-call selection below produce a
    batch whose size depends on the data, and a batch size MPS has not seen
    before costs about 20 ms per operator to compile a graph for -- which on
    this machine ate the whole saving and then some. Rounding up to four
    steps per octave (8, 10, 12, 14, 16, 20, ...) keeps the shapes to a set
    that warms up once, at the price of carrying at most 25% dead rows.
    Nothing about the arithmetic depends on it: a padded row is either a
    finished auction, which is a no-op, or a call whose target the loss
    multiplies by zero.
    """
    if k <= 8:
        return 8
    e = k.bit_length() - 3
    return (-(-k >> e)) << e


def _live_first(flag, k):
    """A permutation putting the `k` set rows of `flag` first, both halves in
    their original order -- `argsort(~flag, stable=True)`, but by cumsum, and
    with the input's shape rather than a data-dependent one."""
    n = flag.shape[0]
    ar = torch.arange(n, device=flag.device)
    c = torch.cumsum(flag.long(), 0)
    slot = torch.where(flag, c - 1, k + ar - c)
    return torch.empty(n, dtype=torch.long, device=flag.device).scatter_(0, slot, ar)


def _write_back(dst, src, where, rows=None):
    """dst[where] = src[rows], field by field. `rows=None` means all of src.

    `index_put_` and not `index_copy_`: they compute the same thing, but on
    MPS `index_copy_` into an (n, 35, 4) grid takes six SECONDS at rollout
    width against this one's millisecond. The indices are the rows of a
    boolean mask, so they are unique and the two agree.
    """
    for f in STATE_FIELDS:
        v = getattr(src, f)
        getattr(dst, f).index_put_((where,), v if rows is None else v[rows])


@torch.no_grad()
def greedy_finish(net, hands, st, t0, L, T, doubles, ew_pass=False,
                  cache=None, dbl_ends=False):
    """Play on with the frozen net until everyone has passed, or the cap.

    A dead row is a no-op for the rest of the loop: `apply_call_` gates
    every write on `alive`, so whatever call the argmax hands it, nothing it
    touches can change again. It was still carried through the net on every
    remaining turn, and by turn four of a rollout most rows are dead -- half
    the forward passes in a step were spent on auctions that had ended. So
    the live rows are compacted out and the loop runs on those; the finished
    rows are written back into `st` at the end, which is where the caller
    reads them. Same picks, same arithmetic, fewer rows.

    The compacted batch is padded up to a `_bucket` size and only shrinks
    when it shrinks by `COMPACT_AT`, both of which cost dead rows to save
    bookkeeping; neither can change a number, because a dead row is a no-op.
    """
    idx, cur = None, st
    for u in range(t0, T - 1):
        n_alive = int(cur.alive.sum())
        if n_alive == 0:
            break
        nb = min(_bucket(n_alive), cur.n)
        if nb < cur.n and nb * COMPACT_AT <= cur.n:
            order = _live_first(cur.alive, n_alive)
            keep, drop = order[:nb], order[nb:]
            if idx is not None:
                # The dropped rows are finished, and `st` is a copy that has
                # not seen them since the last compaction.
                _write_back(st, cur, idx[drop], drop)
            idx = keep if idx is None else idx[keep]
            cur = cur[keep]
        seat = u % 4
        m = legal_mask(cur, seat, L, doubles, ew_pass)
        pick = q_at(net, hands, cur, u, cache).masked_fill(~m, NEG_INF).argmax(-1)
        cur = apply_call_(cur, pick, seat, L, dbl_ends)
    if idx is not None:
        _write_back(st, cur, idx)
    return st


@torch.no_grad()
def rollout_value(net, hands, st, t, L, T, score_fn, doubles, chunk=200000,
                  ew_pass=False, cache=None, calls_mask=None, dbl_ends=False,
                  nterm=1):
    """(N, L+2): the true score of every call, the rest played by the frozen
    net. `score_fn(finished, t)` returns it already signed for the side
    acting at turn t -- which side that is, is the only thing the two scoring
    rules disagree about, so the sign lives there rather than here.

    `calls_mask` (N, L+2) rolls out only the calls it marks and leaves the
    rest 0. The trainer's loss multiplies every entry by `legal_mask * w`, so
    an illegal call's target is multiplied by exactly zero and never reaches
    a gradient -- and once the frozen net has bid the ladder up, three
    quarters of the entries are illegal bids below the standing one. Passing
    that same mask in is therefore free, and it is the caller's job to make
    sure it really is the mask the target gets multiplied by.
    """
    A, dev = L + 2, st.last.device
    if calls_mask is None:
        big = st.repeat(A)
        calls = torch.arange(A, device=dev).repeat(st.n)
        sel = None
    else:
        flat = calls_mask.reshape(-1)
        k = int(flat.sum())
        if k == 0:
            z = torch.zeros(st.n, A, nterm, device=dev)
            return z if nterm > 1 else z[..., 0]
        # Padding up to a bucket rolls out a few calls the loss will throw
        # away, which is cheaper than a new batch shape.
        sel = _live_first(flat, k)[:min(_bucket(k), flat.shape[0])]
        big = st[torch.div(sel, A, rounding_mode="floor")]
        calls = sel % A
    kind = call_kind(big, calls, L, dbl_ends)
    big = apply_call_(big, calls, t % 4, L, dbl_ends)
    out = torch.zeros(st.n * A, nterm, device=dev)
    val = out if sel is None else torch.empty(big.n, nterm, device=dev)
    for i in range(0, big.n, chunk):
        sl = slice(i, min(i + chunk, big.n))
        v = score_fn(greedy_finish(net, hands, big[sl], t + 1, L, T,
                                   doubles, ew_pass, cache, dbl_ends),
                     t, kind[sl])
        val[sl] = v if v.dim() == 2 else v[:, None]
    if sel is not None:
        out[sel] = val
    out = out.view(st.n, A, nterm)
    return out if nterm > 1 else out[..., 0]


@torch.no_grad()
def sampled_states(net, hands, deal, t, L, S, eps, doubles, window=0,
                   ew_pass=False, cache=None, dbl_ends=False):
    """Practice positions for turn t.

    The actor's own earlier calls are drawn uniformly over its legal
    options -- drop your own reach, so its present taste cannot decide how
    much it learns below the calls it avoids. Every other seat is sampled
    from the frozen net's floored greedy policy and the row carries the
    probability mass that sample stands for. A call that would have ENDED
    the auction before turn t is struck out of that mass, so a state that
    cannot be reached carries weight 0 instead of a wrong target.
    """
    dev = deal.device
    d = deal.repeat_interleave(S, 0)
    st = St.empty(d, L)
    w = torch.ones(d.shape[0], device=dev)
    for u in range(t):
        seat = u % 4
        ok = (legal_mask(st, seat, L, doubles, ew_pass)
              & ~ends_mask(st, L, dbl_ends))
        okf = ok.float()
        # The window applies to the RANDOM draws only. A seat's greedy call is
        # never blocked, so the policy can always bid what it actually wants.
        okw = (ok & window_mask(st, L, window)).float()
        okw = torch.where(okw.sum(-1, keepdim=True) > 0, okw, okf)
        if seat == t % 4:
            call = torch.multinomial(okw + 1e-12, 1).squeeze(1)
        else:
            m = legal_mask(st, seat, L, doubles, ew_pass)
            q = q_at(net, hands, st, u, cache).masked_fill(~m, NEG_INF)
            one = F.one_hot(q.argmax(-1), L + 2).float()
            f = (m & window_mask(st, L, window)).float()
            f = torch.where(f.sum(-1, keepdim=True) > 0, f, m.float())
            f = f / f.sum(-1, keepdim=True).clamp(min=1)
            p = ((1 - eps) * one + eps * f) * okf
            w = w * p.sum(-1)
            p = torch.where(p.sum(-1, keepdim=True) > 0, p, okf)
            call = torch.multinomial(p + 1e-12, 1).squeeze(1)
        st = apply_call_(st, call, seat, L, dbl_ends)
    return st, w


# ------------------------------------------------------------ evaluation

def dbl_down_mask(st, tricks_all, sdd, ssi, lvl, L, seat_map=None):
    """(N,) doubled, (N,) went_down: bools over a batch of FINISHED auctions.

    `doubled` is which rows have a contract that was in fact doubled;
    `went_down` is which rows' declarer took fewer tricks than the contract
    needed, regardless of doubling -- `dbl_rate` in `evaluate` reads the
    second restricted to the first. Same declarer/strain/tricks lookup as
    `ns_score` and `own_score`; `lvl` is `scoring4.contract_levels` as a
    tensor, indexed by rung.
    """
    has = st.last >= 0
    doubled = st.dblflag.bool() & has
    c = st.last.clamp(min=0)
    decl = declarer_of(st, ssi, L)
    dd_seat = decl if seat_map is None else seat_map[decl]
    tricks = tricks_all[st.deal, dd_seat, sdd[c]].long()
    went_down = tricks < (lvl[c] + 6)
    return doubled, went_down


def dbl_rate_from_masks(doubled, went_down):
    """(scalar) of `doubled` rows, the share where `went_down` is true --
    the doubling side won points on it. 0 when nothing was doubled, not
    NaN, so a run with no doubles yet still logs a plain number.
    """
    denom = doubled.float().sum().clamp(min=1e-6)
    return round(float((went_down & doubled).float().sum() / denom), 4)


@torch.no_grad()
def play(net, hands, deal, L, T, doubles, keep=16, ew_pass=False, cache=None,
         dbl_ends=False):
    st = St.empty(deal, L)
    picks = torch.full((deal.shape[0], keep), -1, dtype=torch.long,
                       device=deal.device)
    ncalls = torch.zeros(deal.shape[0], dtype=torch.long, device=deal.device)
    live = st.alive
    for t in range(T):
        if not bool(st.alive.any()):
            live = st.alive
            break
        seat = t % 4
        if t == T - 1:
            live = st.alive
            pick = torch.full_like(st.last, L)
        else:
            m = legal_mask(st, seat, L, doubles, ew_pass)
            q = q_at(net, hands, st, t, cache)
            pick = q.masked_fill(~m, NEG_INF).argmax(-1)
        if t < keep:
            picks[:, t] = torch.where(st.alive, pick, torch.full_like(pick, -1))
        ncalls = ncalls + st.alive.long()
        st = apply_call_(st, pick, seat, L, dbl_ends)
    return st, picks, ncalls, live


@torch.no_grad()
def evaluate(net, hands, tricks_all, tbl4, sdd, ssi, lvl, idx, L, T, doubles,
             par, chunk=20000, ew_pass=False, own_par=None, winner=None,
             dbl_ends=False, pass_cost=0.0):
    """Play the held-out deals, then play them again with East dealing.

    EW's score is minus NS's, so it is not printed. What is printed is the
    gap to par and which side opened it: `nsloss` is the points NS gave up
    against par, `ewloss` the points EW gave up. `ns_rot` is only a sanity
    check that one net really does sit in both chairs -- it should be `ns`
    with the sign flipped.

    `own_par` (the two sides' cooperative pars on `idx`) switches on the
    non-competing readout: both sides scored on their own highest bid, out
    of the SAME unrotated auctions, so the two numbers are comparable.

    `winner` is (loser_mode, loser_scale, the two pars over ALL deals) and
    adds what --winner-only actually pays each side, off those same
    auctions. `pass_cost` > 0 adds what --pass-cost pays each side, off the
    same auctions again. The own-bid numbers stay in both cases: they are
    the language yardstick and the only ones comparable across the stages.
    """
    dev = idx.device

    def one(rot, own=False):
        hh = [hands[(j + rot) % 4] for j in range(4)]
        sm = torch.tensor([(j + rot) % 4 for j in range(4)], device=dev)
        acc = {k: [] for k in ("sc", "dbl", "nsd", "con", "nc", "by3", "op",
                               "own0", "own1", "nb0", "nb1", "down",
                               "win0", "win1", "pc0", "pc1")}
        for i in range(0, len(idx), chunk):
            s = idx[i:i + chunk]
            st, picks, nc, live = play(net, hh, s, L, T, doubles,
                                       ew_pass=ew_pass, dbl_ends=dbl_ends)
            raw = ns_score(st, tricks_all, tbl4, sdd, ssi, L, sm)
            acc["sc"].append(raw if rot % 2 == 0 else -raw)
            acc["dbl"].append(st.dblflag.float() * (st.last >= 0).float())
            acc["nsd"].append(((declarer_of(st, ssi, L) % 2 == 0)
                               & (st.last >= 0)).float())
            acc["con"].append(st.last)
            acc["nc"].append(nc)
            acc["by3"].append(~live)
            acc["op"].append(picks[:, 0])
            acc["down"].append(dbl_down_mask(st, tricks_all, sdd, ssi, lvl,
                                             L, sm)[1].float())
            for side in (0, 1) if own else ():
                acc[f"own{side}"].append(
                    own_score(st, side, tricks_all, tbl4, sdd, ssi, L, sm))
                acc[f"nb{side}"].append(st.bid[:, :, side].sum(1)
                                        + st.bid[:, :, side + 2].sum(1))
                if winner is not None:
                    mode, scale, wpar = winner
                    acc[f"win{side}"].append(
                        winner_score(st, side, tricks_all, tbl4, sdd, ssi, L,
                                     mode, scale, wpar[0], wpar[1], sm))
                if pass_cost > 0:
                    acc[f"pc{side}"].append(
                        pass_cost_score(st, side, tricks_all, tbl4, sdd, ssi,
                                        L, pass_cost, sm))
        return {k: torch.cat(v) for k, v in acc.items() if v}

    a = one(0, own_par is not None)
    b = one(1)
    sc, con = a["sc"], a["con"]
    has = con >= 0
    lv = lvl[con.clamp(min=0)]
    gap = par - sc
    op = a["op"].cpu().numpy()
    own = {}
    if own_par is not None:
        for side, name in ((0, "ns"), (1, "ew")):
            own[f"{name}_own"] = round(float(a[f"own{side}"].mean()), 2)
            own[f"{name}_bids"] = round(float(a[f"nb{side}"].mean()), 2)
            own[f"{name}_nobid"] = round(float((a[f"nb{side}"] == 0)
                                               .float().mean()), 4)
            own[f"{name}_par_own"] = round(float(own_par[side].mean()), 2)
            if winner is not None:
                own[f"{name}_win"] = round(float(a[f"win{side}"].mean()), 2)
            if pass_cost > 0:
                own[f"{name}_pc"] = round(float(a[f"pc{side}"].mean()), 2)
        if winner is not None:
            own["loser_mode"] = winner[0]
        if pass_cost > 0:
            own["pass_cost"] = pass_cost
    dbl = {}
    if doubles:
        # Of the contracts that ended up doubled, the share that went DOWN
        # -- the doubling side won points on it. A doubling policy that is
        # just reflexive (measured: 97-100% of everything) makes this look
        # like the base down-rate; a correct one makes it much higher.
        dbl["dbl_rate"] = dbl_rate_from_masks(a["dbl"] > 0.5, a["down"] > 0.5)
    return dict(
        **own,
        **dbl,
        ns=round(float(sc.mean()), 2),
        ns_rot=round(float(b["sc"].mean()), 2),
        par=round(float(par.mean()), 2),
        gap=round(float(gap.mean()), 2),
        absgap=round(float(gap.abs().mean()), 2),
        nsloss=round(float(gap.clamp(min=0).mean()), 2),
        ewloss=round(float((-gap).clamp(min=0).mean()), 2),
        passout=round(float((~has).float().mean()), 4),
        end3=round(float(a["by3"].float().mean()), 4),
        calls=round(float(a["nc"].float().mean()), 2),
        maxcalls=int(a["nc"].max()),
        doubled=round(float(a["dbl"].mean()), 4),
        ns_decl=round(float(a["nsd"].mean()), 4),
        lvl12=round(float(((lv <= 2) & has).float().mean()), 4),
        lvl34=round(float(((lv >= 3) & (lv <= 4) & has).float().mean()), 4),
        lvl5=round(float(((lv == 5) & has).float().mean()), 4),
        slam=round(float((lv >= 6).float().mean()), 4),
        bits=round(_bits(np.bincount(op[op >= 0], minlength=L + 2)
                         .astype(float)), 3),
    )


def log_line(s):
    line = (f"s:{s['step']:<6} ns:{s['ns']:+7.1f} par:{s['par']:6.1f} "
            f"gap:{s['gap']:+7.1f} |gap|:{s['absgap']:6.1f} "
            f"nsloss:{s['nsloss']:6.1f} "
            f"ewloss:{s['ewloss']:6.1f} | po:{s['passout']:5.1%} "
            f"x:{s['doubled']:5.1%} nsd:{s['ns_decl']:5.1%} "
            f"len:{s['calls']:4.1f}/{s['maxcalls']:<2d} "
            f"end3:{s['end3']:5.1%} bits:{s['bits']:.2f} "
            f"| rot:{s['ns_rot']:+6.1f} mse:{s['mse']:.3f} {s['secs']:.0f}s")
    if "dbl_rate" in s:
        line += f"  dbldown:{s['dbl_rate']:5.1%}"
    if "ns_own" in s:
        line += (f"\n{'':>8}own  ns:{s['ns_own']:+7.1f}/par {s['ns_par_own']:5.0f}"
                 f"  ew:{s['ew_own']:+7.1f}/par {s['ew_par_own']:5.0f}"
                 f"  bids ns:{s['ns_bids']:.2f} ew:{s['ew_bids']:.2f}"
                 f"  nobid ns:{s['ns_nobid']:5.1%} ew:{s['ew_nobid']:5.1%}")
    if "ns_pc" in s:
        line += (f"\n{'':>8}pass ns:{s['ns_pc']:+7.1f} ew:{s['ew_pc']:+7.1f}"
                 f"  lambda:{s['pass_cost']:g}")
    if "ns_win" in s:
        line += (f"\n{'':>8}win  ns:{s['ns_win']:+7.1f} ew:{s['ew_win']:+7.1f}"
                 f"  loser:{s['loser_mode']}")
    return line


def lvl_line(s):
    return (f"{'':>8}levels  1-2:{s['lvl12']:5.1%}  3-4:{s['lvl34']:5.1%}  "
            f"5:{s['lvl5']:5.1%}  slam:{s['slam']:5.1%}")


# ---------------------------------------------------------------- driver

# The heads that may legitimately start fresh on a warm start: each one is
# an ADDITION to the trunk, so a checkpoint without it still carries every
# shared tensor. Going the other way -- a checkpoint that HAS one loading
# into a net that does not -- is refused, because it would silently discard
# something trained.
FRESH_HEADS = ("dbl_head.", "den_head.", "dblq_head.")


def load_init(net, path):
    """Warm start: the net weights out of a checkpoint this script saved.

    The competitive stage starts from the non-competing one, so only the
    weights carry over -- the optimizer state belongs to the old objective
    and would drag it along. weights_only is off because our own checkpoint
    also carries the config and the history. Returns the checkpoint's saved
    config (or None if it has none), so the caller can check what game it
    was trained under.

    --dbl-head is the one shape change this file has, and it goes exactly
    one way. A checkpoint WITHOUT the head loads into a net WITH one: every
    shared tensor is copied and the head alone starts fresh, which is the
    normal warm start for the new arm, and it says so. The reverse is
    refused: dropping a trained head would silently hand the Double column
    back to `dbl_key`, which in such a checkpoint is whatever the random
    init left there.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["net"]
    here = net.state_dict()
    have = {p: any(k.startswith(p) for k in here) for p in FRESH_HEADS}
    saved = {p: any(k.startswith(p) for k in sd) for p in FRESH_HEADS}
    if saved["dbl_head."] and not have["dbl_head."]:
        raise SystemExit(
            f"--init {path} was trained with --dbl-head (its Double column "
            "comes from dbl_head.*), but this run has no --dbl-head. "
            "Loading it would drop the trained head and read Double off "
            "`dbl_key`, which never trained in that checkpoint. Add "
            "--dbl-head, or init from a checkpoint without one.")
    for pre in ("den_head.", "dblq_head."):
        if saved[pre] and not have[pre]:
            raise SystemExit(
                f"--init {path} was trained with --triple-q (it carries "
                f"{pre}*), but this run has none. Loading it would throw the "
                "trained denial and double terms away and keep only Q_own. "
                "Add --triple-q, or init from a checkpoint without it.")
    fresh = [p for p in FRESH_HEADS if have[p] and not saved[p]]
    if not fresh:
        net.load_state_dict(sd, strict=True)
        return ck.get("config")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    stray = [k for k in missing
             if not any(k.startswith(p) for p in fresh)]
    if stray or unexpected:
        raise SystemExit(f"--init {path} does not fit this net: "
                         f"missing {stray}, unexpected {list(unexpected)}")
    print(f"warm start: {path} has no {', '.join(f.rstrip('.') for f in fresh)}"
          f". Loaded every shared tensor; those {len(missing)} tensors are "
          "FRESH and untrained. A head that starts fresh has no opinion worth "
          "anything until it has trained, which is what --dbl-bias exists to "
          "say about Double.")
    return ck.get("config")


def apply_dbl_bias(net, L, value):
    """Set the Double column's constant to `value`, in place.

    That is `bias[L+1]` in the shared output head, and the output bias of
    the separate head under --dbl-head -- the same knob either way, since
    both are the one number added to the Double logit and nothing else.
    Nothing else in `net` is touched. Returns the value it replaced, so the
    caller can log what changed.
    """
    with torch.no_grad():
        if net.dbl_head is None:
            old = float(net.bias[L + 1])
            net.bias[L + 1] = value
        else:
            b = net.dbl_head[-1].bias
            old = float(b[0])
            b[0] = value
    return old


def dbl_bias_name(net, L):
    return "dbl_head output bias" if net.dbl_head is not None else f"bias[{L + 1}]"


def warn_stale_double(init_cfg, doubles):
    """Warn when --init loads a checkpoint whose Double head never trained.

    --own-bid forces doubles off and --no-double turns them off directly;
    either way Double never appears in a legal mask for that whole run, so
    its output column (bias[L+1], dbl_key) gets no gradient and arrives in
    the checkpoint at its random init, unrelated to the rest of the head.
    Loading that into a run with doubles ON is exactly the setup that has
    been measured collapsing to doubling 97-100% of contracts (WBridge5:
    9.5%). This only warns -- it does not change behaviour; --dbl-bias does.
    """
    if init_cfg is None or not doubles:
        return
    stale = init_cfg.get("no_double") or init_cfg.get("own_bid")
    if not stale:
        return
    print("WARNING: --init checkpoint was trained with doubles OFF "
          f"(no_double={init_cfg.get('no_double')} "
          f"own_bid={init_cfg.get('own_bid')}), but this run has doubles "
          "ON. The Double head (bias[L+1], dbl_key) never trained in that "
          "checkpoint and is unrelated to the rest of the output head -- "
          "runs started this way have been measured doubling 97-100% of "
          "contracts (WBridge5 doubles 9.5%). Consider --dbl-bias to start "
          "sceptical about Double instead of enthusiastic.")


def run(cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = cfg.device
    contracts = [c for c in FULL_CONTRACTS if c[1] is not None]
    names = [c[0] for c in contracts]
    L, T = len(contracts), cfg.max_turns
    doubles = not cfg.no_double and not cfg.own_bid

    hands, tricks_all = load_deals4(cfg.deals, dev, cfg.limit)
    n = hands[0].shape[0]
    tbl4 = torch.tensor(build_score_table4(contracts, cfg.vulnerable), device=dev)
    sdd = torch.tensor(contract_strain_dd(contracts), device=dev)
    ssi = torch.tensor(contract_strains(contracts), device=dev)
    lvl = torch.tensor(contract_levels(contracts), device=dev)

    perm = np.random.permutation(n)
    n_test = min(cfg.n_test, n // 5)
    test = torch.tensor(perm[:n_test], device=dev)
    train = torch.tensor(perm[n_test:], device=dev)
    # --own-bid and --winner-only both take the side's OWN cooperative par
    # as the baseline, so the minimax par is only the eval's yardstick there
    # and does not need computing over every deal.
    # --route-blame grades most rows on the acting side's own contract,
    # so its natural centring is own par too. It must be ONE baseline for
    # every call at a state: the call kind varies across the row, so a
    # per-kind baseline would reorder calls instead of just recentring.
    own_base = (cfg.own_bid or cfg.winner_only or cfg.route_blame
                or cfg.triple_q)
    par_deal = (par_all(tricks_all, tbl4, sdd)
                if cfg.par_baseline and not own_base else None)
    own_par = ([coop_par_side(tricks_all, tbl4, sdd, s) for s in (0, 1)]
               if own_base else None)
    par = (par_deal[test] if par_deal is not None
           else par_table(tricks_all[test], tbl4, sdd))
    own_par_test = [p[test] for p in own_par] if own_par is not None else None

    net = SeatNet(L, cfg.hidden, cfg.d_hand, cfg.d_rung, cfg.layers,
                  dbl_head=cfg.dbl_head, triple=cfg.triple_q,
                  triple_attached=cfg.triple_attached,
                  w_denial=cfg.w_denial, w_double=cfg.w_double).to(dev)
    if cfg.init:
        init_cfg = load_init(net, cfg.init)
        warn_stale_double(init_cfg, doubles)
        if cfg.dbl_bias is not None:
            old = apply_dbl_bias(net, L, cfg.dbl_bias)
            print(f"--dbl-bias: {dbl_bias_name(net, L)} (Double) was "
                  f"{old:.4f}, now {cfg.dbl_bias:.4f}")
    frozen = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    nparam = sum(p.numel() for p in net.parameters())

    print(f"ONE NET, FOUR SEATS   doubles={doubles}"
          f"{'(ending)' if cfg.dbl_ends else ''}  cap={T} turns  "
          f"seed={cfg.seed}  dev={dev}")
    print(f"deals={n} train={len(train)} test={n_test} rungs={L} "
          f"calls={L + 2 if doubles else L + 1}")
    print(f"one net, {nparam/1e6:.2f}M parameters, in all four seats; "
          f"frozen rollout copy refreshed every {cfg.refresh} steps")
    print(f"{cfg.steps} steps, batch {cfg.batch}, {cfg.states} states/deal, "
          f"{cfg.turns_per_step} of turns 0..{cfg.train_turns - 1} per step, "
          f"balanced NS/EW")
    print(f"random practice calls windowed to {cfg.window or 'all'} rungs "
          f"above the standing bid")
    if cfg.init:
        print(f"warm start: net weights loaded from {cfg.init}")
    if cfg.dbl_ends:
        print("DOUBLE ENDS THE AUCTION: a Double is the last call. The "
              "contract is the standing rung, doubled, scored exactly as "
              "always. THIS IS NOT BRIDGE -- running from a double is legal "
              "in the real game -- so it is a training wheel, and any result "
              "under it has to be re-checked without it.")
    if cfg.dbl_head:
        print("DOUBLE HEAD: Double's logit comes from its own two layers on "
              "the trunk, with the gradient stopped there, so it learns from "
              "the language and cannot move it. One argmax, as before.")
    if cfg.huber > 0:
        print(f"loss: Huber, delta={cfg.huber:g} (target units, i.e. "
              f"{cfg.huber * 100:.0f} points) instead of squared error")
    if cfg.clip_target > 0:
        print(f"target clipped to +-{cfg.clip_target:g} points after the "
              "baseline")
    if cfg.ew_pass:
        print("GATE: East and West may only Pass. Target is exp9's 110.")
    if cfg.winner_only:
        print("WINNER-ONLY: only the side that won the auction is paid, on "
              "its real contract, doubled if it was doubled.")
        print(f"loser side: {cfg.loser_mode} x {cfg.loser_scale:g}")
    if cfg.own_bid:
        print("NON-COMPETING: each side scored on its own highest bid. "
              "Targets: WBridge5 97.8, two-player 111.3")
        print("doubles OFF: a side cannot double itself and the opponents' "
              "double is not read")
    if cfg.triple_q:
        print(f"TRIPLE Q (ticket 008): Q = Q_own + {cfg.w_denial:g} x Q_denial"
              f" + {cfg.w_double:g} x Q_double, ONE argmax over the sum. The "
              "three targets are read off the same rollout and sum EXACTLY to "
              "the real zero-sum score: own contract, what their contract "
              "does to us, what the double was worth.")
        print("  the denial and double heads sit on a "
              + ("LIVE" if cfg.triple_attached else "DETACHED")
              + " trunk, so the language "
              + ("can" if cfg.triple_attached else "cannot")
              + " move with them.")
    if cfg.inner_steps > 1:
        print(f"INNER STEPS {cfg.inner_steps}: each batch of rollouts is "
              f"regressed {cfg.inner_steps} times before the next one is "
              f"drawn, so the net takes {cfg.inner_steps} x {cfg.steps} "
              "gradient steps for the rollout cost of "
              f"{cfg.steps}. The rollout is ~70% of a step, so this is "
              "close to free.")
    if cfg.route_blame:
        print(f"ROUTE BLAME scale={cfg.route_scale:g}: a bid, and a pass that "
              "leaves the auction alive, are graded on the acting side's OWN "
              "contract" + ("" if cfg.route_plain_own else
                            " (doubled if the board really doubled us)") +
              ". The pass that ENDS the auction, and a Double, are graded on "
              "the real zero-sum score. The call that chose the contract "
              "pays for it; the calls that only talked do not.")
    if cfg.pass_baseline:
        print("PASS BASELINE: every target has the value of Pass AT THE SAME "
              "STATE subtracted, so the net fits what one call ADDED rather "
              "than what the board was worth. A per-state constant, so no "
              "argmax can move.")
    if cfg.pass_cost > 0:
        print(f"PASS COST lambda={cfg.pass_cost:g}: a side that never bid is "
              "charged that times the opponents' ACTUAL final score (their "
              "sign, doubled if it was doubled); a side that bid is scored on "
              "its own highest bid exactly as always. Silence has a "
              "deal-dependent price and a sacrifice pays for itself.")
    if own_base:
        print(f"own par: ns {own_par_test[0].mean():.1f}  "
              f"ew {own_par_test[1].mean():.1f}")
    print(f"double-dummy par (both sides bid, doubles priced) = "
          f"{par.mean():.1f} +- {par.std():.0f}")
    base = 'own par' if own_base else 'par'
    print(f"target: {f'score - {base} (baseline on)' if cfg.par_baseline else 'raw score'}\n")

    # Every rule takes the call kind now; only --route-blame reads it.
    if cfg.triple_q:
        score_fn = lambda s, t, k: triple_score(s, (t % 4) % 2, tricks_all,
                                                tbl4, sdd, ssi, L)
    elif cfg.route_blame:
        score_fn = lambda s, t, k: route_blame_score(
            s, (t % 4) % 2, tricks_all, tbl4, sdd, ssi, L, k,
            cfg.route_scale, not cfg.route_plain_own)
    elif cfg.own_bid and cfg.pass_cost > 0:
        score_fn = lambda s, t, k: pass_cost_score(s, (t % 4) % 2, tricks_all,
                                                   tbl4, sdd, ssi, L,
                                                   cfg.pass_cost)
    elif cfg.own_bid:
        score_fn = lambda s, t, k: own_score(s, (t % 4) % 2, tricks_all, tbl4,
                                             sdd, ssi, L)
    elif cfg.winner_only:
        score_fn = lambda s, t, k: winner_score(s, (t % 4) % 2, tricks_all,
                                                tbl4, sdd, ssi, L,
                                                cfg.loser_mode,
                                                cfg.loser_scale, own_par[0],
                                                own_par[1])
    else:
        score_fn = lambda s, t, k: sign_of(t) * ns_score(s, tricks_all, tbl4,
                                                         sdd, ssi, L)
    hist_log, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        if (step - 1) % cfg.refresh == 0:
            frozen.load_state_dict(net.state_dict())
        b = train[torch.randint(len(train), (cfg.batch,), device=dev)]
        # One net, so it does not need every turn every step: turn 0 always,
        # and a fresh sample of the rest. Cost is linear in turns_per_step
        # rather than quadratic in the cap.
        #
        turns = pick_turns(rng, cfg.turns_per_step, cfg.train_turns,
                           cfg.ew_pass)
        batch_items = []
        # The frozen net picks every call in every rollout and the batch is
        # 512 deals, so its hand encodings are the same few thousand numbers
        # all step. Compute them once.
        cache = HandCache(frozen, hands, b, n)
        for t in sorted(turns, reverse=True):
            if t == 0:
                st, w = St.empty(b, L), torch.ones(cfg.batch, device=dev)
            else:
                st, w = sampled_states(frozen, hands, b, t, L, cfg.states,
                                       cfg.explore, doubles, cfg.window,
                                       cfg.ew_pass, cache, cfg.dbl_ends)
            mask = legal_mask(st, t % 4, L, doubles,
                              cfg.ew_pass).float() * w[:, None]
            with torch.no_grad():
                # Only the calls the loss actually weighs are rolled out: an
                # illegal call, or a state the sampler gave weight 0, is
                # multiplied by exactly zero below whatever its target says.
                tgt = rollout_value(frozen, hands, st, t, L, T, score_fn,
                                    doubles, cfg.roll_chunk, cfg.ew_pass,
                                    cache, mask != 0, cfg.dbl_ends,
                                    3 if cfg.triple_q else 1)
                if cfg.pass_baseline:
                    # Ticket 008. Pass is legal at every state, so column L
                    # was rolled out anyway and this is free. A constant of
                    # the STATE, so like --par-baseline it cannot reorder two
                    # calls; it only changes what the regression must fit.
                    tgt = tgt - tgt[:, L:L + 1].clone()
                if cfg.par_baseline:
                    # A constant of the deal and of the acting side, so it
                    # cannot reorder two calls at a state. It only takes the
                    # deal's luck out of the target, which is where this
                    # trainer's variance lives. True under --pass-cost too:
                    # the side's cooperative par is still one number per deal
                    # per side, so subtracting it cannot change any argmax,
                    # whichever branch of the reward the row landed in.
                    b0 = (own_par[(t % 4) % 2][st.deal] if own_base
                          else sign_of(t) * par_deal[st.deal])
                    if cfg.triple_q:
                        # Only the own-contract column lives on the own-par
                        # scale. Denial and double are already DIFFERENCES of
                        # two scores, so they are centred at zero already and
                        # subtracting par again would just bias them.
                        tgt = torch.cat([tgt[:, :, :1] - b0[:, None, None],
                                         tgt[:, :, 1:]], -1)
                    else:
                        tgt = tgt - b0[:, None]
                tgt = clip_target(tgt, cfg.clip_target) / 100.0
            # The rollout above is ~70% of the step and the target is now
            # fixed. Hold on to it so --inner-steps can be regressed on it
            # more than once; the forward below is the cheap part.
            batch_items.append((st, t, tgt, mask))

        # Same quantity, same legal mask, same reach weighting, same
        # normalisation. --huber changes only the per-entry penalty and
        # --clip-target only what is regressed.
        for _ in range(cfg.inner_steps):
            total = 0.0
            for st, t, tgt, mask in batch_items:
                if cfg.triple_q:
                    # Each head regresses its OWN unweighted target, so
                    # annealing w_denial or w_double changes what the policy
                    # plays without disturbing what any head predicts.
                    parts = net.parts(hands[t % 4][st.deal].float(), st,
                                      t % 4, t)
                    err = sum(q_err(parts[i] - tgt[:, :, i], cfg.huber)
                              for i in range(3))
                else:
                    err = q_err(q_at(net, hands, st, t) - tgt, cfg.huber)
                total = total + (err * mask).sum() / mask.sum().clamp(min=1e-6)
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        if step % 200 == 0:
            free_cache(dev)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            free_cache(dev)
            net.eval()
            s = evaluate(net, hands, tricks_all, tbl4, sdd, ssi, lvl, test,
                         L, T, doubles, par, ew_pass=cfg.ew_pass,
                         own_par=own_par_test,
                         winner=((cfg.loser_mode, cfg.loser_scale, own_par)
                                 if cfg.winner_only else None),
                         dbl_ends=cfg.dbl_ends, pass_cost=cfg.pass_cost)
            net.train()
            s.update(step=step, secs=round(time.time() - t0, 1),
                     mse=round(float(total.item()), 4))
            hist_log.append(s)
            print(log_line(s), flush=True)
            print(lvl_line(s), flush=True)
            if cfg.save:
                os.makedirs(os.path.dirname(cfg.save) or ".", exist_ok=True)
                torch.save({"step": step, "hist": hist_log, "config": vars(cfg),
                            "names": names, "net": net.state_dict()}, cfg.save)
            if cfg.out:
                os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
                json.dump({"config": vars(cfg), "history": hist_log},
                          open(cfg.out, "w"), indent=2)
            free_cache(dev)
    return hist_log[-1] if hist_log else {}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", default="data_emergent/pgx1M.npz")
    p.add_argument("--max-turns", type=int, default=40, dest="max_turns",
                   help="SAFETY CAP, not a rule. The auction ends on three "
                        "passes; bids only ascend so it always ends. Almost "
                        "nothing reaches this")
    p.add_argument("--train-turns", type=int, default=12, dest="train_turns",
                   help="turn indices the trainer draws practice states from")
    p.add_argument("--turns-per-step", type=int, default=6,
                   dest="turns_per_step",
                   help="how many of those to train each step. Half from each "
                        "side, turn 0 always, so keep it EVEN or NS gets more "
                        "gradient than EW")
    p.add_argument("--window", type=int, default=6,
                   help="a RANDOM practice call is drawn from Pass, Double and "
                        "the next this-many rungs above the standing bid, not "
                        "flat over all 35. A greedy call is never windowed. "
                        "0 turns it off, which is the old behaviour")
    p.add_argument("--ew-pass", action="store_true", dest="ew_pass",
                   help="GATE, not a curriculum: East and West may only Pass, "
                        "so NS bids alone. This is exp9's game inside the "
                        "four-seat code, and it must land near exp9's 110 "
                        "(--open-pass, 35 rungs). If it does not, the "
                        "four-seat plumbing is wrong and every competitive "
                        "number after it is worthless")
    p.add_argument("--own-bid", action="store_true", dest="own_bid",
                   help="NON-COMPETING pretraining: each side is scored on "
                        "its own highest bid, as if it declared it, whoever "
                        "actually won the auction. Outbidding the opponents "
                        "wins nothing; what interference costs is the room it "
                        "takes off the ladder. Turns doubles off by itself. "
                        "Targets: WBridge5 97.8, two-player 111.3")
    p.add_argument("--pass-cost", type=float, default=0.0, dest="pass_cost",
                   help="REWARD, and only with --own-bid: a side that never "
                        "bid is charged this lambda times the opponents' "
                        "ACTUAL final score (doubled if it was doubled), "
                        "instead of the 0 --own-bid gives it. A side that "
                        "bid is scored on its own highest bid exactly as "
                        "before, so the attributable branch phase 1 is built "
                        "on is untouched and the only thing that changes is "
                        "what silence costs. Never par: par is privileged "
                        "information about the deal. This buys the sacrifice "
                        "incentive for free -- -100 in your own contract "
                        "beats letting them have +420 -- while keeping the "
                        "deliberate fiction that being outbid still pays you "
                        "for your own bid, which is why phase 1 is stable. "
                        "0 is off and bit-identical to plain --own-bid; 1 is "
                        "the full price; anneal it down later")
    p.add_argument("--winner-only", action="store_true", dest="winner_only",
                   help="only the side that WON the auction is paid, on its "
                        "real contract, doubled if it was doubled. One "
                        "contract survives an auction, so the losing side is "
                        "not paid for a bid nobody played -- what it gets "
                        "instead is --loser-mode. Doubles stay ON unless "
                        "--no-double. Not compatible with --own-bid")
    p.add_argument("--loser-mode", default="zero", dest="loser_mode",
                   choices=list(LOSER_MODES),
                   help="what the side that lost the auction is paid under "
                        "--winner-only: zero (nothing, so Pass costs what it "
                        "did under --own-bid), dbl (undoubled minus doubled, "
                        "which is what its Double was worth and nothing "
                        "else), potential (the winner's own cooperative par "
                        "minus what it made), potential-clip (that, floored "
                        "at 0), zerosum (minus the winner's score, i.e. the "
                        "real game -- a check, not a curriculum)")
    p.add_argument("--loser-scale", type=float, default=1.0, dest="loser_scale",
                   help="multiplies the --loser-mode term only")
    p.add_argument("--init", default="",
                   help="warm start the net from a checkpoint this script "
                        "saved (--save). The optimizer starts fresh")
    p.add_argument("--dbl-bias", type=float, default=None, dest="dbl_bias",
                   help="after --init, set bias[L+1] (the Double column) to "
                        "this value. An own-bid checkpoint's Double head "
                        "never trained and arrives at its random init; "
                        "this lets a warm-started doubled run start "
                        "sceptical about Double instead of enthusiastic")
    p.add_argument("--huber", type=float, default=0.0,
                   help="OPTIMISER, not reward: pay for an error with Huber "
                        "instead of squared error, elbow at this delta. 0 is "
                        "off. Units are the loss's, i.e. POINTS/100, because "
                        "the target is divided by 100 -- so 3 is a 300-point "
                        "elbow. Why: with Double legal the level-0 screen "
                        "measured a per-state target spread of 1345 points "
                        "and an sd of 465 against own-bid's 620 and 236, and "
                        "a doubled 13-down contract is -3500 where undoubled "
                        "it is -350. Squared error chases those; Huber does "
                        "not. Start at 3 (an elbow just above own-bid's sd, "
                        "well inside the doubled one), and note that a very "
                        "large delta is exactly the old loss")
    p.add_argument("--clip-target", type=float, default=0.0, dest="clip_target",
                   help="OPTIMISER, not reward: clamp the baselined target to "
                        "+-this many POINTS before it enters the loss. 0 is "
                        "off. Bounds what is regressed where --huber bounds "
                        "what an error costs; the two are independent and "
                        "combine. 1000 is legible next to the 1345-point "
                        "spread the screen measured with Double legal")
    p.add_argument("--dbl-head", action="store_true", dest="dbl_head",
                   help="ticket 006: give Double its OWN parameters. Today "
                        "call k<L is a rung whose input and output weights "
                        "are one set of numbers, so a saturating Double "
                        "logit drags the language with it. With this, bids "
                        "and Pass read the trunk through the rung embeddings "
                        "as now, Double reads it through a small separate "
                        "head, and the gradient is STOPPED where that head "
                        "meets the trunk -- Double learns from the language "
                        "and can never change it. Still one argmax over all "
                        "L+2 calls. --dbl-bias sets the new head's output "
                        "bias. --init from a checkpoint without a head loads "
                        "everything else and starts the head fresh")
    p.add_argument("--dbl-ends", action="store_true", dest="dbl_ends",
                   help="FINE-TUNING TRAINING WHEEL, NOT BRIDGE: a Double "
                        "ENDS the auction where it is made. The contract is "
                        "the standing rung, doubled, and it scores through "
                        "the same code as always. Why: with the real rule "
                        "every arm escalates into a doubling ladder -- bid, "
                        "double, bid higher, double -- because the bidding "
                        "side can always run one more rung, and W1d measured "
                        "89.9% of contracts doubled, 76.4% slams and 23.0 "
                        "calls an auction against phase 1's 12.5. With this "
                        "the bidding side cannot run, so every bid has to be "
                        "one it will accept doubled, and the pressure "
                        "against overbidding comes from the opponents "
                        "instead of from a tuning constant. Rollouts also "
                        "get shorter, so the -3500 disasters are rarer and "
                        "the targets calmer. Removable by construction: "
                        "without the flag the code is bit-identical, and any "
                        "result under it MUST be re-checked without it")
    p.add_argument("--triple-q", action="store_true", dest="triple_q",
                   help="REWARD + ARCHITECTURE, ticket 008. Split the value "
                        "into Q_own + Q_denial + Q_double, three heads on one "
                        "trunk, three targets read off the SAME rollout that "
                        "sum exactly to the real zero-sum score, and ONE "
                        "argmax over the weighted sum. Q_own is phase 1 and "
                        "is the only term whose gradient reaches the "
                        "language. Turns the curriculum into two "
                        "coefficients instead of a sequence of games.")
    p.add_argument("--w-denial", type=float, default=1.0, dest="w_denial",
                   help="how much of Q_denial the policy reads. 0 is phase 1, "
                        "1 is the full zero-sum term. The head trains either "
                        "way, so this can be ramped without a restart.")
    p.add_argument("--w-double", type=float, default=1.0, dest="w_double",
                   help="how much of Q_double the policy reads. Same dial, "
                        "for what the double itself was worth.")
    p.add_argument("--triple-attached", action="store_true",
                   dest="triple_attached",
                   help="let the denial and double heads move the shared "
                        "trunk. Off by default: denial is the term that "
                        "destroyed the language in RESULTS 18 and 19.")
    p.add_argument("--route-blame", action="store_true", dest="route_blame",
                   help="REWARD. Grade a call by WHICH DECISION IT WAS. A bid, "
                        "and a pass that leaves the auction alive, are graded "
                        "on the acting side's own contract, as in phase 1. "
                        "The pass that ENDS the auction and a Double are "
                        "graded on the real zero-sum score. An opening bid "
                        "did not choose the doubled disaster three calls "
                        "later, and the zero-sum number is far easier to fit "
                        "at the last call than at turn 0.")
    p.add_argument("--route-scale", type=float, default=1.0,
                   dest="route_scale",
                   help="how much of the zero-sum price the terminal calls "
                        "pay under --route-blame. 0 is phase 1 row for row, "
                        "1 is the full price. The dial for annealing.")
    p.add_argument("--route-plain-own", action="store_true",
                   dest="route_plain_own",
                   help="under --route-blame, price a bid on the UNDOUBLED "
                        "own contract, as --own-bid does. Off by default "
                        "because it makes a sacrifice look free.")
    p.add_argument("--pass-baseline", action="store_true", dest="pass_baseline",
                   help="TARGET. Subtract the value of Pass AT THE SAME STATE "
                        "instead of a per-deal constant, so the net fits the "
                        "marginal effect of one call. Ticket 008. Free -- "
                        "Pass is always legal and already rolled out -- and "
                        "it cannot change any argmax.")
    p.add_argument("--par-baseline", action="store_true", dest="par_baseline",
                   help="regress score - par instead of score. A per-deal "
                        "constant, so it changes no decision; it removes "
                        "about 490 points of target noise per deal. Under "
                        "--own-bid the baseline is the side's own "
                        "cooperative par")
    p.add_argument("--no-double", action="store_true", dest="no_double",
                   help="ABLATION: take Double away. Expect escalation, "
                        "because then a sacrifice is cheaper than any game")
    p.add_argument("--inner-steps", type=int, default=1, dest="inner_steps",
                   help="OPTIMISER. Gradient steps taken on EACH batch of "
                        "rollouts. The rollout is ~70% of a step and the "
                        "target is fixed once it is computed, so a second "
                        "step on it costs almost nothing. Why it matters: "
                        "`slowroll` measured the net closing about 3 points "
                        "of a 300-point gap per step at lr 2e-4, while the "
                        "frozen copy that defines the target refreshes every "
                        "100 steps -- the target moves about as fast as the "
                        "net can chase it, which is what a limit cycle looks "
                        "like. This buys fitting speed per unit of rollout. "
                        "Note what this does to the refresh window: at "
                        "--refresh 100 the net now takes 100 x this many "
                        "gradient steps against one frozen opponent, so it "
                        "can actually converge to the target instead of "
                        "chasing it. The risk is the opposite one -- "
                        "overfitting to a frozen opponent it will never meet "
                        "again -- and that is what the arm is testing. "
                        "--steps still counts ROLLOUT "
                        "batches, so the net takes steps x inner-steps "
                        "gradient steps in total")
    p.add_argument("--refresh", type=int, default=100,
                   help="steps between refreshes of the frozen copy that "
                        "picks every call in every rollout")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=250, dest="eval_every")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--states", type=int, default=16)
    p.add_argument("--explore", type=float, default=0.2)
    p.add_argument("--roll-chunk", type=int, default=200000, dest="roll_chunk")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--d-hand", type=int, default=64, dest="d_hand")
    p.add_argument("--d-rung", type=int, default=48, dest="d_rung")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-test", type=int, default=20000, dest="n_test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--vulnerable", action="store_true")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default="")
    p.add_argument("--save", default="")
    a = p.parse_args()
    if a.own_bid and a.winner_only:
        p.error("--own-bid and --winner-only are two different games; "
                "pick one")
    if a.triple_q and (a.own_bid or a.winner_only or a.route_blame):
        p.error("--triple-q is its own reward rule; do not combine it with "
                "--own-bid, --winner-only or --route-blame")
    if a.triple_q and not a.dbl_head:
        p.error("--triple-q needs --dbl-head: without it Double shares the "
                "output layer with the bids and moves the rung embeddings, "
                "which is the exact failure RESULTS 19.5 measured")
    if a.triple_q and a.no_double:
        p.error("--triple-q has a term for what the double was worth; "
                "--no-double makes it identically zero")
    if a.route_blame and (a.own_bid or a.winner_only):
        p.error("--route-blame is its own reward rule; do not combine it "
                "with --own-bid or --winner-only")
    if a.route_blame and a.no_double:
        p.error("--route-blame exists to price Double; --no-double removes it")
    if a.pass_baseline and a.par_baseline:
        p.error("--pass-baseline and --par-baseline are exclusive")
    if a.inner_steps < 1:
        p.error("--inner-steps is a count of gradient steps; 1 is the old "
                "behaviour")
    if a.pass_cost < 0:
        p.error("--pass-cost is a price, not a sign; use 0 to turn it off")
    if a.pass_cost > 0 and not a.own_bid:
        p.error("--pass-cost only changes what a SILENT side is paid under "
                "--own-bid; it means nothing on its own. Add --own-bid.")
    if a.huber < 0:
        p.error("--huber is an elbow, not a sign; use 0 to turn it off")
    if a.clip_target < 0:
        p.error("--clip-target is a magnitude; use 0 to turn it off")
    if a.dbl_ends and (a.no_double or a.own_bid):
        p.error("--dbl-ends is a rule about Double, and --no-double / "
                "--own-bid take Double away, so it would do nothing. Drop "
                "one of them.")
    return a


if __name__ == "__main__":
    run(parse())
