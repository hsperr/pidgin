# Results

How the released models score. Every number names its board count and its opponent.
`±` is one standard error. "Fresh" rows are measured with
[`scripts/make_results.sh`](../scripts/make_results.sh) on the released files; its output
lands in `results/`. "Paper" rows come from the paper draft (Sperr and Dali, in
preparation), measured on weights identical to the released file. "Notes" rows come from
the project's training notes, with the released file.

Models:

| Name | File | What |
|---|---|---|
| Pidgin V1 | `D_cw_s75k.pt` | bidding net ([pidgin_v1.md](pidgin_v1.md)) |
| Pidgin V2 | `pidginv2_bid_s40000.pt` | bidding net ([pidgin_v2.md](pidgin_v2.md)) |
| BRL | `brl_fsp_weights.npz` | bidding net by Kita et al. 2024 (external) |
| policy net | `play_E48_wideleagueH.pt` | card play, self-play only ([card_play.md](card_play.md)) |
| earlier policy net | `play_E48_leagueE.pt` | card play, the run before |
| Q-net | `play_B2g_s540k.pt` | card play, values from the solver |
| belief net | `belief_r2.pt` | reads an auction, guesses the hidden hands ([belief.md](belief.md)) |

## Bidding strength

Duplicate matches of the bidding alone. Each deal is played at two tables with the seats
swapped; the contract is scored with double-dummy tricks, in IMPs per board, positive for
the first-named side. Standard errors are clustered by deal.

**Fresh.** 160,000 boards: the 10,000 held-out deals (dataset index 99,990,000 on) x 4
dealers x 4 vulnerabilities. `tools/match.py`, greedy calls, no bidding search.

| Match | IMPs/board |
|---|---|
| Pidgin V2 vs Pidgin V1 | +0.37 ± 0.03 |
| Pidgin V1 vs `rule:sayc` | +3.18 ± 0.05 |
| Pidgin V2 vs `rule:sayc` | +3.39 ± 0.05 |
| Pidgin V1 vs punisher | −1.79 ± 0.03 |
| Pidgin V2 vs punisher | +0.00 ± 0.03 |

`rule:sayc` is the small rule bidder in this repo, not a full SAYC robot. The punisher
bids like Pidgin V1 but doubles exactly the contracts that go down double dummy. It
measures how much a side loses to a doubler that always knows.

**Paper.** Pidgin V1 only (the draft predates V2), same held-out deals.

| Match | Boards | IMPs/board |
|---|---|---|
| Pidgin V1 vs BRL | 160,000 | −0.38 ± 0.03 |
| Pidgin V1 vs BRL's supervised net (Kita et al., not in this repo) | 160,000 | +0.56 ± 0.03 |
| Pidgin V1 vs EPBot 2/1 | 8,000 | +0.62 ± 0.07 |
| Pidgin V1 vs EPBot SAYC | 8,000 | +0.65 ± 0.07 |
| Pidgin V1 vs EPBot Acol | 8,000 | +0.69 ± 0.07 |
| Pidgin V1 vs EPBot Polish Club | 8,000 | +0.67 ± 0.07 |
| Pidgin V1 vs EPBot Precision | 8,000 | +0.73 ± 0.07 |

EPBot is a rule-based bridge engine with five convention cards.

Where Pidgin V1 loses to BRL (paper, the same 160,000 boards):

| Boards | Share | IMPs/board on them |
|---|---|---|
| only V1's contract is doubled | 17.0% | −2.16 |
| only BRL's contract is doubled | 4.2% | −0.50 |
| both doubled | 1.4% | −0.47 |
| no double at either table | 77.4% | +0.02 |

## Openings

**Fresh.** 100,000 held-out hands per seat and vulnerability, the model's greedy call
with no earlier calls but passes. `tools/openings.py`. Rates have a standard error of
about 0.2 percentage points.

How often each bidder opens (in brackets: the HCP at which half the hands open):

| Seat | V1 not vul | V1 vul | V2 not vul | V2 vul | BRL not vul | BRL vul |
|---|---|---|---|---|---|---|
| 1 | 68.5% (8) | 56.6% (10) | 68.3% (8) | 55.1% (10) | 80.5% | 78.8% |
| 2 | 60.6% (9) | 53.1% (10) | 60.0% (9) | 51.5% (10) | 77.0% | 78.3% |
| 3 | 52.3% (10) | 40.8% (12) | 61.3% (9) | 43.1% (11) | 99.5% | 99.6% |
| 4 | 47.0% (11) | 40.4% (12) | 47.3% (11) | 42.3% (11) | 96.4% | 98.7% |

BRL opens most hands with any HCP: its 1♣ is a catch-all (median 7 HCP, 27% hold four or
more clubs).

Light openings (share of 0–7 HCP hands that open, preempts included) and 1NT/2NT on an
unbalanced hand (share of NT openings):

| | V1 | V2 | BRL |
|---|---|---|---|
| light, seat 1 not vul | 13.1% | 17.3% | 98.8% |
| light, all seats | 4.4% | 6.5% | 96.2% |
| unbalanced NT, all seats | 64.0% | 66.5% | 21.8% |

Every opening of Pidgin V1 and V2, seat 1, not vulnerable. HCP is the 5% / 50% / 95%
range; 5+ is the share with five or more cards in the named suit:

| Call | V1 share | V1 HCP | V1 5+ | V2 share | V2 HCP | V2 5+ |
|---|---|---|---|---|---|---|
| 1♣ | 13.0% | 9 / 12 / 17 | 69% | 15.0% | 9 / 12 / 17 | 54% |
| 1♦ | 12.6% | 8 / 11 / 16 | 77% | 14.7% | 9 / 12 / 17 | 59% |
| 1♥ | 19.0% | 7 / 11 / 16 | 62% | 13.8% | 8 / 12 / 16 | 64% |
| 1♠ | 17.6% | 7 / 11 / 15 | 66% | 9.9% | 8 / 11 / 15 | 91% |
| 1NT | 5.5% | 14 / 18 / 22 | | 4.1% | 15 / 19 / 23 | |
| 2♣ | 0.0% | | | 3.0% | 3 / 8 / 11 | 100% |
| 2♦ | 0.1% | | | 3.0% | 4 / 8 / 11 | 100% |
| 2♥ | – | | | 2.2% | 3 / 7 / 10 | 100% |
| 2♠ | – | | | 1.2% | 5 / 7 / 10 | 100% |
| 3-level and up | 0.7% | | | 1.5% | | |
| Pass | 31.5% | 2 / 6 / 9 | | 31.7% | 2 / 6 / 9 | |

Observations:

- 1NT is the strong opening of both: 14–23 HCP, about two thirds unbalanced, and nearly
  every hand with 20+ HCP.
- V2 has weak two-bids in all four suits, on six-card suits about 90% of the time.
  V1 has none: with 7+ spades and at most 10 HCP it opens 1♠ (94%), V2 opens 2♠ (85%).
- V2's 1♠ promises five cards far more often than V1's (91% against 66%).
- 1-level suit openings of both hold four or more cards in the suit (99.9% or more).

## Simplicity

A code word is a call partner cannot read at face value: a suit bid without four cards
(three when raising partner), a double of a contract at level 3 or below, a redouble, or
an artificial 2♣ opening ([scripts/README.md](../scripts/README.md)).

**Fresh.** 8,000 self-play auctions per bidder (all four seats are the same net),
`tools/simplicity.py --boards 8000`. Rates are per 100 auctions unless named otherwise.
Standard error of the code-word rate: about 0.4.

| | Pidgin V1 | Pidgin V2 |
|---|---|---|
| code words / 100 auctions | 9.9 | 10.4 |
| … suit bids without length | 5.2 | 2.8 |
| … low doubles | 4.4 | 7.5 |
| … redoubles | 0.24 | 0.01 |
| calls per auction | 10.8 | 11.3 |
| code words / 100 calls | 0.92 | 0.92 |
| suit bids with length | 98.7% | 99.3% |
| both sides bid | 67% | 60% |

V2 bids short suits half as often as V1 and makes more low doubles.

**Paper.** 8,000 boards; only the named side's calls, per 100 partnership auctions.
Pidgin V1 faces EPBot 2/1, BRL faces Pidgin V1, the EPBot systems face an earlier Pidgin
net. The denominator differs from the table above, so compare within this table only.

| System | Code words / 100 | without X/XX | suit bids with length | low X / 100 | XX / 100 |
|---|---|---|---|---|---|
| Pidgin V1 | 5.3 | 2.6 | 98.7% | 1.8 | 0.94 |
| EPBot 2/1 | 37.4 | 21.3 | 86.5% | 16.0 | 0.05 |
| EPBot SAYC | 37.1 | 21.1 | 86.7% | 16.0 | 0.06 |
| BRL | 84.5 | 50.4 | 72.7% | 33.7 | 0.33 |

## Card play

**Fresh.** Plain nets, no search. `python -m training.play.match` on
`server/models/bench_100k.npz`: 49,927 boards with a fixed contract. The challenger
declares at one table, the reference at the other; the policy net defends at both.
IMPs per board, positive for the challenger.

| Declarer vs declarer | IMPs/board | Declarer tricks |
|---|---|---|
| policy net vs random cards | +6.30 ± 0.03 | 9.07 vs 8.57 |
| earlier policy net vs random cards | +5.96 ± 0.03 | 9.06 vs 8.57 |
| policy net vs earlier policy net | +0.44 ± 0.02 | 9.07 vs 9.06 |

**Notes.** The Q-net needs its own player, not `training.play.match`. Both nets play every
seat for their side; the policy net defends at both tables.

| Match | Boards | Result |
|---|---|---|
| Q-net vs policy net, plain nets, IMPs/board | 5,990 | +0.93 ± 0.07 |
| tricks lost to a perfect defence when declaring: Q-net / policy net | 999 | 0.47 / 0.71 |
| tricks lost to a perfect declarer when defending: Q-net / policy net | 999 | 0.50 / 0.68 |
| Q-net + PIMC vs policy net + PIMC (the served players), IMPs/board | 5,990 | +0.05 ± 0.06 |
| policy net + PIMC vs policy net plain, IMPs/board (defence: an earlier unreleased net) | 3,994 | +0.99 ± 0.06 |

The PIMC rows search as the served players do: 20 layouts of the hidden cards sampled
from the policy net's belief head, each solved double dummy, the card with the best total
played. With search on, the two nets are level within one standard error.

## Belief net

**Fresh**, with `belief/analyze.py` on the float32 training checkpoint that `belief_r2.pt`
was cut from (same weights, before the cut to float16 and the system head). Test set:
20,000 held-out auctions from many bidders (Pidgin nets, BRL, EPBot systems, WBridge5),
each read from all four seats; two of the bidders never appear in training. Per hidden
card, three choices (left, partner, right). Systems not told. With over a million hidden
cards per row, standard errors are below the printed precision.

| Calls heard | Card log-loss | Card right | Suit length error | HCP error per hand |
|---|---|---|---|---|
| 0 | 1.099 | 33.3% | 1.03 | 3.12 |
| 1 | 1.088 | 36.7% | 0.98 | 2.84 |
| 2 | 1.077 | 39.4% | 0.93 | 2.57 |
| 4 | 1.059 | 42.8% | 0.83 | 2.15 |
| 6 | 1.050 | 44.0% | 0.77 | 1.99 |
| 8 | 1.042 | 44.8% | 0.73 | 1.86 |
| 10 | 1.038 | 45.2% | 0.71 | 1.81 |
| full auction | 1.041 | 44.8% | 0.71 | 1.78 |

Rows past 4 calls include only auctions that long (18,914 at 6, 12,413 at 10). Errors are
mean absolute errors of the expected length (cards) and HCP. Random guessing gives 1.099
and 33.3%. A guesser told every hidden hand's exact suit lengths, and nothing else, reaches
1.012 and 47.1%. Told both sides' systems, the full-auction numbers are 1.037, 45.2%,
0.70 and 1.59.

Most of the information arrives in the first four calls. At the end of the auction the
net places 44.8% of hidden cards correctly against 47.1% for exact shape knowledge.

## Full teams (bidding and play)

**Notes.** Full boards: auction, then card play with PIMC, duplicate, scored on the real
cards. 4,000 boards from deals at dataset index 61,800,000 on. Teams as in
`server/models/teams.json`.

| Match | Boards | IMPs/board |
|---|---|---|
| PidginV2 team vs PidginV1 team | 4,000 | +0.33 ± 0.10 |
| V2 bidding with search vs BRL bidding, the same Q-net card play at both tables | 4,000 | −0.21 ± 0.11 |

In the second row both sides play with the Q-net and PIMC, with shape-first layouts and
search on the opening lead; that is close to, but not exactly, the served card player.
PidginV2 bids with the belief-net search ([belief.md](belief.md)), the other bidders
without it.
