# bridge_server

Three pages at https://bridge.localgeek.jp, one Flask app in one process:

- `/` — the **bid desk**: bid against a trained checkpoint, see the net's outputs
  (policy, Q, trick head, double heads), its inputs, and double-dummy scores. A
  dropdown picks the model per browser tab.
- `/play` — the **play desk**: watch the E48 card-play net play out a contract.
  When an auction on the bid desk ends, "Play it out →" carries that deal, that
  auction and that contract straight over.
- `/table` — **play a board**: you hold one chair and play that one hand, the nets
  hold the other three. Bid the auction, play the cards, get a duplicate score
  against par, deal again. A hint toggle turns on teaching notes before every call
  and every card.

## Run locally

    python3 -m emergent.bidserver                           # http://127.0.0.1:8787, /play and /table too

## Update to the newest snapshots, then deploy

    ./sync_models.sh      # bridgezero code + E21/E20b/E18 snapshots from ~/code/bridge_new
    ./deploy.sh

Server config (nginx, systemd) lives in `~/code/infra` (app `bridge`, port 3500).
The unit still passes `ck.pt` to `create_app`; that argument is ignored now.

## Models

`models/models.json` lists what the dropdown shows; the first entry is the
default. Two families are supported:

- bridgezero four-seat checkpoints (`bridgezero-fourseat-0.1`: E18, E20b, E21),
  copied by `sync_models.sh` without the critic. Edit its `copy` lines to add runs.
- phase 1 `SeatNet` checkpoints (`models/own_s16_step5250.pt`).

## Frozen copies

`bridgezero/` is copied by `sync_models.sh`. E28's classes (D5OWN4XC) are only on
branch `integrate-fast-xxsac`, so it copies from that worktree by default; after the
merge run `CODE=~/code/bridge_new ./sync_models.sh`.
`emergent/phase1.py` is the inference half of the phase 1 net from
`~/code/bridge/emergent/` (`exp11four.SeatNet`, `exp10four.St`, `fullinfo.SuitEncoder`),
with the same parameter names so its checkpoints load; the experiment scripts are gone.

## Machine APIs (`/apis/…`) — seat our bots at other sites' tables

`emergent/apis.py`, on the same app. Stateless: every request carries the whole position.
Default models are the first entries of `models/models.json` (bidding) and
`models/play_models.json` (card play); `model=<id>` / `play_model=<id>` pick another.
Dealer and vulnerability are real inputs here (the desks fix North / nobody).

- **BBO robot.php format** — `GET /apis/bbo.php?pov=S&d=N&v=-&n=..&e=..&s=..&w=..&h=1c-p`
  (hands `s.h.d.c` lowercase, `h` calls from the dealer joined by `-`, `v` one of `- n e b`,
  `botstyle` ignored). Answers `<sc_bm ...><r type="bid" bid="1N" meaning="..."/></sc_bm>`.
  Card play too: once the auction is over `h` goes on with the cards played
  (`...-4h-p-p-p-H9`) and the answer is `<r type="play" card="HJ"/>`; when dummy is on turn
  the declarer is asked. Only what `pov` can see is read: its hand, dummy after the lead.
- **Brill Seat Robot API** (https://brill.aalborgdata.dk/seat-api.html), base URL
  `https://bridge.localgeek.jp/apis/brill`: `GET /` (200), `/bid`, `/lead`, `/play`.
  `/bid` answers `bid`, `alert`, `explanation` (from `models/corpus_<id>.json`), `candidates`.
  Card play uses the play net; unseen cards are random filler that never reaches its input
  (checked in `tests/test_apis.py`). §11 answers: one process serves all four seats, no state
  between requests; cold start = server start (models load at boot), then ~10–50 ms per
  request; `meanings` and `matchtype` are accepted and ignored; one system per model id.

    python3 -m pytest -q tests/test_apis.py      # Brill's §9 checklist + robot.php, full boards

## Why this call (the section under the bidding box)

Three parts, all from the deployed net alone:

1. **Its own numbers** — its chance for its call and the runner-up, and how far below the
   best possible contract it expects to finish after each (`why()` in `bidserver.py`).
2. **What this call means** — `models/corpus_<model id>.json`, built offline from 1,000,000
   greedy self-play auctions (dealer North, no vulnerability, the desk's setting) by
   `bridge_new/experiments/explain/selfplay.py` + `corpus.py`. Positions with fewer than 200
   hands are dropped; 8 calls deep.
3. **Play it out** — `emergent/explain.py`: 256 random completions of the unseen 39 cards,
   weighted by how likely the net thinks the calls so far were, then the net finishes the
   auction on each. About 1.7 s on the droplet, in a background thread, newest request wins.
   The samples are the net's guess: the real hand ranks around the 18th percentile of them,
   and the weights are softened (eps 0.02, beta chosen for ESS = 25% of the samples).

## The play desk (`/play`)

`emergent/playdesk.py` registers `/play` and `/api/play/*` on the same Flask app,
and `emergent/bidserver_static/play.html` is the page. The net runs through the
exact classes it was trained with, copied by `sync_models.sh`:
`bridgezero/play/{data,model,match}.py` and `bridgezero/bridge/play.py`.

Where a board comes from, in order of preference:

1. **A link from the bid desk** — `#d=<deal>&a=<auction>&dr=0&v=none`. Same deal
   codec and same call codec as the bid desk, so the two pages share links.
2. **The frozen benchmark** — `models/bench_100k.npz`, the slice every E48 number
   was measured on. `#b=<row>` loads one. Each row brings its own E46 auction,
   contract, vulnerability and stored `dd_tricks`, so the page can say what the
   benchmark thought the board was worth. Only the first `PLAY_BENCH_LIMIT` rows
   are read (4000 by default); loading all 100k takes ~30 s.
3. **Configured on the page** — a random deal or the cards already on the table,
   plus level, strain, declarer, doubled and vulnerability. There is no real
   auction then, so the net is handed the shortest one that lands in the
   contract: declarer bids it, everybody passes. The page says so.

The page shows the net's whole information set for the seat on turn (its own
hand, the face-up hand, every card played by relative seat, the contract, both
readings of the auction, all 740 input numbers), its probability over the 52
cards with the illegal ones greyed out, and its belief head as a per-card split
over the hands it cannot see, with the true holder as the cell border.

Double dummy comes from endplay/DDS, the same solver that produced the
benchmark's `dd_tricks`: `calc_all_tables` for the board, `solve_board` at the
live position for what each card is worth, and one solve per card after the play
for the card-by-card review. 52 solves cost about 50 ms. If endplay is missing or
throws, every DD field is `None` and the page says there is no answer rather than
guessing.

`models/play_models.json` lists the checkpoints the dropdown shows, like
`models/models.json` does for the bid desk.

## Play a board (`/table`)

`emergent/tabledesk.py` registers `/table` and `/api/table/*`, and
`emergent/bidserver_static/table.html` is the page. It adds no model of its own:
the auction runs through the bid desk's `MODELS` and the play through
`playdesk.MODELS` and a `playdesk`-shaped game dict, so a board here is played by
exactly the code the other two pages use. `python -m emergent.bidserver` imports
the bid desk twice, so `tabledesk.load(sys.modules[__name__])` hands over the
module that actually loaded the checkpoints instead of importing it again.

You hold one chair and the nets hold the other three. North deals and nobody is
vulnerable, the same setting as the other two desks, because the "what this call
means" corpus was built that way.

**The play follows a real table.** Declarer plays both of the declaring side's
hands: when you declare, you pick the card for your own hand and for dummy's
(face up opposite you after the opening lead). When you are dummy you play
nothing — your partner, the net, declares and plays both hands while yours lies
face up. As a defender you play your own hand. Each hand on the page is labelled
with who plays it.

The board is scored with duplicate scoring and compared against `dd_par_score`,
in points and in IMPs. `result.par` is that par from your side; `par_ns` and
`par_mine` are the same numbers under the older names.

### Sharing a board

The address bar always holds the position on screen, so copying it (or pressing
**Copy link** in the header) shares exactly that board:

    /table#d=<deal>&a=<calls>&dr=0&v=none&p=<cards>&s=<N|E|S|W>&m=<bid model>&pm=<play model>&h=<0|1>
    /table#d=WsFp3_tWPNyvIKiARQ&a=AD&dr=0&v=none&s=S&m=D_cw_s75k&pm=E48_wideleagueH&h=0

- `d`, `a`, `dr`, `v` and `m` are the bid desk's (`bidserver.encode_deal`, one
  `CALL_CHARS` character per call) and `p` is the play desk's (one `CARD_CHARS`
  character per card, in play order), so the same link also opens on `/` and `/play`.
  `a` and `p` are left out while empty. `dr` and `v` are always `0` and `none` here.
- `s` is your chair, `pm` the card-play model id, `h` hints on/off. Speed, search
  and the solver peek are the viewer's own settings and are not in the link.

The page keeps the hash in sync with `history.replaceState` (no reload, no history
entry per card; `state_dump` sends the pieces as `code`). While the nets' moves are
still being played out on screen, `a` and `p` are cut back to the frame being
drawn, not to where the server already is.

Opening a link posts it to `POST /api/table/load`, which builds a new game in the
visitor's own `X-Game` — nothing is shared between people. Everything is checked
(`board_from_link`): the deal code, the chair, that both models exist on this
server, every call legal in turn by the rules, every card legal in turn and only
after an auction that produced a contract. Anything wrong is a JSON 400 naming the
first bad piece; the page then says the link could not be opened and deals a fresh
board.

The board stops exactly at the link's position, even when a net is on turn there:
`load` never calls `advance`, and the page holds its auto-advance and shows
**Continue from here** until the visitor presses it (or plays, or starts something
new). From there the nets act as usual. With search on, a net's next card may
differ from the one the sharer saw — the link is the position, not its future.

**The link contains all four hands** — it has to, to rebuild the board — and so
does the `code` in every state payload. The page still draws only what the
visitor's chair may see, exactly as in normal play: their own hand, dummy after
the opening lead, all four once the board is over. Anyone who decodes the URL
knows the deal, so share mid-board links with people who won't.

### Playing the nets' moves out

`advance` can answer with a whole trick, or with the rest of the deal when the
user is dummy. Dropping that on screen in one jump is unreadable, so the page
holds the payload and walks to it one call or one card at a time, deriving every
intermediate frame from the state it already has — **no extra round trips**. A
card animates in from its seat, the finished trick pauses to be read, then slides
to the side that won it. `frameOf()` in `table.html` rebuilds a whole synthetic
state for a given `(nCalls, nCards)`, so the drawing code never learns that an
animation exists.

A speed control in the header (slow / normal / fast / instant) is remembered in
`localStorage`; `prefers-reduced-motion` drops the transforms and defaults to
fast. Clicking the table, or the Skip button, jumps to the end of what the nets
just did. Nothing is clickable and no hint is shown until the page has caught up.

### The page is the product; `/play` is the lab

`/table` shows a bridge game and a teacher: the felt, the hands, the trick, the
score, a short block of plain-English advice and one big button. Every
percentage, sample count and policy grid now sits inside a **"Show the numbers"**
disclosure, closed by default, with its source tags intact. Nothing was deleted —
and the raw article (full policy grids, the 740 inputs, the belief map, card-by-card
double dummy) is what `/play` is for. `/play` is untouched.

**The Next button** commits whatever is highlighted: the ringed call during the
auction, the ringed card during the play, "Next board" once the deal is scored,
"Play the rest out" while the others are still going. Enter or space does the
same. Clicking a call or a card directly still works, and hovering a call in the
bidding box previews what that call would say without committing to it.

**Hints are off by default**, on every new game. With them off the page shows no
suggestion of any kind, and the payload carries none: `hint` and `suggest` are
both `null`, and `POST /api/table/explain` refuses. The Next button then only says
whose turn it is. With them on, `suggest` comes from the hint's own forward pass
(`suggest_from_hint`). A choice made with the toggle lasts for the rest of that
browser tab's session, new boards included.

Below 980px the Next button is fixed to the bottom of the viewport, so a board can
be played one-handed.

**Nothing on the table moves when a card is played.** The felt is a grid of fixed
tracks; every hand keeps a slot for all thirteen of its cards for the whole board
(played ones stay in place, faded; a hidden hand keeps thirteen backs, the played
ones blank), cards shrink rather than wrap when a suit is long, the auction
scrolls inside the centre instead of growing it, and the long model line in the
header is one line with an ellipsis. Only transforms and opacity animate.

### Saying the corpus out loud

The teaching text is generated by template functions in `tabledesk.py`
(`holding_words`, `showed_sentence`, `would_show_sentence`, `fit_sentence`),
never by free text. Every number in a sentence is read out of a corpus entry:

- `stats.hcp` is [5th, 50th, 95th percentile] → "12 to 17 points".
- `stats.suit_len` is the share of hands holding **at least 4, at least 5, at
  least 6** cards of the named suit — *not* 5/6/7. `promised_length()` takes the
  longest one at least half the hands had → "five or more spades". If none
  reaches half, the sentence says so instead: "four or more clubs only about 30%
  of the time".
- `stats.bal` is the balanced share → "a balanced hand" above 0.75, "an
  unbalanced hand" below 0.15, nothing in between.

The thresholds (`PROMISE_AT`, `BALANCED_HIGH`, `BALANCED_LOW`) are named
constants: they are our judgement, and the wording around them is ours, but every
quantity is measured. `fit_sentence` compares the user's actual hand against the
same promised length the sentence quoted, so the prose and the fit test can never
disagree. Nothing the corpus does not measure is said at all — in particular the
page never claims a stopper, a control, or what a call *asks for* as opposed to
what it showed.

### What the hints may say

The net has no explanation head, so nothing on the page claims a reason it did
not have. Every claim carries a tag saying where it came from:

- **the net** — the deployed net's own output for the seat on turn, given exactly
  the user's information set (their hand, the face-up hand, the cards played).
  During the auction: its policy and Q over the legal calls. During the play: its
  policy over the 52 cards and its belief head, cut down to the missing honours,
  with the true holder stripped out until the deal is over.
- **self-play** — `models/corpus_<model id>.json`, a million greedy self-play
  auctions summarised by prefix (`bridge_new/experiments/explain/corpus.py`).
  What a call actually held: point band, balanced share, suit lengths. Positions
  with fewer than 200 hands, and anything past eight calls, are not in the file.
  There the advice leads with the net's own policy on the user's cards
  (`net_sentence`: "You have 13 HCP and 5 hearts: with exactly these cards the net
  bids 1♥ 78% of the time"), and quotes the nearest auction the corpus does hold
  (`nearest_position`), tagged **approximate** with what was changed to reach it:
  opening passes left out, an opponent's call replaced by a pass, or the earliest
  calls dropped (keeping at least a round). Every stand-in keeps the user on turn
  and partner two calls back. "If you bid X, partner usually answers…" walks two
  nodes further on and prints the opponent call it assumed to get there.
- **the cards** — arithmetic on what the user can already see: who is winning the
  trick, whether they must follow suit, how many trumps are unaccounted for,
  whether a card is a certain winner, whether a finesse is available. The finesse
  test only fires for declarer, who can see both hands, and only when a low card
  in one hand faces an honour in the other with a higher card still out.
- **standard advice** — an ordinary club rule of thumb, ours, shown only for the
  opening lead. The play advice line (`play_advice`) names the card, how often the
  net plays it here, one fact about it, and a second choice once that is at least
  10% likely; it never says *why* the net chose it, because the net cannot say.
- **solver** — double dummy, which looks at all four hands. Off by default,
  always labelled. It also drives the after-the-deal review, which re-solves the
  board card by card (`playdesk.review`) and lists the user's own cards that cost
  a trick against a defence that never errs.

### Asking the server for things

Every POST sends a JSON body, even an empty one, and every endpoint reads it with
`body_of(request)` (`get_json(silent=True) or {}`). Declaring
`Content-Type: application/json` and then sending nothing makes Flask's
`request.json` raise, and the 400 comes back as an HTML page — which is how "New
board" and "Next board" died while the seat buttons, which happen to send
`{seat}`, went on working. Both ends are fixed so neither alone can bring it back.

Failures are shown, never swallowed: `post()` reads the body as text before
parsing, so a proxy error page reports its status instead of a bare SyntaxError,
and `go()` puts whatever went wrong in the header's error slot and the console.

Requests are serialised, but a click that starts something new — new board, a
different chair, restart, play the rest out — goes through `goNow()`, which bumps
a generation counter, cancels the pending auto-advance and abandons the queue.
Work from an older generation neither fires nor draws, so a button never has to
wait out a run of `advance` calls it has just made irrelevant.

"Where does each call lead?" reuses `emergent/explain.py` unchanged, including
its guard against the effective sample size collapsing: `auction_weights` mixes
2% of a flat policy into every call and flattens the log weights by a `beta`
chosen so the ESS reaches 25% of the samples. The page prints the ESS next to the
sample count. 128 samples by default here, half what the bid desk uses.
