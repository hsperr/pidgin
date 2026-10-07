# Pidgin — bridge bidding from self-play

Train a bridge bidder from random weights through the Pidgin V1 recipe: grounding,
own-contract self-play, a simplicity fine-tune, and table-score self-play.
The bidder sees its own hand and public calls. Double-dummy (DD) trick tables
show how many tricks can be taken with perfect play and all hands visible. They
supply training targets and rewards; they are never bidder inputs.
No expert auctions or bidding labels are required.

## What is here

| Folder | What |
|---|---|
| `training/` | The model code: bidding nets, trainers, card play. Shared by training and the server. |
| `server/` | The site and APIs that serve the models ([docs/serving.md](docs/serving.md)). |
| `belief/` | Belief net for the bidding search ([docs/belief.md](docs/belief.md)). |
| `qnet/` | Card-play Q-net trainer and its data pipeline ([docs/card_play.md](docs/card_play.md)). |
| `docs/` | How each served model was trained: [Pidgin V1](docs/pidgin_v1.md), [Pidgin V2](docs/pidgin_v2.md), [belief](docs/belief.md), [card play](docs/card_play.md). How they score: [results](docs/results.md). How simple they bid: [simplicity](docs/simplicity.md). |
| `scripts/` | Commands to use the models: download, bid, generate auctions, serve ([scripts/README.md](scripts/README.md)). |
| `tools/` | Matches, dashboard, analysis. |

## Results at a glance

| | |
|---|---|
| Pidgin V2 vs Pidgin V1, bidding | +0.37 ± 0.03 IMPs/board (160,000 boards) |
| Pidgin V1 vs BRL, bidding | −0.38 ± 0.03 IMPs/board (160,000 boards) |
| Pidgin V1 vs EPBot, five systems | +0.62 to +0.73 IMPs/board (8,000 boards each) |
| Pidgin V2 team vs Pidgin V1 team, full play | +0.33 ± 0.10 IMPs/board (4,000 boards) |
| Q-net opening leads (DDOLAR / ADDOLAR) | 81.3% / 74.7%, level with top experts |
| Belief net, hidden cards placed after the auction | 44.8% (exact shapes would give 47.1%) |

Details, board counts and how to rerun: [docs/results.md](docs/results.md).

## Set up

Use Python 3.10 or newer. From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r server/requirements.txt
scripts/get_models.sh
```

The download puts released weights in `server/models/`. Training and model
comparisons also need the DD dataset described below.

## Use the released models

```bash
python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P" --model PidginV1
python scripts/generate_auctions.py --n 100000 --out auctions.npz --pbn auctions.txt
scripts/serve.sh                                         # the site and APIs on localhost:8787
```

The first command prints the chosen call and the four most likely legal calls:

```text
PidginV1 as S: 1S
   1S  0.916
```

The hand uses spades.hearts.diamonds.clubs, with `T` for ten. `--auction` is the
public calls so far (`P` means pass); the hand belongs to the next player to call.
`--model` picks a team (`PidginV1`, `PidginV2`, `BRL`) or a checkpoint file.
These command-line tools use the bidding network directly. The Pidgin V2 team
through the Brill API also uses bidding search, so its calls can differ from
the scripts.
`generate_auctions.py --help` describes its options and output format.

## Quick start

Run commands from this checkout:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
./train.sh --smoke runs/smoke
python tools/dashboard.py --runs runs/smoke --data data/smoke_128.npz --deals 16
```

Open <http://localhost:8770>. The smoke run uses the included 128-deal fixture,
a tiny network, and two updates per stage. It checks the pipeline, not strength.
Training currently runs on CPU; `THREADS` controls PyTorch's thread count.

## Training data

Download `dds_results_100M.npy` (3.2 GB) from
[`sotetsuk/dds_dataset`](https://huggingface.co/datasets/sotetsuk/dds_dataset).
The dataset is Apache-2.0 and was generated with
[PGX](https://github.com/sotetsuk/pgx). See its dataset card for the paper citation.

```bash
python -m pip install huggingface_hub
hf download sotetsuk/dds_dataset dds_results_100M.npy --repo-type dataset --local-dir data
./train.sh
```

Alternative paths and settings:

```bash
DATA=/path/to/dds_results_100M.npy THREADS=8 SEED=1 ./train.sh runs/my_run
CODE_WORD_PENALTY=0 ./train.sh runs/no_simplicity_cost
```

## Train Pidgin V1

`train.sh` runs four stages. The folder names are checkpoint paths used by the
script; `4_D` is the final Pidgin V1 stage. IMPs (International Match Points)
measure the score difference between two tables playing the same deal.

| Output directory | Training | Checkpoint selection |
|---|---|---|
| `1_ground` | Learn what contracts can make from DD results, then learn to bid while opponents pass | Contract score with opponents passing |
| `2_own` | All four seats bid; each partnership learns to score its own final contract | Own-contract score |
| `3_simple` | Continue stage 2 for up to 30,000 steps, charging for calls flagged as hard to read | Own-contract score |
| `4_D` | Learn from final contract scores, with the same call cost and past versions as opponents | Paired IMPs against `3_simple/last.pt` |

Stage 2 starts from `1_ground/best.pt`; stages 3 and 4 start from the preceding
stage's `last.pt`. The final selected model is **`runs/training/4_D/best.pt`**.
Pidgin V1 enables doubles and redoubles at any legal seat. Its starting policy is
measured and kept as a fallback when updates fail to improve or trip a guard.
Changing the legal doubling policy at this transition can change play even
before the first update, so its starting IMP score need not be zero.

This recipe trains from scratch and selects its final checkpoint against the
preceding stage. Results vary with training and may differ from the released
Pidgin V1 weights. No pretrained weights are needed.

Default dataset ranges are zero-based, with exclusive ends:

| Purpose | Deals |
|---|---|
| Validation and checkpoint selection | `[3,028,000, 3,033,000)` |
| Training pool | `[3,033,000, 99,990,000)` |
| Final reports and default matches | `[99,990,000, 100,000,000)` |

Training shuffles full one-million-deal blocks. A partial final block is dropped;
episodes sample within each block **with replacement**, so a sweep does not
visit every deal. Grounding validates every 1,000 steps and stops after 5,000
without improvement. Own-contract training completes its block sweep unless
a guard trips. Pidgin V1 validates every 3,000 steps and stops after 12,000 without
improvement. Every stage logs metrics and saves its run settings.

Completed stages are skipped when rerunning the script with the same output
folder. Use a new folder when changing settings. Interrupted four-seat stages
can resume with `python -m training.fourseat.train --resume`, the same
`--out`, and the original options recorded in that stage's `run.json`.
Grounding has no resume support; use a fresh output folder if it is interrupted.

## Dashboard and analysis

```bash
python tools/dashboard.py --runs runs/training
python tools/openings.py runs/training/4_D/best.pt
python tools/match.py --a four:runs/training/4_D/best.pt \
  --b four:runs/training/3_simple/last.pt --out results/pidgin_v1_vs_parent
python tools/simplicity.py runs/training/4_D/best.pt --boards 4000
python tools/weakspots.py results/pidgin_v1_vs_parent
```

Pass `--data` to each analysis command for a different dataset. On the smoke
fixture, also pass `--deals 16` to `openings.py` and `match.py`.
The dashboard's `--data` and `--deals` apply to both openings and match jobs.

The dashboard charts grounding, training, validation, and simplicity metrics.
Its best-step marker follows the trainer's accepted checkpoint. It can run
opening analysis and IMP matches against earlier checkpoints and the bundled
`sayc`, `weakclub`, and `happy` rule bidders. These are lightweight diagnostic
bots, not complete implementations of established bridge systems. The default
recipe trains against itself and its own league. Use dashboard `--reference PATH`
to add a public checkpoint and a DD-oracle punisher variant to diagnostic matches.
The punisher doubles failing contracts using hidden DD results; it measures
exposure to punishment, not the strength of a realistic opponent.

`match.py` plays duplicate boards at two tables with the players swapped.
By default, it uses the last 10,000 deals with all 4 dealers and 4 vulnerability
patterns. `--boards` selects a seeded subset; `--start` and `--deals` choose
a different slice. It checks self-match symmetry, replays sampled auctions
through the reference scorer, and reports a deal-clustered bootstrap interval.
The match output includes auctions for simplicity and weak-spot analysis.

Keep evaluation deals out of checkpoint selection. Repeated dashboard inspection
of one slice makes it diagnostic; use fresh deals and several training seeds
for a strength claim. In-training rewards are not a strength estimate.

## Simplicity

How Pidgin's openings compare with BRL and SAYC: [docs/simplicity.md](docs/simplicity.md).

A **code word** is a call flagged by any of these rules:

- A suit bid with fewer than 4 cards, or fewer than 3 when raising partner.
- A double of a contract at level 3 or below.
- A redouble.
- A 2♣ opening with fewer than 5 clubs or at least 20 high-card points (HCP).

Each flagged call counts once, even when multiple rules apply. The trainer logs
`code_words_per_100` on greedy validation self-play and `code_word_share` on
sampled training calls. The 0.2 cost is in score units of 100 points and is
charged to that call's policy advantage; it does not alter the reported IMPs.
Set `CODE_WORD_PENALTY` to change it in stages 3 and 4.

`tools/simplicity.py` measures self-play or saved match auctions and also
reports natural suit bids, cue bids, jumps, 4NT, and auction length:

```bash
python tools/simplicity.py --boards-npz results/pidgin_v1_vs_parent/boards.npz
```

For a match between different players, the Python `analyse(..., who="A")`
API counts A's calls across both tables; the default counts all calls at table 1.
This heuristic measures face-value readability, not human learnability.
Compare both IMPs and simplicity when assessing a model.

The sacrifice value head ranks candidate contracts using a DD counterfactual
where the opponents double and everyone passes. Actual opponents can respond
differently. Pidgin V1's policy reward uses the final auction, but candidate ranking
still uses this proxy; inspect sacrifice credit and positive share alongside IMPs.

## Public contents and license

The repo holds the model code, training recipes, server, scripts, tools, tests and a
small test fixture. Weights are on Hugging Face:
[hsperr/pidgin](https://huggingface.co/hsperr/pidgin) (`scripts/get_models.sh`).
Training outputs, large datasets and working notes are not in Git.

Apache-2.0. See [LICENSE](LICENSE).

### Third-party

- **BRL** (the `BRL` team's bidder, `brl_fsp_weights.npz`) is not ours. It is the FSP
  bidding model from [harukaki/brl](https://github.com/harukaki/brl), by Kita et al.,
  "A Simple, Solid, and Reproducible Baseline for Bridge Bidding AI", IEEE CoG 2024.
  Apache-2.0; its licence ships with the weights as `brl_LICENSE`.
  `server/emergent/brl_player.py` is a PyTorch port of its network.
- **Training data**: the [`sotetsuk/dds_dataset`](https://huggingface.co/datasets/sotetsuk/dds_dataset)
  double-dummy deals (Apache-2.0), generated with [PGX](https://github.com/sotetsuk/pgx).
