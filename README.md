# Pidgin — bridge bidding from self-play

Train a bridge bidder from random weights through the D recipe: grounding,
own-contract self-play, a simplicity fine-tune, and table-score self-play.
The bidder sees its own hand and public calls. Double-dummy trick tables
supply training targets and rewards; they are never bidder inputs.
No expert auctions or bidding labels are required.

## What is here

| Folder | What |
|---|---|
| `training/` | The model code: bidding nets, trainers, card play. Shared by training and the server. |
| `server/` | The site and APIs that serve the models ([docs/serving.md](docs/serving.md)). |
| `belief/` | Belief net for the bidding search ([docs/belief.md](docs/belief.md)). |
| `play/` | Card-play Q-net (B2g) trainer and data generators ([docs/card_play.md](docs/card_play.md)). |
| `docs/` | How each served model was trained: [Pidgin V1](docs/pidgin_v1.md), [Pidgin V2](docs/pidgin_v2.md), [belief](docs/belief.md), [card play](docs/card_play.md). |
| `scripts/` | Use the released models: download, bid, generate auctions, serve. |
| `tools/` | Matches, dashboard, analysis. |

## Use the released models

```bash
python -m pip install -e . && python -m pip install -r server/requirements.txt
scripts/get_models.sh                                    # weights from Hugging Face, ~130 MB
python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P"  # one call, with the top four
python scripts/generate_auctions.py --n 100000 --out auctions.npz --pbn auctions.txt
scripts/serve.sh                                         # the site and APIs on localhost:8787
```

`--model` picks a team (`PidginV1`, `PidginV2`, `BRL`) or a checkpoint file.
`generate_auctions.py` bids about 4,000 deals a second on a laptop CPU; `--help` lists
the output format.

## Quick start

Use Python 3.10 or newer. Run commands from this checkout:

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

## Random → D

| Output directory | Training | Checkpoint selection |
|---|---|---|
| `1_ground` | Regress DD tricks and contract values on silent-opponent prefixes; learn a policy from predicted values | Silent-opponent contract score |
| `2_own` | Four-seat self-play, rewarded for each partnership's own contract | Own-contract score |
| `3_simple` | Continue own-contract training for up to 30,000 steps with a 0.2 code-word cost | Own-contract score |
| `4_D` | Real table rewards, 0.2 code-word cost, and a league of past snapshots | Paired IMPs against `3_simple/last.pt` |

Stage 2 starts from `1_ground/best.pt`; stages 3 and 4 start from the preceding
stage's `last.pt`. The final selected model is **`runs/training/4_D/best.pt`**.
D enables doubles and redoubles at any legal seat. Its starting policy is
measured and kept as a fallback when updates fail to improve or trip a guard.
Changing the legal doubling policy at this transition can change play even
before the first update, so its starting IMP score need not be zero.

This is a public adaptation of the historical D experiment. It preserves the
own-contract simplicity fine-tune and D's reward settings, but selects against
its public parent. The historical experiment used a separate private reference
for selection. This recipe does not promise the same chosen weights or strength.
No pretrained weights or sibling repositories are needed.

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
a guard trips. D validates every 3,000 steps and stops after 12,000 without
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
  --b four:runs/training/3_simple/last.pt --out results/d_vs_parent
python tools/simplicity.py runs/training/4_D/best.pt --boards 4000
python tools/weakspots.py results/d_vs_parent
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

A **code word** is a call flagged by any of these rules:

- A suit bid with fewer than 4 cards, or fewer than 3 when raising partner.
- A double of a contract at level 3 or below.
- A redouble.
- A 2♣ opening with fewer than 5 clubs or at least 20 HCP.

Each flagged call counts once, even when multiple rules apply. The trainer logs
`code_words_per_100` on greedy validation self-play and `code_word_share` on
sampled training calls. The 0.2 cost is in score units of 100 points and is
charged to that call's policy advantage; it does not alter the reported IMPs.
Set `CODE_WORD_PENALTY` to change it in stages 3 and 4.

`tools/simplicity.py` measures self-play or saved match auctions and also
reports natural suit bids, cue bids, jumps, 4NT, and auction length:

```bash
python tools/simplicity.py --boards-npz results/d_vs_parent/boards.npz
```

For a match between different players, the Python `analyse(..., who="A")`
API counts A's calls across both tables; the default counts all calls at table 1.
This heuristic measures face-value readability, not human learnability.
Compare both IMPs and simplicity when assessing a model.

The sacrifice value head ranks candidate contracts using a DD counterfactual
where the opponents double and everyone passes. Actual opponents can respond
differently. D's policy reward uses the final auction, but candidate ranking
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
