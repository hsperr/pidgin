# Use Pidgin from the command line

Every command runs from the repo root. First, once:

```bash
python -m pip install -e . && python -m pip install -r server/requirements.txt
scripts/get_models.sh                       # released weights -> server/models/
```

Commands that measure on real deals need the DDS dataset at
`data/dds_results_100M.npy` (see the main README, "Training data").
The examples use released weights in `server/models/`.

## Use a model

| To … | Run |
|---|---|
| get one call for a hand | `python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P" --model PidginV1` |
| bid many random deals | `python scripts/generate_auctions.py --n 100000 --out auctions.npz` |
| read the auctions as text | add `--pbn auctions.txt` (one line: deal, dealer, vulnerability, calls) |
| bid deals that have double-dummy tables | add `--data data/dds_results_100M.npy --start 0` |
| sample calls instead of the best call | add `--temperature 1,3,5` (one temperature drawn per deal) |
| play against the bots in a browser | `scripts/serve.sh`, then open http://localhost:8787 |

The hand is spades.hearts.diamonds.clubs (`T` means ten), and `--auction` lists
calls already made (`P` means pass). With the example auction, the next seat
holds the supplied hand. `--model` takes a team (`PidginV1`, `PidginV2`, `BRL`)
or a checkpoint file. These scripts use the bidding network directly; the
Pidgin V2 team through the Brill API adds bidding search, so its calls can differ.

## Look inside a bidder

| To … | Run |
|---|---|
| see its openings by high-card points and suit lengths | `python tools/openings.py server/models/pidginv2_bid_s40000.pt` |
| openings in one seat or vulnerable | add `--seat 3 --vul` |
| count Pidgin V1's code words | `python tools/simplicity.py server/models/D_cw_s75k.pt --boards 4000` |
| compare openings of several bidders side by side | `python scripts/opening_tables.py "Pidgin V1=server/models/D_cw_s75k.pt" "BRL=brl:server/models/brl_fsp_weights.npz"` ([docs/simplicity.md](../docs/simplicity.md)) |

`D_cw_s75k.pt` is the released Pidgin V1 bidding checkpoint;
`pidginv2_bid_s40000.pt` is Pidgin V2's. A code word is a call flagged by the
repo's simplicity heuristic, such as a suit bid without the usual suit length.

## Compare two bidders

```bash
python tools/match.py --a four:server/models/pidginv2_bid_s40000.pt \
  --b four:server/models/D_cw_s75k.pt --out results/v2_vs_v1
python tools/weakspots.py results/v2_vs_v1
```

`match.py` plays each board at two tables with the partnerships swapped. It
uses double-dummy trick tables to score the results in International Match
Points (IMPs). Other players include `rule:sayc`, `rule:weakclub`, and
`rule:happy` (simple rule bidders), `pass`, and `punish:PATH` (a diagnostic
opponent that uses the hidden trick table to double failing contracts).
`weakspots.py` breaks the saved match down by doubled contracts, competitive
auctions, and declarer.

## Reproduce the results page

With the DDS dataset installed, `scripts/make_results.sh` runs the released
models through bidding matches, opening and simplicity checks, and card-play
matches reported in [docs/results.md](../docs/results.md). It writes reports
under `results/`; runtime depends on the machine.

## Watch training

`python tools/dashboard.py --runs runs/` opens a local dashboard (http://localhost:8770)
with every run's metrics, its openings, and matches against the rule bidders.

## Train

| Model | Doc |
|---|---|
| Pidgin V1 bidding: `./train.sh runs/v1` | [docs/pidgin_v1.md](../docs/pidgin_v1.md) |
| Pidgin V2 bidding | [docs/pidgin_v2.md](../docs/pidgin_v2.md) |
| belief net | [docs/belief.md](../docs/belief.md) |
| card play (policy net and Q-net) | [docs/card_play.md](../docs/card_play.md) |

## Maintainer

`python scripts/upload_models.py --push` publishes the team models to Hugging Face
(dry run without `--push`).
