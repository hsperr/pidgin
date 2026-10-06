# How to …

Every command runs from the repo root. First, once:

```bash
python -m pip install -e . && python -m pip install -r server/requirements.txt
scripts/get_models.sh                       # released weights -> server/models/
```

Commands that measure on real deals need the DDS dataset at
`data/dds_results_100M.npy` (see the main README, "Training data").
`M=server/models` below.

## Use a model

| To … | Run |
|---|---|
| get one call for a hand | `python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P" --model PidginV1` |
| bid many random deals | `python scripts/generate_auctions.py --n 100000 --out auctions.npz` |
| read the auctions as text | add `--pbn auctions.txt` (one line: deal, dealer, vulnerability, calls) |
| bid deals that have double-dummy tables | add `--data data/dds_results_100M.npy --start 0` |
| sample calls instead of the best call | add `--temperature 1,3,5` (one temperature drawn per deal) |
| play against the bots in a browser | `scripts/serve.sh`, then open http://localhost:8787 |

`--model` takes a team (`PidginV1`, `PidginV2`, `BRL`) or a checkpoint file. Teams bid
exactly as on the site; `bid.py` and `generate_auctions.py` skip V2's bidding search.

## Look inside a bidder

| To … | Run |
|---|---|
| see its openings by HCP and shape | `python tools/openings.py $M/pidginv2_bid_s40000.pt` |
| openings in one seat or vulnerable | add `--seat 3 --vul` |
| count its code words (how simple it is) | `python tools/simplicity.py $M/D_cw_s75k.pt --boards 4000` |

A code word is a call partner cannot read at face value: a suit bid without length in
the suit, a low double, a redouble, or an artificial 2♣. Fewer means easier to follow.

## Compare two bidders

```bash
python tools/match.py --a four:$M/pidginv2_bid_s40000.pt --b four:$M/D_cw_s75k.pt --out results/v2_vs_v1
python tools/weakspots.py results/v2_vs_v1
```

`match.py` plays every deal twice with the seats swapped and scores the difference in
IMPs per board, with double-dummy tricks. Other players: `rule:sayc`, `rule:weakclub`,
`rule:happy` (rule bidders), `pass`, `punish:PATH` (doubles exactly the contracts that go
down). `weakspots.py` breaks the result down: doubled contracts, competitive auctions,
who declared.

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
