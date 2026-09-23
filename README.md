# BridgeZero — contract bridge bidding from self-play

A bridge bidding net trained from random weights. No expert auctions, no hand-authored
conventions, no bidding labels. The double-dummy table scores the auction and never enters
a gradient.

Training is one chain, three stages, run back to back:

1. **Grounding** — regress the double-dummy score of every legal endpoint, so the net
   learns what each contract is worth and nothing about competing.
2. **Cooperative** — policy gradient with silent opponents. The partnership learns to bid
   its own cards.
3. **Adversarial** — four-seat self-play where the return is the real duplicate table
   result, so a call is worth what it wins at the table, not what it makes.

Stage 3 is a single flag. `--cooperative` stops after stage 2's objective and keeps the
opponents silent; leaving it off is what the released model was trained with.

## Install

```bash
pip install -e ".[dev]"          # numpy, torch, pytest
pip install -e ".[dds]"          # + endplay, only to generate deals yourself
python -m pytest -q              # 115 tests, ~30 s, no dataset needed
```

## Getting deals

Both stages read a memory-mapped array of dealt hands and their double-dummy trick counts.

**Use the published one.** The numbers below were measured on
[`sotetsuk/dds_dataset`](https://huggingface.co/datasets/sotetsuk/dds_dataset) (Apache-2.0),
file `dds_results_100M.npy` — 100M deals, 3.2 GB, shape `(2, 100000000, 4)` int32.

```bash
huggingface-cli download sotetsuk/dds_dataset dds_results_100M.npy \
  --repo-type dataset --local-dir data/
```

**Or make your own**, which needs no download and no account:

```bash
python -m bridgezero.bridge.deals --out data/deals.npz --deals 1000000 --seed 1
```

That solves each deal with `endplay` and writes the same `owners`/`tricks` arrays. It is
fine up to about a million deals; it is not how you would rebuild 100M.

`data/smoke_128.npz` (128 deals) ships with the repo and is what the tests run on.

## Training the released model

Stage 1 — grounding, then cooperative, from random weights (~25 min for grounding):

```bash
python -u -m bridgezero.cooperative.train \
  --data data/dds_results_100M.npy --out runs/stage1 \
  --train-start 2028000 --train-count 1000000 \
  --val-start 3028000 --val-count 5000 --eval-start -10000 --eval-count 10000 \
  --ground-steps 6000 --pg-steps 6000 --batch 1024 --episodes 512 \
  --ground-lr 0.001 --pg-lr 0.0003 --critic-lr 0.001 \
  --width 768 --suit-width 64 --depth 3 --eval-every 500 --seed 1 --device cpu
```

Stage 2 — adversarial self-play from stage 1's `best.pt`:

```bash
python -u -m bridgezero.fourseat.train \
  --data data/dds_results_100M.npy --out runs/stage2 --init runs/stage1/best.pt \
  --steps 60000 --episodes 512 --pg-lr 1e-4 --lr-schedule constant \
  --table-weight 1.0 --silent-frac 0.25 \
  --redouble --sacrifice --double-value --double-gate --fast-rollout \
  --train-block-every 2000 --train-block-size 1000000 \
  --train-pool-start 3033000 --train-pool-end 99990000 \
  --val-start 3028000 --val-count 5000 --eval-start -10000 --eval-count 10000 \
  --snapshot-every 1000 --eval-every 500 \
  --max-level5-rise 0.15 --max-own-drop 60 --max-double-rate 0.6 \
  --seed 1 --device cpu
```

Add `--cooperative` to that second command to train the same architecture without the
adversarial objective. It forces `--silent-frac 1.0` and `--table-weight 0`, and refuses
`--pool`/`--league-frac`.

## Measuring

Strength is paired duplicate IMPs per board against a frozen opponent, on the same boards,
with a confidence interval. Nothing else here has ever been trustworthy — in-training
scores mislead, and the maximum over many snapshots is inflated by the noise it was picked
from.

```bash
python -u tools/match.py --a four:runs/stage2/ckpt_step60000.pt --b four:OTHER.pt \
  --data data/dds_results_100M.npy --deals 10000 --out results/a_vs_b
```

Boards are the last `--deals` deals x 4 dealers x 4 vulnerabilities. Each board is played
at two tables with the sides swapped. Two checks run every time: a player against itself
must score exactly zero, and a sample of auctions is replayed through the scoring code.

## What the stages are worth

160,000 paired boards each, same frozen opponent, 95% confidence intervals:

| run | IMPs/board |
|---|---|
| grounding → adversarial, skipping the cooperative stage | +0.199 [+0.138, +0.262] |
| grounding → cooperative → adversarial (released) | about +0.60 |

The cooperative stage earns its place: without it the same adversarial training reaches
about a third of the strength.

Against the previous best net from a different training line, head to head on 160,000
paired boards, the released model wins by **+0.063 IMP/board [+0.009, +0.116]**. Real, and
small. Adding a league of past snapshots on top of it did not help (−0.056 [−0.101,
−0.013] against it).

Weights are published separately; they are not in this repository.

## What is not here

Card play, the web server, and every match player that needs a third-party engine
(the brl net, the EPBot rule bot, the BridgeBase robot). `tools/match.py` plays
`four:`, `zero:` and `pass:` only.

## Licence

Apache-2.0. See `LICENSE`.
