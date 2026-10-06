# Belief net (bidding search)

Reads one seat's hand, the vulnerability and the auction so far, and guesses who holds
each hidden card, plus each hidden hand's HCP and suit lengths. The `PidginV2` team uses
it to sample deals that fit the auction. It scores its top candidate calls on those
deals and changes its call only when another one is clearly better
(`server/emergent/bidsearch.py`).

Released file: `belief_r2.pt` (float16, without the training-only system head).

Code in `belief/`: `train.py` (trainer), `gen.py` (training data), `analyze.py` (how the
guess sharpens over an auction), `train_shape.py` (a variant that predicts shapes first).

## Data

The net must read many bidding styles, not one. `belief/gen.py` deals hands from
`dds_results_100M.npy`, lets a pool of bidders bid them against each other, and writes
shards to `belief/data/{train,test}/`. `belief/data/systems.json` lists the pool: Pidgin
nets, BRL, EPBot systems (through libEPBot) and recorded WBridge5 auctions
(`$BRIDGE_WB5`; `--no-wb5` skips them).

```bash
OMP_NUM_THREADS=4 python -u belief/gen.py test
OMP_NUM_THREADS=4 python -u belief/gen.py train     # keeps writing shards until stopped
```

## Train it

Two runs: card guesses first, then the HCP and suit-length heads on top.

```bash
python -u belief/train.py --out runs/belief/r0 --steps 100000 --lr 3e-4 --d 1024 --layers 4 \
  --holdout ep_wj,E28 --wb5-frac 0.2 --reload 2000
python -u belief/train.py --out runs/belief/r2 --init runs/belief/r0/last.pt --steps 30000 \
  --lr 1e-4 --d 1024 --layers 4 --summary --sum-weight 0.3 \
  --holdout ep_wj,E28 --wb5-frac 0.2 --reload 1000000
```

`--holdout` names bidders from `systems.json` that stay out of training and are used
only for testing. Add `--device mps` or `--device cuda` for a GPU.
