# Belief net (bidding search)

Reads one seat's hand, the vulnerability and the auction so far, and guesses who holds
each hidden card, plus each hidden hand's HCP and suit lengths. The `PidginV2` team and
`/debug`'s search toggle use it to sample deals that fit the auction
(`server/emergent/bidsearch.py`).

Served file: `server/models/belief_r2.pt` (run r2, step 30,000). `sync_models.sh` writes
it from `last.pt` as float16 without the system head.

Code: `belief/` (`train.py` trainer, `gen.py` auction generator, `analyze.py`,
`train_shape.py` for the shape-first variant rC).

## Data

`belief/gen.py` bids fresh deals from `dds_results_100M.npy` with many bidders against
each other and writes shards to `belief/data/{train,test}/`. `belief/data/systems.json`
names the bidders. They include:

- our nets (D75, E46, E28, …) from `$BRIDGE_KEEP` (default `~/code/bridge/archive/notes/keep`);
- brl FSP/SL (Kita et al. 2024);
- EPBot systems through libEPBot (Mac only).

`gen.py` imports the lab's match harness and EPBot wrapper from `$BRIDGE_LAB`
(default `~/code/bridge/lab`). It does not run from this repo alone.

WBridge5 auctions (system 0) are read from `$BRIDGE_WB5`
(default `data/wbridge5/{}.npz`). `--no-wb5` skips them.

```bash
OMP_NUM_THREADS=4 python -u belief/gen.py test
OMP_NUM_THREADS=4 python -u belief/gen.py train     # runs until stopped
```

## Train it

```bash
# r0: from scratch
python -u belief/train.py --out runs/belief/r0 --steps 100000 --lr 3e-4 --d 1024 --layers 4 \
  --holdout ep_wj,E28 --wb5-frac 0.2 --reload 2000 --device mps
# r2: from r0, with the HCP / suit-length heads
python -u belief/train.py --out runs/belief/r2 --init runs/belief/r0/last.pt --steps 30000 \
  --lr 1e-4 --d 1024 --layers 4 --summary --sum-weight 0.3 \
  --holdout ep_wj,E28 --wb5-frac 0.2 --reload 1000000 --device mps
```

Both commands come from each run's `args.json` in the lab
(`lab/experiments/belief_multi/runs/{r0,r2}/args.json`). Every other flag was the default.

## Not reproducible yet

- The training shards depend on how long `gen.py train` ran and on the external bots
  (EPBot, brl, WBridge5). The exact r0/r2 shard set is not recorded.
- `gen.py` needs the lab repo for its bidders.
