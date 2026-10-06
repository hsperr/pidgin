# Card play

Two kinds of card-play net are released. On the site both play with PIMC search: sample
deals that fit what the seat knows, solve them double dummy, and pick the best card on
average (`training/play/search.py`, settings in `server/emergent/engine.py`).

| Released file | Used by | Code |
|---|---|---|
| `play_E48_wideleagueH.pt` | `PidginV1`; deal sampler for the Q-net | `training/play/train.py` |
| `play_E48_leagueE.pt` | `/debug` | same |
| `play_B2g_s540k.pt` | `PidginV2`, `BRL` | `qnet/train_play.py` |

## Policy net: self-play only

The net sits in all four chairs and plays the deal out. The only reward is the tricks
each side takes. A critic gives the baseline, and a belief head learns where the hidden
cards are. The double-dummy table only reports a score; it never reaches a gradient.

Auctions to play: a Pidgin bidder samples its calls at temperatures 1, 3 and 5, so the
net also meets odd contracts.

```bash
python scripts/generate_auctions.py --model PidginV1 --n 1000000 --temperature 1,3,5 \
  --data data/dds_results_100M.npy --start 0 --out data/play/auctions_1M.npz
```

Training, four runs:

```bash
python -m training.play.train data/play/auctions_1M.npz --out runs/play/D --width 512 --batch 512 --steps 25000
python -m training.play.train data/play/auctions_1M.npz --out runs/play/E --init runs/play/D/last.pt --pool-size 5 --steps 20000
python -m training.play.train data/play/auctions_1M.npz --out runs/play/F --width 1024 --batch 512
python -m training.play.train data/play/auctions_1M.npz --out runs/play/H --init runs/play/F/last.pt --pool-size 5 --belief-final 0.1
```

`E` is `play_E48_leagueE.pt` and `H` is `play_E48_wideleagueH.pt`. Both play against a
pool of their own past snapshots. `--help` lists every option.

Match two nets on fixed boards:
`python -m training.play.match data/play/bench_100k.npz --challenger X --reference random`.

## Q-net: values from the solver

For every legal card at every position, the net learns the double-dummy tricks relative
to the best card. A belief head (`--bhead`) feeds PIMC sampling. Code in `qnet/`.

```bash
qnet/make_auc.sh        # Pidgin V1 auctions, 600 shards x 2,000 deals -> qnet/auc/
qnet/worker.sh          # play them out and solve every legal card -> qnet/pos/ (pip install endplay)
python qnet/train_play.py --run B
python qnet/train_play.py --run B2 --bhead --init runs/play_q/B/last.pt
python qnet/train_play.py --run B2g --bhead --init runs/play_q/B2/last.pt --device cuda
```

`worker.sh` plays each auction out with the policy net, with 20% random cards in the
training shards, and solves every legal card with endplay DDS. Run several workers at
once to go faster. The released file is `B2g` at step 540,000.
