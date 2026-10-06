# Card play

Two card-play nets are served.

| Served file | Used by | Code |
|---|---|---|
| `play_E48_wideleagueH.pt` | `PidginV1`; also the deal sampler for B2g | `training/play/train.py` |
| `play_E48_leagueE.pt` | `/debug` only | same |
| `play_B2g_s540k.pt` | `PidginV2`, `BRL` (with PIMC) | `play/train_play.py` |

On the site both play with PIMC search (`training/play/search.py`, settings in
`server/emergent/engine.py`).

## E48: self-play REINFORCE

Learns from self play only: the net sits in all four chairs. The reward is tricks won.
There are no solver labels; the double-dummy table only reports a score.

Data: auctions from `play/gen_auctions.py` (sampled from the E46 bidder, every dealer and
vulnerability, temperature 1/3/5 per deal):

```bash
python play/gen_auctions.py <E46 ckpt> data/play/auctions_1M.npz 1000000
```

Chain (from the lab's run logs and `E48_PLAY_PLAN.md`):

1. `bigD`: width 512, batch 512, 25,000 steps, about 978k training deals.
2. `leagueE`: from `bigD/last.pt`, against a pool of 5 past selves, 20,000 steps.
3. `wideF`: width 1024, from scratch.
4. `wideleagueH`: from `wideF/last.pt`, pool of 5 past selves, belief loss decayed to 0.1.

```bash
python -m training.play.train data/play/auctions_1M.npz --out runs/play_bigD --width 512 --batch 512 --steps 25000
python -m training.play.train data/play/auctions_1M.npz --out runs/play_leagueE --init runs/play_bigD/last.pt --league ...
```

The trainer's flags: `--width --depth --batch --steps --init --league --pool-size
--pool-every --belief-weight --belief-final --belief-decay-by --group`.
`--help` lists the rest.

`runs/play_E48_bigD/last.pt` (lab) is the fixed yardstick for card-play matches:
`python -m training.play.match data/play/bench_100k.npz --challenger X --reference random`.

## B2g: Q-net from solver values

Learns, for every legal card at every position, the double-dummy tricks relative to the
best card (MSE). Has a belief head (`--bhead`) for PIMC sampling.

Data, two steps:

1. `play/make_auc.sh`: V1 (`D_cw_s75k.pt`) greedy auctions on 600 × 2,000 deals
   (`play/gen_d.py`, deals from 62,000,000).
2. `play/worker.sh`: plays each auction out with E48 greedy plus 20% random cards (test
   set 0%), and solves every legal card with endplay DDS (`play/gen_play.py`).
   Needs `pip install endplay`.

Train:

```bash
python play/train_play.py --run B                                   # from scratch
python play/train_play.py --run B2 --bhead --init runs/play_q/B/last.pt
python play/train_play.py --run B2g --bhead --init <b2g_resume.pt> --step0 500000 --device cuda
```

Runs log to `runs/play_q/<run>/`.

## Not reproducible yet

- E48: no command line was recorded. The checkpoints hold only `{width, depth}`. The
  chain above is rebuilt from logs and notes, and the flags marked `...` are unknown.
- E48 auctions came from E46, which is not in this repo (`lab/runs/E46_*`).
- B2g: the starting checkpoint `b2g_resume.pt` and the GPU-box run before step 500,000
  are not recorded.
