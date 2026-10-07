# Card play

The released teams use two kinds of card-play model. A **policy model** picks a
card directly and learns through self-play. A **Q-net** estimates the value of
each legal card from double-dummy solver results. On the site, both use search:
they sample possible hidden deals, solve them, and pick the card with the best
average result. See `training/play/search.py` and `server/emergent/engine.py`.

| Public name | Released filename | Used by |
|---|---|---|
| Pidgin V1 card play | `play_E48_wideleagueH.pt` | Pidgin V1; also samples hidden deals for the Q-net. |
| Compact self-play card-play model | `play_E48_leagueE.pt` | Alternative in the site's debug view. |
| Pidgin Q-net card play | `play_B2g_s540k.pt` | Pidgin V2 and BRL. |

The old codes in these filenames identify published files. Download them with
`scripts/get_models.sh`. To compare the Pidgin V1 policy model with random legal
play on the published benchmark (without search):

```bash
python -m pip install -e .
python -m pip install -r server/requirements.txt
scripts/get_models.sh
python -m training.play.match server/models/bench_100k.npz \
  --challenger server/models/play_E48_wideleagueH.pt \
  --reference random --deals 100
```

This command evaluates direct policy decisions. The site also runs search,
which can change the result.

## Train the policy model

The trainer is `training/play/train.py`. It needs auctions paired with hands and
double-dummy trick tables. Download `data/dds_results_100M.npy` as shown in the
[README](../README.md#training-data), then generate sampled Pidgin V1 auctions:

```bash
python -m pip install -r server/requirements.txt
mkdir -p data/play
python scripts/generate_auctions.py --model PidginV1 --n 1000000 \
  --temperature 1,3,5 --data data/dds_results_100M.npy --start 0 \
  --out data/play/auctions_1M.npz
```

The historical policy training used two model widths. The compact model uses
the first pair; Pidgin V1 card play uses the second. Each second run starts from
the matching first run and plays against a pool of past snapshots:

```bash
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/compact_base --width 512 --batch 512 --steps 25000
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/compact_league --init runs/play/compact_base/last.pt \
  --width 512 --pool-size 5 --steps 20000
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/v1_base --width 1024 --batch 512
python -m training.play.train data/play/auctions_1M.npz \
  --out runs/play/v1_league --init runs/play/v1_base/last.pt \
  --width 1024 --pool-size 5 --belief-final 0.1
```

The latter two commands use the trainer's default `--steps 200`. Increase the
step count for a serious training run. The released files are historical
checkpoints; the commands show the stages and do not promise identical weights.

## Train the Q-net

The Q-net learns solver values for every legal card. Its data pipeline first
generates Pidgin V1 auctions, then plays each deal with the policy model and
solves each legal card. It requires the full DDS dataset, the released Pidgin V1
bidder and card-play model, and the `endplay` package for double-dummy solving.
From the repository root:

```bash
python -m pip install endplay
scripts/get_models.sh
bash qnet/make_auc.sh
bash qnet/worker.sh
python qnet/train_play.py --run values --steps 10000
python qnet/train_play.py --run belief --bhead \
  --init runs/play_q/values/last.pt --steps 10000
python qnet/train_play.py --run refined --bhead \
  --init runs/play_q/belief/last.pt --device cuda --steps 10000
```

`make_auc.sh` writes auction files under `qnet/auc/`; `worker.sh` writes solved
positions under `qnet/pos/`. The worker can be run in several processes and
exits when all current auctions are processed. `--device cuda` requires a CUDA
GPU; choose `--device cpu` when one is unavailable. The 10,000-step values above
are an example, not the released training duration. The published Q-net is a
checkpoint from the final stage at step 540,000.
