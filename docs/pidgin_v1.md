# Pidgin V1 (bidding)

The bidding net behind the `PidginV1` team. Served file: `server/models/D_cw_s75k.pt`
(lab name D75: stage 3 at step 75,000). Card play for this team is E48, see
[card_play.md](card_play.md).

Data: `dds_results_100M.npy` (see the README). No other model or label is needed.

## Train it

```bash
./train.sh runs/v1
```

`train.sh` is the public V1 recipe. It produces a Pidgin-V1-like model, not the same
weights (see the gaps below). Its four stages:

| Stage | Output | What it trains |
|---|---|---|
| 1 ground | `1_ground/best.pt` | DD tricks and contract values from silent-opponent auctions |
| 2 own | `2_own/last.pt` | four-seat self-play, reward = each side's own contract |
| 3 simple | `3_simple/last.pt` | stage 2 plus a 0.2 code-word cost, 30,000 steps |
| 4 D | `4_D/best.pt` | real table score, 0.2 code-word cost, league of past selves |

## The recorded run (2026-09-25)

The served file came from these commands. Common flags:

```
--data data/dds_results_100M.npy --val-start 3028000 --val-count 5000
--eval-start -10000 --eval-count 10000 --train-pool-start 3033000
--train-pool-end 99990000 --train-block-size 1000000
```

1. Ground: `python -m training.ground --seed 1 --threads 8 --patience 5000 --eval-every 1000 --out 1_ground`
   (recorded: width 768, lr 1e-3, batch 1024, max 96,000 steps).
2. Own contract, from `1_ground/best.pt`, then a full pass with `--resume --patience 0`:
   `--episodes 512 --silent-frac 0.25 --select own --table-weight 0 --pg-lr 1e-4
   --redouble --sacrifice --double-value --double-gate --fast-rollout
   --block-order permuted --lr-schedule constant --max-level5-rise 0.15 --max-own-drop 60 --max-double-rate 0.6`
3. Code-word cost, from `2_own/last.pt`: same as 2 plus `--code-word-penalty 0.2 --steps 30000 --seed 2`.
4. Table score, from `2_own_cw0.2/last.pt` (`--seed 3`):
   `--select imp --imp-opponent <E46> --table-weight 1.0 --code-word-penalty 0.2
   --pg-lr 1e-5 --critic-lr 1e-3 --double-tau 0.1 --xx-tau 0.1 --sac-tau 0.5
   --double-value-lr 1e-5 --xx-value-lr 1e-5 --sac-value-lr 1e-5
   --double-cf-weight 0 --xx-cf-weight 0 --sac-cf-weight 0 --gate-pg
   --any-seat-double --league-frac 0.5 --league-every 1000
   --eval-every 3000 --patience 12000 --max-level5-rise 1 --max-sac-rate 0.2 --max-double-rate 0.4`.
   The served file is `ckpt_step75000.pt` of this stage.

All four used `python -m training.fourseat.train` except stage 1. Run records:
`runs/scratch_20260925/*/run.json` (local only, not in git).

## Not reproducible yet

- Stages 2–4 ran from frozen code copies (`code`, `code_v2`–`code_v4`), not today's
  `training/`. Today's trainer differs in 11 files; new flags default off.
- Stage 4 selected checkpoints against E46, a lab model that is not in this repo.
  `train.sh` selects against its own stage 3 instead.
