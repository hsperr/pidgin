# Pidgin V1 (bidding)

The bidding net of the `PidginV1` team, trained from random weights with no expert
auctions. Released file: `D_cw_s75k.pt`. The team plays the cards with the self-play
policy net ([card_play.md](card_play.md)).

Data: `dds_results_100M.npy` (see the README). Nothing else.

## Train it

```bash
./train.sh runs/v1
```

| Stage | Output | What it trains |
|---|---|---|
| 1 ground | `1_ground/best.pt` | double-dummy tricks and contract values on silent-opponent auctions |
| 2 own | `2_own/last.pt` | four-seat self-play; each side is rewarded for its own contract |
| 3 simple | `3_simple/last.pt` | stage 2 plus a 0.2 cost per code word, 30,000 steps |
| 4 table | `4_D/best.pt` | real table score with the code-word cost, against a league of past selves |

Each stage starts from the one before. Stage 4 picks its best checkpoint by paired IMPs
against stage 3. Training runs on CPU.

## Settings

Shared by every stage:

```
--data data/dds_results_100M.npy --val-start 3028000 --val-count 5000
--eval-start -10000 --eval-count 10000 --train-pool-start 3033000
--train-pool-end 99990000 --train-block-size 1000000
```

1. Ground (`python -m training.ground`): width 768, lr 1e-3, batch 1024, patience 5,000.
2. Own contract (`python -m training.fourseat.train`):
   `--episodes 512 --silent-frac 0.25 --select own --table-weight 0 --pg-lr 1e-4
   --redouble --sacrifice --double-value --double-gate --fast-rollout
   --max-level5-rise 0.15 --max-own-drop 60 --max-double-rate 0.6`
3. Code-word cost: stage 2 plus `--code-word-penalty 0.2 --steps 30000`.
4. Table score:
   `--select imp --table-weight 1.0 --code-word-penalty 0.2
   --pg-lr 1e-5 --critic-lr 1e-3 --double-tau 0.1 --xx-tau 0.1 --sac-tau 0.5
   --double-value-lr 1e-5 --xx-value-lr 1e-5 --sac-value-lr 1e-5
   --double-cf-weight 0 --xx-cf-weight 0 --sac-cf-weight 0 --gate-pg
   --any-seat-double --league-frac 0.5 --league-every 1000
   --eval-every 3000 --patience 12000 --max-level5-rise 1 --max-sac-rate 0.2 --max-double-rate 0.4`

The released file is stage 4 at step 75,000.
