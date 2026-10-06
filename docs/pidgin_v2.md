# Pidgin V2 (bidding)

The bidding net behind the `PidginV2` team. Served file:
`server/models/pidginv2_bid_s40000.pt` (lab name g_s2o_hi3_lo, step 40,000). The team
bids with the belief-net search ([belief.md](belief.md)) and plays with B2g
([card_play.md](card_play.md)).

Data: `dds_results_100M.npy`. Training also needs Pidgin V1 (`D_cw_s75k.pt`) as the
fixed opponent and selection yardstick, written `$D` below.

Common flags for every stage:

```
python -m training.fourseat.train --data data/dds_results_100M.npy --episodes 2048
  --val-start 3028000 --val-count 5000 --eval-start -10000 --eval-count 10000
  --train-pool-start 3033000 --train-pool-end 99990000 --train-block-size 1000000
  --train-block-every 2000 --silent-frac 0.25 --snapshot-every 1000 --patience 0
  --select imp --imp-opponent $D --table-weight 1 --code-word-penalty 0.2
```

## Stages

1. Ground: V1's stage 1 (`1_ground/best.pt`), see [pidgin_v1.md](pidgin_v1.md).
2. Table reward with perfect doublers, from `1_ground/best.pt` (GPU, `--seed 101`):
   `--table-down-doubled own --pg-lr 3e-4 --steps 40000 --eval-every 2000
   --max-level5-rise 0.15 --max-double-rate 0.6 --device cuda`
3. Real table score, from stage 2 `best.pt` (`--seed 102`):
   `--pg-lr 3e-5 --critic-lr 1e-3 --double-tau 0.1 --xx-tau 0.1 --sac-tau 0.5
   --double-value-lr 3e-5 --xx-value-lr 3e-5 --sac-value-lr 3e-5
   --double-cf-weight 0 --xx-cf-weight 0 --sac-cf-weight 0 --gate-pg
   --any-seat-double --league-frac 0.2 --league-every 1000
   --punisher-frac 0.4 --punisher-miss 0 --punisher-refresh 1000
   --fixed-frac 0.2 --fixed-opponents $D
   --steps 60000 --eval-every 2000 --max-level5-rise 1 --max-sac-rate 0.2 --max-double-rate 0.4`
   The first ~33k steps ran on a GPU box, the rest on CPU with `--resume`.
4. Light-opening penalty, from stage 3 `ckpt_step32000.pt` (`--seed 103`): the stage 3
   flags plus `--light-open-penalty 0.5 --steps 30000`, then continued with
   `--resume --steps 90000`. The served file is `ckpt_step40000.pt`.

Recorded scripts (local only, not in git): `runs/exp_20260930/box/g_chain.sh` (stages
2–3), `runs/exp_20260930/g_s2o_hi3_local.sh`, `g_s2o_hi3_lo_local.sh`,
`g_s2o_hi3_lo2_local.sh`, and each run's `run.json`.

## Not reproducible yet

- Stage 2's LR (3e-4) and `--table-down-doubled own` are in its `run.json`, not in the
  chain script, which took them as arguments.
- Stage 3 was split between a GPU box and the Mac; the GPU part's seed stream is not
  the same as one uninterrupted run.
- The served file still carries its critic (17 MB). `sync_models.sh` strips critics; this
  file was copied by hand.
