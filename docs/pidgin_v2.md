# Pidgin V2 (bidding)

The bidding net of the `PidginV2` team. Released file: `pidginv2_bid_s40000.pt`. The
team bids with the belief-net search on top ([belief.md](belief.md)) and plays the
cards with the Q-net ([card_play.md](card_play.md)).

V2 starts from V1's grounding and trains on the table score from the start. Two things
push it away from V1's style: perfect doublers in stage 2 (a failing contract always
costs the doubled penalty) and a penalty on light openings in stage 4.

Data: `dds_results_100M.npy`, plus Pidgin V1 (`D_cw_s75k.pt`, written `$V1` below) as a
fixed opponent and the yardstick for checkpoint selection.

Every stage runs `python -m training.fourseat.train` with:

```
--data data/dds_results_100M.npy --episodes 2048
--val-start 3028000 --val-count 5000 --eval-start -10000 --eval-count 10000
--train-pool-start 3033000 --train-pool-end 99990000 --train-block-size 1000000
--train-block-every 2000 --silent-frac 0.25 --snapshot-every 1000 --patience 0
--select imp --imp-opponent $V1 --table-weight 1 --code-word-penalty 0.2
```

## Stages

1. Ground: V1's stage 1, `1_ground/best.pt` ([pidgin_v1.md](pidgin_v1.md)).
2. Perfect doublers, from stage 1:
   `--table-down-doubled own --pg-lr 3e-4 --steps 40000 --eval-every 2000
   --max-level5-rise 0.15 --max-double-rate 0.6`
   A GPU helps here (`--device cuda`).
3. Real table score, from stage 2's `best.pt`:
   `--pg-lr 3e-5 --critic-lr 1e-3 --double-tau 0.1 --xx-tau 0.1 --sac-tau 0.5
   --double-value-lr 3e-5 --xx-value-lr 3e-5 --sac-value-lr 3e-5
   --double-cf-weight 0 --xx-cf-weight 0 --sac-cf-weight 0 --gate-pg
   --any-seat-double --league-frac 0.2 --league-every 1000
   --punisher-frac 0.4 --punisher-miss 0 --punisher-refresh 1000
   --fixed-frac 0.2 --fixed-opponents $V1
   --steps 60000 --eval-every 2000 --max-level5-rise 1 --max-sac-rate 0.2 --max-double-rate 0.4`
4. Light-opening penalty, from stage 3 at step 32,000: the stage 3 settings plus
   `--light-open-penalty 0.5`.

The released file is stage 4 at step 40,000.
