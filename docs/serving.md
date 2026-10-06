# Serving

`server/` is the site at https://bridge.localgeek.jp and the machine APIs. Full details:
[server/README.md](../server/README.md).

## Run locally

```bash
cd server && python -m emergent.bidserver      # http://127.0.0.1:8787
```

The net classes come from `training/`; `server/training` is a link to it.

## Models

Weights live in `server/models/` and are not in git. Manifests are tracked:

- `teams.json`: the three teams of the Brill API (`model=PidginV1|BRL|PidginV2`): bid
  file, play file, bidding search on/off, sampler.
- `models.json`: `/debug`'s bidding menu; the first entry is the default.
- `play_models.json`: card-play menu; reordering it is how a deploy is rolled back.

| Team | Bidding | Card play |
|---|---|---|
| PidginV1 | `D_cw_s75k.pt` ([pidgin_v1.md](pidgin_v1.md)) | `play_E48_wideleagueH.pt` + PIMC |
| PidginV2 | `pidginv2_bid_s40000.pt` ([pidgin_v2.md](pidgin_v2.md)) + belief search ([belief.md](belief.md)) | `play_B2g_s540k.pt` + PIMC |
| BRL | `brl_fsp_weights.npz` (Kita et al. 2024) | `play_B2g_s540k.pt` + PIMC |

`server/sync_models.sh` copies new snapshots from the lab (`$SRC`, default
`~/code/bridge/lab`) and this repo's `runs/`, without the critic. It copies weights only.

## Deploy

```bash
cd server && ./sync_models.sh && ./deploy.sh
```

`deploy.sh` checks every manifest entry exists, refuses uncommitted changes in `server/`
or `training/` (override: `DEPLOY_DIRTY=1`), rsyncs to the droplet and restarts the
service. nginx and systemd config live in `~/code/infra`. The droplet has one core.

## Docker

```bash
docker build -f server/Dockerfile -t pidgin-brill .     # from the repo root
docker run --rm -p 8080:8080 pidgin-brill
curl 'localhost:8080/apis/brill/'
```

## Changing model code

Any change under `training/` can change what the served bots do. Before committing,
bid a fixed set of seeded auctions with every served model before and after, and
compare. The calls must be identical unless the change is meant to alter play.
