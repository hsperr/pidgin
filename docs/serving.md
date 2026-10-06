# Serving

`server/` is the site at https://bridge.localgeek.jp and its bot APIs. Full details:
[server/README.md](../server/README.md).

## Run locally

```bash
scripts/get_models.sh      # weights from Hugging Face into server/models/
scripts/serve.sh           # http://localhost:8787
```

The net classes come from `training/`; `server/training` is a link to it.

## Models

Weights live in `server/models/` and are not in Git. The manifests are:

- `teams.json`: the three teams of the Brill API (`model=PidginV1|BRL|PidginV2`): bidding
  file, card-play file, bidding search on or off.
- `models.json`: `/debug`'s bidding menu; the first entry is the default. Entries whose
  file is missing are skipped.
- `play_models.json`: the card-play menu.

| Team | Bidding | Card play |
|---|---|---|
| PidginV1 | `D_cw_s75k.pt` ([pidgin_v1.md](pidgin_v1.md)) | `play_E48_wideleagueH.pt` + PIMC |
| PidginV2 | `pidginv2_bid_s40000.pt` ([pidgin_v2.md](pidgin_v2.md)) + belief search ([belief.md](belief.md)) | `play_B2g_s540k.pt` + PIMC |
| BRL | `brl_fsp_weights.npz` (Kita et al. 2024) | `play_B2g_s540k.pt` + PIMC |

## Docker

```bash
docker build -f server/Dockerfile -t pidgin-brill .     # from the repo root
docker run --rm -p 8080:8080 pidgin-brill
curl 'localhost:8080/apis/brill/'
```

## Deploy your own

`server/deploy.sh` copies `server/` and `training/` to a host over rsync and restarts a
systemd service there. Set `DEPLOY_HOST=user@host` (or write it to the git-ignored
`server/.deploy.env`). It refuses uncommitted changes in `server/` or `training/`.

## Changing model code

Any change under `training/` can change what the served bots do. Bid a fixed set of
seeded auctions with every model before and after the change and compare; the calls
must match unless the change is meant to alter them.
