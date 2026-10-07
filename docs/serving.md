# Serving Pidgin

The `server/` directory contains the [playable table](https://bridge.localgeek.jp/), debug table, benchmark, and bot APIs. See the [server guide](../server/README.md) for routes and request examples.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install -r server/requirements.txt
scripts/get_models.sh
scripts/serve.sh
```

Visit <http://127.0.0.1:8787/>. `scripts/get_models.sh` downloads the released weights to `server/models/`. The server loads model classes from `training/` and weights from `server/models/`.

To serve the same app in Docker, after downloading the weights:

```bash
docker build -f server/Dockerfile -t pidgin-brill .
docker run --rm -p 8080:8080 pidgin-brill
curl http://localhost:8080/apis/brill/
```

The final request lists the teams available to the Brill API. The Docker image selects `PidginV2` by default; pass `model=PidginV1`, `model=PidginV2`, or `model=BRL` on a Brill request to choose a team explicitly.

## How a team answers a request

A Brill team combines a bidder, a card player, and search settings. The server
loads their weights from the team manifest and reconstructs the position from
each request; the client sends the current position again on its next turn.

```mermaid
flowchart TD
    request["Brill request: position and public team ID"] --> team["teams.json: select models and search settings"]
    team --> phase{"Bidding or card play?"}
    phase -->|Bidding| bid["Bidding model"]
    bid --> bidsearch{"Team enables bidding search?"}
    bidsearch -->|Yes: Pidgin V2| belief["Compare calls using belief-sampled deals"]
    bidsearch -->|No: Pidgin V1 or BRL| call["Return a legal call"]
    belief --> call
    phase -->|Card play| play["Card-play model with sampled-deal search"]
    play --> card["Return a legal card"]
```

This describes requests with an explicit team ID, such as `model=PidginV2`.
The command-line bidding scripts use the bidding network directly; they skip
the bidding-search step. For a copyable request, see the
[Brill API example](../server/README.md#use-a-bot-through-the-brill-api).

## Which files select a model?

| File | Purpose |
|---|---|
| `server/models/teams.json` | Public Brill team IDs, weights, and search settings. |
| `server/models/models.json` | Bidding models offered in the debug table. |
| `server/models/play_models.json` | Card-play models offered in the debug table. |

| Team | Bidding | Card play |
|---|---|---|
| Pidgin V1 (`PidginV1`) | Pidgin V1 bidding model, stored as `D_cw_s75k.pt` | Pidgin V1 card-play model, stored as `play_E48_wideleagueH.pt`, with search |
| Pidgin V2 (`PidginV2`) | Pidgin V2 bidding model, stored as `pidginv2_bid_s40000.pt`, with bidding search | Pidgin Q-net card-play model, stored as `play_B2g_s540k.pt`, with search |
| BRL (`BRL`) | External BRL bidding baseline | The same Pidgin Q-net card-play model as Pidgin V2 |

The filenames preserve checkpoint names because the manifests and loaders depend on them. For how the models were trained, see [Pidgin V1](pidgin_v1.md), [Pidgin V2](pidgin_v2.md), [card play](card_play.md), and [bidding search](belief.md).

## Deploy to your own host

`server/deploy.sh` copies `server/` and `training/` to a configured Linux host and restarts its systemd service. The host's nginx and systemd setup is outside this repository. Set `DEPLOY_HOST=user@host` or put that assignment in the ignored `server/.deploy.env`, then run:

```bash
server/deploy.sh
```

The script checks that weights named in the manifests and `belief_r2.pt` are present, and refuses uncommitted changes under `server/` or `training/` by default. A change to `training/` can change the bots' answers even if the weights are identical; compare the calls and cards on a fixed set of deals when changing model code.
