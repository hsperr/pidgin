# Pidgin repo

One repo to train and serve the bridge bots. Read `README.md` "What is here" for the map.

- `training/`: model code, shared by trainers and `server/` (`server/training` links to it).
- `server/`: the live site. `belief/`, `play/`: trainers moved from the lab.
- `docs/`: one file per served model with its training commands and known gaps.
- `runs/`, `data/`, `server/models/*.pt` are local only (gitignored).

## Commands

- Python: `/Users/hsperr/miniconda3/bin/python`.
- Tests: `python -m pytest -q` in the repo root and again in `server/`.
- Smoke train: `./train.sh --smoke /tmp/smoke`.

## Rules

- Deploy only a committed tree: `cd server && ./deploy.sh`. `./sync_models.sh` copies
  weights only, never code.
- A change under `training/` can change served play. Bid a fixed set of seeded auctions
  with every served model before and after; the calls must match unless the change
  means to alter them.
- New trainers take data paths from flags or `BRIDGE_DATA`, never `/Users/...`.
- Experiments go in `~/code/bridge/lab`. Move code here when it trains a shipped model,
  and add its doc in `docs/`.
