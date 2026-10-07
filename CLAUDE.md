# Pidgin repo

One repo to train and serve the bridge bots. Read `README.md` "What is here" for the map.

- `training/`: model code, shared by trainers and `server/` (`server/training` links to it).
- `server/`: the live site. `belief/`, `qnet/`: trainers for the belief net and the card-play Q-net.
- `docs/`: one file per served model with its training commands.
- `runs/`, `data/`, `server/models/*.pt` are local only (gitignored).

## Commands

- Python: use Python 3.10 or newer in an activated virtual environment.
  Install development dependencies with `python -m pip install -e ".[dev]"`
  and server dependencies with `python -m pip install -r server/requirements.txt`.
- Tests: `python -m pytest -q` in the repo root and again in `server/`.
- Smoke train: `./train.sh --smoke /tmp/smoke`.

## Rules

- Deploy only a committed tree: `cd server && ./deploy.sh`. `./sync_models.sh` copies
  weights only, never code.
- A change under `training/` can change served play. Bid a fixed set of seeded auctions
  with every served model before and after; the calls must match unless the change
  means to alter them.
- New trainers take data paths from flags or `BRIDGE_DATA`, never `/Users/...`.
- New code here trains or serves a released model; add its doc in `docs/`.

## Documentation

- Write for someone seeing the repository for the first time. Use Pidgin V1 and
  Pidgin V2 in prose; explain any research identifier required in a command or file.
- Keep checkpoint filenames and API IDs compatible. Change display labels rather
  than renaming an artifact without migrating its consumers.
- Give commands from the repository root and define required variables. State how a
  model is trained; leave out caveats about exact reproduction or run history.
