"""Maintainer only: publish the team models to the Hugging Face Hub.

    python scripts/upload_models.py                  # dry run: list what would go up
    python scripts/upload_models.py --push           # needs `hf auth login`

Uploads the files the three teams (server/models/teams.json) need, the bidding
search's belief net, the play benchmark, the BRL licence, and a model card.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "server" / "models"
EXTRA = ["belief_r2.pt", "bench_100k.npz", "play_E48_leagueE.pt", "brl_LICENSE"]

CARD = """---
license: apache-2.0
tags: [bridge, contract-bridge, reinforcement-learning, self-play, game-ai]
---
# Pidgin: contract bridge bots from self-play

Weights for the bots in https://github.com/{gh}. Code, training recipes and the server
live there; this repo only holds the weights.

| Team | Bidding | Card play |
|---|---|---|
{rows}

Other files: `belief_r2.pt` (belief net for the bidding search),
`bench_100k.npz` (frozen card-play benchmark), `play_E48_leagueE.pt`.

`brl_fsp_weights.npz` is not ours: it is the FSP bidder of Kita et al. (2024),
redistributed under its Apache-2.0 licence (`brl_LICENSE`).

## Use

```bash
git clone https://github.com/{gh} && cd {name}
scripts/get_models.sh
python scripts/bid.py AKQ2.JT9.876.543 --auction "1H P"
scripts/serve.sh
```
"""


def files() -> list[str]:
    out = []
    for t in json.load(open(MODELS / "teams.json")):
        out += [t["bid"], t["play"]] + ([t["sampler"]] if "sampler" in t else [])
    return sorted(set(out + EXTRA))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("PIDGIN_HF_REPO", "hsperr/pidgin"))
    ap.add_argument("--github", default="hsperr/pidgin", help="code repo named in the model card")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    names = files()
    missing = [f for f in names if not (MODELS / f).exists()]
    if missing:
        raise SystemExit(f"missing in server/models: {missing}")
    total = sum((MODELS / f).stat().st_size for f in names)
    for f in names:
        print(f"  {(MODELS / f).stat().st_size / 1e6:7.1f} MB  {f}")
    print(f"  {total / 1e6:7.1f} MB  total -> {args.repo}")
    if not args.push:
        print("dry run; add --push to upload")
        return

    from huggingface_hub import HfApi
    rows = "\n".join(f"| {t['id']} | `{t['bid']}` | `{t['play']}` |" for t in json.load(open(MODELS / "teams.json")))
    with tempfile.TemporaryDirectory() as tmp:
        import torch
        for f in names:
            ck = torch.load(MODELS / f, map_location="cpu", weights_only=False) if f.endswith(".pt") else None
            if isinstance(ck, dict) and "critic" in ck:   # training-only; the bots never read it
                ck.pop("critic")
                torch.save(ck, Path(tmp) / f)
            else:
                shutil.copy2(MODELS / f, Path(tmp) / f)
        (Path(tmp) / "README.md").write_text(CARD.format(gh=args.github, name=args.github.split("/")[-1], rows=rows))
        api = HfApi()
        api.create_repo(args.repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=args.repo, repo_type="model",
                          commit_message="Pidgin team models")
    print(f"uploaded: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
