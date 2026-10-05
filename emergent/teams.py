"""Teams: a bidding net, bidding search on or off, and a card player, served under one model id.

`models/teams.json` lists them; the Brill API's `model=<id>` (or `model_id=<id>`) picks one for
/bid, /lead and /play alike. Loaded at start only when `BRILL_TEAMS=1` (the Docker image sets it),
so the public site's memory stays as it was.

Every team answer is deterministic, as the Brill API's other answers are: the bidding search draws
its deals from a seed of the position and scores all of them (no clock), and the card search is
seeded by the position as the seat on turn sees it.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch

from bridgezero.bridge.auction import AuctionState
from emergent import engine
from emergent.deck import N_CALLS

TEAMS: "OrderedDict[str, Team]" = OrderedDict()


@dataclass
class Team:
    id: str
    label: str
    bid: object            # a bidserver bidding bot
    bid_search: bool
    play: object           # playdesk.PlayBot or playq.QPlayBot


def enabled(env=os.environ):
    return env.get("BRILL_TEAMS", "0") not in ("0", "", "false", "off")


def load_teams(models_dir=engine.MODELS_DIR):
    """Fill TEAMS from teams.json, reusing any net the desks already loaded from the same file."""
    from emergent import bidserver, playdesk, playq
    with open(os.path.join(models_dir, "teams.json")) as fh:
        manifest = json.load(fh)
    bids = {b.file: b for b in engine.BID_MODELS.values() if getattr(b, "file", None)}
    plays = {b.file: b for b in engine.PLAY_MODELS.values() if getattr(b, "file", None)}

    def bid_bot(file, family):
        if file not in bids:
            path = os.path.join(models_dir, file)
            bot = bidserver.BrlBot(path) if family == "brl" else bidserver.FourSeatBot(path)
            bot.id, bot.file = os.path.splitext(file)[0], file
            bids[file] = bot
        return bids[file]

    def play_bot(file):
        if file not in plays:
            plays[file] = playdesk.PlayBot(os.path.join(models_dir, file))
            plays[file].file = file
        return plays[file]

    for m in manifest:
        if m.get("play_family") == "playq":
            key = f"{m['play']}+{m['sampler']}"
            if key not in plays:
                plays[key] = playq.QPlayBot(os.path.join(models_dir, m["play"]), play_bot(m["sampler"]).net)
            play = plays[key]
        else:
            play = play_bot(m["play"])
        TEAMS[m["id"]] = Team(m["id"], m.get("label", ""), bid_bot(m["bid"], m.get("bid_family")),
                              bool(m.get("bid_search")), play)
    if any(t.bid_search for t in TEAMS.values()):
        engine.bid_searcher()            # load the belief net now, not on the first call


def default_team():
    want = os.environ.get("BRILL_DEFAULT_MODEL")
    return want if want in TEAMS else None


@torch.no_grad()
def team_call(team, hand, calls, dealer, vul):
    """(call, top) as `engine.choose_call`, through the bidding search when the team uses it."""
    call, top = engine.choose_call(team.bid, hand, calls, dealer, vul)
    if not team.bid_search:
        return call, top
    searcher = engine.bid_searcher()
    if searcher is None:
        raise RuntimeError("this team bids with search, but the belief net is missing")
    st = AuctionState.from_calls(list(calls), dealer=dealer)
    legal = engine.legal_calls(team.bid, st)
    d = team.bid.decide(torch.as_tensor(np.asarray(hand), dtype=torch.float32)[None],
                        list(calls), dealer, vul, legal)
    call, _ = searcher.choose(team.bid, np.asarray(hand), list(calls), dealer, vul,
                              d["policy"], legal, budget_ms=None)
    assert 0 <= call < N_CALLS and legal[call]
    return int(call), top
