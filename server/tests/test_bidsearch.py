"""The bidding search (emergent/bidsearch.py): /debug with search on, nowhere else.

    python3 -m pytest -q tests/test_bidsearch.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.bridge.auction import AuctionState  # noqa: E402
from emergent import bidserver, bidsearch, engine  # noqa: E402
from emergent.deck import owners_to_bitmaps, owners_to_pbn  # noqa: E402

PASS = 35


@pytest.fixture(scope="module")
def client():
    if not engine.BID_MODELS:
        bidserver.create_app()
    return bidserver.app.test_client()


@pytest.fixture(scope="module")
def searcher():
    s = engine.bid_searcher()
    assert s is not None, "models/belief_r2.pt missing (sync_models.sh)"
    return s


def owners_of(seed):
    return np.random.default_rng(seed).permutation(np.repeat(np.arange(4), 13))


def test_dd_tricks_match_the_full_table():
    """One contract solved alone = that entry of the whole DD table."""
    from endplay.dds import calc_dd_table
    from endplay.types import Deal, Denom, Player
    owners = owners_of(3)
    table = calc_dd_table(Deal.from_pbn(owners_to_pbn(owners)))
    for declarer, strain, denom in ((0, 3, Denom.spades), (1, 4, Denom.nt), (2, 0, Denom.clubs), (3, 2, Denom.hearts)):
        assert bidsearch.dd_tricks(owners, declarer, strain) == table[denom, Player(declarer)]


@pytest.mark.parametrize("model", ["lo_s28k", "brl_fsp"])
def test_batched_policy_is_decides_policy(client, model):
    """The rollouts' batched forward pass gives the numbers the bot bids with."""
    bot = engine.BID_MODELS[model]
    owners = owners_of(1)
    calls, dealer, vul = [0, PASS, 2], 1, (True, False)
    st = AuctionState.from_calls(calls, dealer=dealer)
    legal = engine.legal_calls(bot, st)
    hand = torch.tensor(owners_to_bitmaps(owners)[st.turn], dtype=torch.float32)[None]
    d = bot.decide(hand, calls, dealer, vul, legal)
    lp = bot.batch_log_probs(hand.expand(2, -1), torch.tensor([calls, calls]), torch.tensor([dealer, dealer]),
                             torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]), torch.tensor([st.turn] * 2),
                             torch.tensor([legal, legal]))
    assert torch.allclose(lp[0].exp(), torch.tensor(d["policy"]), atol=1e-6)
    assert torch.equal(lp[0], lp[1])


def test_samples_keep_the_hand_and_the_seed(searcher):
    owners = owners_of(2)
    hand = owners_to_bitmaps(owners)[1]
    draw = lambda: bidsearch.sample_deals(searcher.net, hand, [0], 0, (False, True), 8,  # noqa: E731
                                          torch.Generator().manual_seed(7))
    a = draw()
    assert a.shape == (8, 52)
    assert ((a == 1) == (hand == 1)[None]).all()                     # East's own cards, only them
    assert all((np.bincount(row, minlength=4) == 13).all() for row in a)
    assert (a == draw()).all()                                       # same seed, same deals


def test_search_picks_a_legal_call_and_falls_back(searcher):
    bot = engine.BID_MODELS["lo_s28k"]
    owners = owners_of(4)
    for calls in ([], [PASS], [0, PASS], [4, PASS, PASS]):
        st = AuctionState.from_calls(calls)
        legal = engine.legal_calls(bot, st)
        hand = owners_to_bitmaps(owners)[st.turn]
        d = bot.decide(torch.tensor(hand, dtype=torch.float32)[None], calls, 0, (False, False), legal)
        small = bidsearch.BidSearch(searcher.net, samples=4, k=3, margin=50.0, budget_ms=None, min_samples=2)
        call, info = small.choose(bot, hand, calls, 0, (False, False), d["policy"], legal)
        assert legal[call]
        if info is None:                                             # one candidate: no search
            assert call == d["pick"]
            continue
        assert info["used"] == 4 and info["cands"][0] == d["pick"] and len(info["means"]) == len(info["cands"])
        # never leaves the net's call without a margin to beat
        never = bidsearch.BidSearch(searcher.net, samples=4, k=3, margin=float("inf"), budget_ms=None)
        assert never.choose(bot, hand, calls, 0, (False, False), d["policy"], legal)[0] == d["pick"]
        # no time: the net's call, and it says so
        late = bidsearch.BidSearch(searcher.net, samples=4, k=3, margin=0.0, budget_ms=0.0)
        call, info = late.choose(bot, hand, calls, 0, (False, False), d["policy"], legal)
        assert call == d["pick"] and "over budget" in info["skip"]
        assert "net's call" not in bidsearch.describe(calls, 0, info)


def test_search_runs_only_on_debug_with_search_on(client, monkeypatch):
    """/debug + search on: the auction goes through the search. Search off, or /: never."""
    seen = []

    def spy(bot, hand, calls, dealer=0, vul=(False, False)):
        seen.append((bot.id, list(calls), dealer, vul))
        return engine.choose_call(bot, hand, calls, dealer, vul)[0]

    monkeypatch.setattr(engine, "search_call", spy)
    h = {"X-Game": "bs-debug"}
    d = client.post("/api/table/debug", json={"dealer": "E", "vul": "ew", "model": "hi3_s60k"}, headers=h).json
    assert d["debug"] and "error" not in d                           # a random deal, hi3 bids
    client.post("/api/table/hints", json={"search": True}, headers=h)
    assert client.post("/api/table/step", headers=h).status_code == 200
    assert seen == [("hi3_s60k", [], 1, (False, True))]              # the picked net, dealer, vul
    client.post("/api/table/hints", json={"search": False}, headers=h)
    client.post("/api/table/step", headers=h)
    assert len(seen) == 1                                            # off: the plain net
    h = {"X-Game": "bs-main"}                                        # /: never
    client.post("/api/table/hints", json={"search": True}, headers=h)
    d = client.post("/api/table/new_board", json={}, headers=h).json      # the nets bid up to South
    assert d["phase"] == "auction" and not d["debug"]
    client.post("/api/table/call", json={"call": PASS}, headers=h)  # and on after South's pass
    assert len(seen) == 1
