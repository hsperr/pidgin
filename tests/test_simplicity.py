"""The public report and training penalty agree on a flagged call."""

from types import SimpleNamespace

import numpy as np
import torch

from training.bridge.calls import PASS
from training.contract.data import TorchDeals
from training.fourseat.competitive import code_word_mask
from tools.simplicity import analyse


def test_artificial_two_clubs_counts_once_even_without_clubs():
    # N holds all spades; 2C is both short and an artificial opening.
    owners = np.repeat(np.arange(4), 13)[None]
    deals = TorchDeals(owners, np.full((1, 4, 5), 7))
    states = SimpleNamespace(actor_seat=torch.tensor([0]), deal=torch.tensor([0]),
                             dealer=torch.tensor([0]), t=torch.tensor([0]),
                             history=torch.full((1, 8), -1))
    assert code_word_mask(states, torch.tensor([5]), deals).tolist() == [True]
    boards = {"dealer": np.array([0]), "hist1": np.array([[5, PASS, PASS, PASS]])}
    assert analyse(boards, owners=owners)["code_words_per_100"] == 100


def test_report_matches_training_flags_for_partner_raise_and_low_double():
    owners = np.repeat(np.arange(4), 13)[None]
    deals = TorchDeals(owners, np.full((1, 4, 5), 7))
    # S has diamonds, N has spades: 1S P 2S X P P P.
    history = [3, PASS, 8, 36, PASS, PASS, PASS]
    flags = []
    for t, action in enumerate(history):
        states = SimpleNamespace(actor_seat=torch.tensor([t % 4]), deal=torch.tensor([0]),
                                 dealer=torch.tensor([0]), t=torch.tensor([t]),
                                 history=torch.tensor([history + [-1]]))
        flags.append(bool(code_word_mask(states, torch.tensor([action]), deals)[0]))
    boards = {"dealer": np.array([0]), "hist1": np.array([history])}
    assert analyse(boards, owners=owners)["code_words_per_100"] == 100 * sum(flags)


def test_light_open_share_counts_0_to_7_hcp_hands_that_open():
    lows = [c for c in range(52) if c % 13 >= 4]          # no A K Q J
    rest = [c for c in range(52) if c not in lows[:13]]
    owners = np.zeros((2, 52), dtype=np.int64)
    owners[:, lows[:13]] = 0                              # N: 0 HCP
    for seat in (1, 2, 3):
        owners[:, rest[(seat - 1) * 13:seat * 13]] = seat
    # board 0: N (dealer, 0 HCP) opens 1S; board 1: N passes, E opens.
    boards = {"dealer": np.array([0, 0]),
              "hist1": np.array([[3, PASS, PASS, PASS, -1], [PASS, 3, PASS, PASS, PASS]])}
    out = analyse(boards, owners=owners)
    # light chances: N on both boards (E, S, W hold 12+ HCP here) -> 1 of 2 opened
    assert out["light_open_share"] == 0.5


def test_trainer_logs_the_shared_simplicity_numbers():
    from training.bridge.deals import load_dataset
    from training.contract.targets import TorchScorer
    from training.fourseat import competitive as C
    from training.fourseat.model import FourSeatCompetitiveNet
    from training.simplicity import analyse_batch
    from pathlib import Path
    deals = TorchDeals(*load_dataset(Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"))
    torch.manual_seed(2)
    net = FourSeatCompetitiveNet(24, 8, 2, redouble=True, sacrifice=True)
    seen = {}
    play = C.play
    try:
        C.play = lambda *a, **k: seen.setdefault("batch", play(*a, **k))
        logged = C.competitive_validation(net, deals, TorchScorer())["simplicity"]
    finally:
        C.play = play
    shared = analyse_batch(seen["batch"], deals)
    assert "light_open_share" in logged
    assert all(logged[k] == shared[k] for k in logged)


def test_opening_numbers_flag_unbalanced_nt_and_wide_calls():
    from training.simplicity import opening_numbers
    one_nt, one_s = 4, 3
    calls = [one_nt] * 4 + [one_s] * 40 + [PASS] * 2
    hcp = [16, 17, 18, 20] + list(range(0, 20)) * 2 + [3, 5]
    balanced = [True, True, False, False] + [True] * 40 + [True, True]
    out = opening_numbers(calls, hcp, balanced)
    assert out["unbalanced_nt_share"] == 0.5
    assert out["light_open_share"] == 16 / 18         # 0-7 HCP: 16 opened 1S, 2 passed
    assert 15 < out["opening_hcp_spread"] < 19        # only 1S has 20+ hands: 1..18

def test_light_open_flags_openings_on_7_hcp_or_less_but_not_preempts():
    from training.fourseat.competitive import code_word_parts
    # N: spades T..2 (9 cards) and hearts 5..2, 0 HCP; the rest dealt in order to E, S, W.
    north = [s for s in range(4, 13)] + [13 + r for r in range(9, 13)]
    rest = [c for c in range(52) if c not in north]
    owners = np.zeros(52, dtype=np.int64)
    owners[rest] = np.repeat([1, 2, 3], 13)
    deals = TorchDeals(owners[None], np.full((1, 4, 5), 7))

    def light(action, history=()):
        h = list(history) + [-1] * (8 - len(history))
        states = SimpleNamespace(actor_seat=torch.tensor([0]), deal=torch.tensor([0]),
                                 dealer=torch.tensor([(4 - len(history)) % 4]),
                                 t=torch.tensor([len(history)]), history=torch.tensor([h]))
        return bool(code_word_parts(states, torch.tensor([action]), deals)["light_open"][0])

    assert light(3)                          # 1S opening on 0 HCP
    assert light(3, [PASS, PASS])            # same after two passes
    assert not light(8)                      # 2S with 9 spades: a natural preempt
    assert not light(PASS)
    assert not light(3, [0])                 # 1C was opened before: not an opening
