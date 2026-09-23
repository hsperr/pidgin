"""Stage A: exact scoring oracle. No neural network in this file."""

import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from bridgezero.bridge.calls import CONTRACTS, parse_call
from bridgezero.bridge.deals import load_dataset
from bridgezero.bridge.scoring import contract_score, dd_cooperative_score
from bridgezero.contract.targets import (
    PAIR_PASS,
    TorchScorer,
    contract_score_table,
    cooperative_ceiling,
    endpoint_targets,
    flat_scores,
    pair_action_name,
)

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def brute_force(tricks, seat, vulnerable):
    best = 0
    table = np.zeros((2, 35), dtype=np.int64)
    strain_map = {"C": 3, "D": 2, "H": 1, "S": 0, "NT": 4}
    for c, (name, level, bid_strain) in enumerate(CONTRACTS):
        table_strain = strain_map[name[1:]]
        for rel, declarer in enumerate((seat, (seat + 2) % 4)):
            score = contract_score(level, bid_strain, int(tricks[declarer, table_strain]),
                                   0, vulnerable)
            table[rel, c] = score
            best = max(best, score)
    return table, best


def random_tables(n, seed):
    return np.random.default_rng(seed).integers(0, 14, size=(n, 4, 5))


def test_table_matches_brute_force_and_ceiling_on_random_tables():
    for tricks in random_tables(300, 0):
        for seat, vul in itertools.product(range(4), (False, True)):
            table = contract_score_table(tricks, seat, vul)
            expected, best = brute_force(tricks, seat, vul)
            assert np.array_equal(table, expected)
            assert cooperative_ceiling(table) == best
            assert best == dd_cooperative_score(tricks, seat % 2, vul)


def test_real_dds_rows_agree_with_existing_ceiling():
    owners, tricks = load_dataset(SMOKE)
    for table in tricks:
        for seat, vul in itertools.product(range(4), (False, True)):
            assert cooperative_ceiling(contract_score_table(table, seat, vul)) \
                == dd_cooperative_score(table, seat % 2, vul)


def test_pass_is_zero_and_ceiling_never_negative():
    tricks = np.zeros((4, 5), dtype=np.int64)
    table = contract_score_table(tricks, 0, True)
    assert table.max() < 0
    assert cooperative_ceiling(table) == 0
    targets = endpoint_targets(table)
    assert targets[PAIR_PASS] == 0.0
    assert targets.max() == 0.0


def test_grand_slam_vulnerable_and_nonvulnerable():
    tricks = np.full((4, 5), 13)
    assert cooperative_ceiling(contract_score_table(tricks, 1, False)) == 1520
    assert cooperative_ceiling(contract_score_table(tricks, 1, True)) == 2220


def test_strain_mapping_and_declarer():
    # Only South (partner of North) takes 10 tricks, in hearts.
    tricks = np.zeros((4, 5), dtype=np.int64)
    tricks[2, 1] = 10
    table = contract_score_table(tricks, 0, False)
    four_hearts = parse_call("4H")
    assert table[1, four_hearts] == 420   # partner declares
    assert table[0, four_hearts] == -500  # actor declares, 10 down
    assert table[1, parse_call("4S")] == -500
    assert cooperative_ceiling(table) == 420
    flat = flat_scores(table)
    assert pair_action_name(int(flat.argmax())) == "4H-partner"
    # Same deal seen from South: South is now "self".
    assert contract_score_table(tricks, 2, False)[0, four_hearts] == 420


def test_game_and_slam_boundaries():
    tricks = np.zeros((4, 5), dtype=np.int64)
    tricks[0, 4] = 9   # 3NT game
    assert cooperative_ceiling(contract_score_table(tricks, 0, False)) == 400
    tricks[0, 4] = 8   # 2NT partscore
    assert cooperative_ceiling(contract_score_table(tricks, 0, False)) == 120
    tricks[0, 3] = 11  # 5C game beats 2NT
    assert cooperative_ceiling(contract_score_table(tricks, 0, True)) == 600
    tricks[0, 0] = 12  # 6S small slam
    assert cooperative_ceiling(contract_score_table(tricks, 0, True)) == 1430


def test_endpoint_target_is_baseline_shifted_points_over_100():
    tricks = random_tables(1, 5)[0]
    table = contract_score_table(tricks, 3, True)
    targets = endpoint_targets(table)
    ceiling = cooperative_ceiling(table)
    assert np.allclose(targets * 100 + ceiling, flat_scores(table))


def test_torch_exact_and_one_hot_expected_scores_match_numpy():
    scorer = TorchScorer()
    tables = random_tables(64, 9)
    seats = np.arange(64) % 4
    vul = np.arange(64) % 2
    rel = np.stack([np.stack([t[s], t[(s + 2) % 4]]) for t, s in zip(tables, seats)])
    rel_t = torch.as_tensor(rel)
    vul_t = torch.as_tensor(vul)
    exact = scorer.exact(rel_t, vul_t)
    one_hot = torch.nn.functional.one_hot(rel_t, 14).float()
    expected = scorer.expected(one_hot, vul_t)
    for i in range(64):
        table = contract_score_table(tables[i], seats[i], bool(vul[i]))
        assert np.array_equal(exact[i].numpy(), table)
        assert np.allclose(expected[i].numpy(), table)
    assert torch.equal(scorer.ceiling(exact),
                       torch.as_tensor([cooperative_ceiling(contract_score_table(
                           tables[i], seats[i], bool(vul[i]))) for i in range(64)],
                           dtype=torch.float32))
    make = scorer.make_probability(one_hot)
    assert torch.equal(make > 0.5, exact > 0)


def test_expected_score_is_mixture_of_exact_scores():
    scorer = TorchScorer()
    probs = torch.zeros(1, 2, 5, 14)
    probs[0, :, :, 9] = 0.5
    probs[0, :, :, 8] = 0.5
    expected = scorer.expected(probs, torch.tensor([0]))
    three_nt = parse_call("3NT")
    assert float(expected[0, 0, three_nt]) == pytest.approx(0.5 * 400 + 0.5 * -50)
    assert float(scorer.make_probability(probs)[0, 0, three_nt]) == pytest.approx(0.5)
