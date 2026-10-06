import numpy as np

from training.bridge.deals import (
    PGX_RANK_TO_OURS,
    PackedPGXOwners,
    PackedPGXTricks,
)


def encode_owners(owners):
    keys = np.zeros(4, dtype=np.int32)
    inverse = np.argsort(PGX_RANK_TO_OURS)
    for suit in range(4):
        for raw_rank in range(13):
            our_rank = inverse[raw_rank]
            keys[suit] |= int(owners[suit * 13 + our_rank]) << (2 * (12 - raw_rank))
    return keys


def test_lazy_pgx_decoding_matches_known_layout():
    owners = np.repeat(np.arange(4, dtype=np.uint8), 13)
    # Encode directly from the documented raw rank ordering.
    keys = np.zeros(4, dtype=np.int32)
    for suit in range(4):
        for raw_rank, our_rank in enumerate(PGX_RANK_TO_OURS):
            keys[suit] |= int(owners[suit * 13 + our_rank]) << (2 * (12 - raw_rank))
    packed = np.zeros((2, 1, 4), dtype=np.int32)
    packed[0, 0] = keys
    decoded = PackedPGXOwners(packed)[0]
    assert np.array_equal(decoded, owners)


def test_lazy_trick_decoding_and_strain_permutation():
    packed = np.zeros((2, 1, 4), dtype=np.int32)
    expected_raw = np.asarray([
        [1, 2, 3, 4, 5],
        [5, 6, 7, 8, 9],
        [9, 8, 7, 6, 5],
        [2, 4, 6, 8, 10],
    ], dtype=np.uint8)
    for player in range(4):
        value = 0
        for strain in range(5):
            value |= int(expected_raw[player, strain]) << (4 * (4 - strain))
        packed[1, 0, player] = value
    decoded = PackedPGXTricks(packed)[0]
    assert np.array_equal(decoded, expected_raw[:, [3, 2, 1, 0, 4]])
