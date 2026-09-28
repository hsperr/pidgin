"""Random deals, compact storage, PBN conversion, and DDS generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from bridgezero.bridge.calls import STRAIN_PERM

SUITS = ("S", "H", "D", "C")
RANKS = "AKQJT98765432"
SEATS = ("N", "E", "S", "W")
PGX_RANK_TO_OURS = np.asarray([0] + [13 - r for r in range(1, 13)], dtype=np.int64)
PGX_STRAIN_TO_OURS = np.asarray(STRAIN_PERM, dtype=np.int64)


class PackedPGXOwners:
    """Lazy owner decoder for a memory-mapped Pgx `(2,n,4)` array."""

    def __init__(self, packed: np.ndarray):
        self.packed = packed

    def __len__(self) -> int:
        return int(self.packed.shape[1])

    def __getitem__(self, index):
        keys = np.asarray(self.packed[0, index])
        out = np.empty(keys.shape[:-1] + (52,), dtype=np.uint8)
        for suit in range(4):
            key = keys[..., suit].astype(np.uint32)
            for raw_rank in range(13):
                out[..., suit * 13 + PGX_RANK_TO_OURS[raw_rank]] = (
                    key >> np.uint32(2 * (12 - raw_rank))) & np.uint32(3)
        return out


class PackedPGXTricks:
    """Lazy trick-table decoder paired with :class:`PackedPGXOwners`."""

    def __init__(self, packed: np.ndarray):
        self.packed = packed

    def __len__(self) -> int:
        return int(self.packed.shape[1])

    def __getitem__(self, index):
        values = np.asarray(self.packed[1, index])
        raw = np.empty(values.shape + (5,), dtype=np.uint8)
        for player in range(4):
            value = values[..., player].astype(np.uint32)
            for strain in range(5):
                raw[..., player, strain] = (
                    value >> np.uint32(4 * (4 - strain))) & np.uint32(15)
        return raw[..., PGX_STRAIN_TO_OURS]


def random_deals(n: int, seed: int) -> np.ndarray:
    """Return uint8 owner arrays with shape (n, 52)."""
    rng = np.random.default_rng(seed)
    owners = np.empty((n, 52), dtype=np.uint8)
    base = np.repeat(np.arange(4, dtype=np.uint8), 13)
    for i in range(n):
        owners[i] = rng.permutation(base)
    return owners


def owner_to_hands(owners: np.ndarray) -> np.ndarray:
    return (owners[:, None, :] == np.arange(4, dtype=np.uint8)[None, :, None]).astype(np.uint8)


def deal_to_pbn(owners: np.ndarray) -> str:
    hands = []
    for seat in range(4):
        pieces = []
        for suit in range(4):
            pieces.append("".join(
                rank for r, rank in enumerate(RANKS)
                if int(owners[suit * 13 + r]) == seat
            ))
        hands.append(".".join(pieces))
    return "N:" + " ".join(hands)


def format_hand(hand: np.ndarray) -> str:
    return " ".join(
        f"{SUITS[s]}:{''.join(RANKS[r] for r in range(13) if hand[s * 13 + r]) or '-'}"
        for s in range(4)
    )


def hand_hcp(hand: np.ndarray) -> int:
    return int(sum(int(hand[s * 13 + r]) * (4 - r) for s in range(4) for r in range(4)))


def compute_dd_tables(owners: np.ndarray) -> np.ndarray:
    try:
        from endplay.dds import calc_dd_table
        from endplay.types import Deal, Denom, Player
    except ImportError as exc:
        raise RuntimeError("DDS generation requires the optional 'endplay' package") from exc

    denoms = (Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt)
    out = np.empty((len(owners), 4, 5), dtype=np.uint8)
    for i, owner in enumerate(owners):
        table = calc_dd_table(Deal(deal_to_pbn(owner)))
        for strain, denom in enumerate(denoms):
            for seat, player in enumerate(Player):
                out[i, seat, strain] = table[denom, player]
        if (i + 1) % 25 == 0 or i + 1 == len(owners):
            print(f"DDS {i + 1}/{len(owners)}", flush=True)
    return out


def save_dataset(path: str | Path, n: int, seed: int) -> None:
    owners = random_deals(n, seed)
    tricks = compute_dd_tables(owners)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, owners=owners, tricks=tricks, seed=np.int64(seed))
    print(f"saved {n} deals to {path}")


def load_dataset(path: str | Path):
    path = Path(path)
    if path.suffix == ".npy":
        packed = np.load(path, mmap_mode="r")
        if packed.shape[0] != 2 or packed.shape[2] != 4 or packed.dtype != np.int32:
            raise ValueError("packed Pgx file must have shape (2,n,4) and dtype int32")
        return PackedPGXOwners(packed), PackedPGXTricks(packed)

    data = np.load(path)
    owners = np.asarray(data["owners"], dtype=np.uint8)
    tricks = np.asarray(data["tricks"], dtype=np.uint8)
    if owners.ndim != 2 or owners.shape[1] != 52:
        raise ValueError("owners must have shape (n, 52)")
    if tricks.shape != (len(owners), 4, 5):
        raise ValueError("tricks must have shape (n, 4, 5)")
    counts = np.stack([(owners == s).sum(1) for s in range(4)], axis=1)
    if not np.all(counts == 13):
        raise ValueError("every seat must hold exactly 13 cards")
    return owners, tricks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random deals with DDS tables")
    parser.add_argument("--out", required=True)
    parser.add_argument("--deals", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    save_dataset(args.out, args.deals, args.seed)


if __name__ == "__main__":
    main()
