"""Load dealt hands and their double-dummy trick tables (PGX .npy or .npz)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bridgezero.bridge.calls import STRAIN_PERM

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
