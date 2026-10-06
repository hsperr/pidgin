"""Deal ranges and leak-free model inputs for the contract-finder stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..bridge.deals import load_dataset

# Stage -> which hands the network may see.
STAGE_INPUTS = {"B": "partnership", "C": "single"}
INPUT_HANDS = {"single": 1, "partnership": 2}


def resolve_range(total: int, start: int, count: int) -> tuple[int, int]:
    first = start if start >= 0 else total + start
    if count <= 0 or first < 0 or first + count > total:
        raise ValueError(f"range start={start} count={count} is outside 0..{total}")
    return first, first + count


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


class TorchDeals:
    """A contiguous deal range as dense tensors.

    ``hands[i, seat, card]`` is 1 when ``seat`` holds ``card``; ``tricks[i, seat,
    table_strain]`` is the DDS table. Tricks are labels only and never inputs.
    """

    def __init__(self, owners: np.ndarray, tricks: np.ndarray,
                 device: torch.device | str = "cpu", first_index: int = 0,
                 source: str = ""):
        owners = np.asarray(owners, dtype=np.int64)
        tricks = np.asarray(tricks, dtype=np.int64)
        counts = np.stack([(owners == s).sum(1) for s in range(4)], axis=1)
        if not np.all(counts == 13):
            raise ValueError("every seat must hold exactly 13 cards")
        if tricks.shape != (len(owners), 4, 5) or tricks.min() < 0 or tricks.max() > 13:
            raise ValueError("invalid trick tables")
        self.n = len(owners)
        self.first_index = first_index
        self.source = source
        self.hands = torch.as_tensor(
            owners[:, None, :] == np.arange(4)[None, :, None],
            dtype=torch.float32, device=device)
        self.tricks = torch.as_tensor(tricks, device=device)

    def head(self, count: int) -> "TorchDeals":
        """The first ``count`` deals as a view sharing tensors."""
        out = TorchDeals.__new__(TorchDeals)
        out.__dict__.update(self.__dict__)
        out.n = min(count, self.n)
        out.hands, out.tricks = self.hands[:out.n], self.tricks[:out.n]
        return out

    def inputs(self, idx: torch.Tensor, seat: torch.Tensor, mode: str) -> torch.Tensor:
        """Visible hands ``(B, H, 52)``: actor first, then partner if allowed."""
        own = self.hands[idx, seat % 4]
        if mode == "single":
            return own[:, None]
        if mode == "partnership":
            return torch.stack([own, self.hands[idx, (seat + 2) % 4]], dim=1)
        raise ValueError(f"unknown input mode {mode!r}")

    def rel_tricks(self, idx: torch.Tensor, seat: torch.Tensor) -> torch.Tensor:
        """Label table ``(B, 2, 5)`` for actor and partner as declarer."""
        return torch.stack([self.tricks[idx, seat % 4],
                            self.tricks[idx, (seat + 2) % 4]], dim=1)


def load_range(path: str | Path, start: int, count: int,
               device: torch.device | str = "cpu") -> TorchDeals:
    """Load ``count`` deals from ``start`` (negative = from the end) lazily."""
    owners, tricks = load_dataset(path)
    first, stop = resolve_range(len(owners), start, count)
    return TorchDeals(owners[first:stop], tricks[first:stop], device, first, str(path))


def dataset_size(path: str | Path) -> int:
    owners, _ = load_dataset(path)
    return len(owners)


def block_starts(pool_start: int, pool_end: int, size: int, seed: int) -> list[int]:
    """Non-overlapping ``size``-deal blocks of [pool_start, pool_end) in a seeded order.

    A partial final block is dropped. Episodes sample within blocks with replacement.
    """
    starts = np.arange(pool_start, pool_end - size + 1, size)
    return [int(s) for s in np.random.default_rng(seed).permutation(starts)]


def validate_training_pool(total: int, start: int, end: int, size: int, every: int) -> None:
    """Reject invalid pools before loading blocks or allocating a model."""
    if not 0 <= start < end <= total:
        raise ValueError(f"training pool [{start}, {end}) is outside 0..{total}")
    if size <= 0 or size > end - start:
        raise ValueError("training block size must be positive and fit inside the pool")
    if every <= 0:
        raise ValueError("training block interval must be positive")
