"""Exact cooperative contract score tables and differentiable expected scores.

Conventions (see DATA_CONVENTIONS.md):

- DDS tables are ``tricks[seat, table_strain]`` with seats N,E,S,W and table
  strains S,H,D,C,NT.
- Contracts use bidding strains C,D,H,S,NT; ``CONTRACTS`` index 0..34.
- A partnership table is indexed by *relative declarer* 0=actor (``seat``),
  1=partner (``seat + 2``).
- A flat partnership action is ``declarer * 35 + contract``; index 70 is Pass.

Endpoint targets are ``(score - cooperative_ceiling) / 100``.
"""

from __future__ import annotations

import numpy as np
import torch

from ..bridge.calls import CONTRACTS
from ..bridge.scoring import contract_score
from training.bridge.calls import STRAIN_PERM

N_CONTRACTS = len(CONTRACTS)
N_DECLARERS = 2
PAIR_PASS = N_DECLARERS * N_CONTRACTS
N_PAIR_ACTIONS = PAIR_PASS + 1
BID_TO_TABLE_STRAIN = np.asarray(STRAIN_PERM, dtype=np.int64)
CONTRACT_LEVEL = np.asarray([level for _, level, _ in CONTRACTS], dtype=np.int64)
CONTRACT_BID_STRAIN = np.asarray([strain for _, _, strain in CONTRACTS], dtype=np.int64)
CONTRACT_TABLE_STRAIN = BID_TO_TABLE_STRAIN[CONTRACT_BID_STRAIN]
TARGET_SCALE = 100.0

# SCORE_LOOKUP[vulnerable, contract, tricks] = exact undoubled declarer score.
SCORE_LOOKUP = np.asarray([
    [[contract_score(level, strain, k, 0, bool(vul)) for k in range(14)]
     for _, level, strain in CONTRACTS]
    for vul in (0, 1)
], dtype=np.int32)


def pair_action_name(action: int) -> str:
    if action == PAIR_PASS:
        return "Pass"
    declarer, contract = divmod(int(action), N_CONTRACTS)
    return f"{CONTRACTS[contract][0]}-{('self', 'partner')[declarer]}"


def contract_score_table(tricks: np.ndarray, seat: int, vulnerable: bool) -> np.ndarray:
    """Exact ``(2, 35)`` undoubled scores for one deal, actor ``seat``."""
    rel = np.stack([tricks[seat % 4], tricks[(seat + 2) % 4]])
    taken = rel[:, CONTRACT_TABLE_STRAIN].astype(np.int64)
    return SCORE_LOOKUP[int(bool(vulnerable)), np.arange(N_CONTRACTS)[None, :], taken]


def cooperative_ceiling(score_table: np.ndarray) -> int:
    """Best cooperative result; Pass (zero) is always available."""
    return max(0, int(score_table.max()))


def flat_scores(score_table: np.ndarray) -> np.ndarray:
    """``(2,35)`` table -> ``(71,)`` pair-action scores with Pass=0 at the end."""
    return np.concatenate([score_table.reshape(-1), [0]]).astype(np.int64)


def endpoint_targets(score_table: np.ndarray) -> np.ndarray:
    """Per-deal dense targets ``(S - C*) / 100`` for all 71 pair actions."""
    return (flat_scores(score_table) - cooperative_ceiling(score_table)) / TARGET_SCALE


# ---------------------------------------------------------------------------
# Batched torch versions (used by training and evaluation).


class TorchScorer:
    """Vectorized exact and expected cooperative scores on one device."""

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.lookup = torch.as_tensor(SCORE_LOOKUP, dtype=torch.float32, device=self.device)
        self.table_strain = torch.as_tensor(CONTRACT_TABLE_STRAIN, device=self.device)
        self.level = torch.as_tensor(CONTRACT_LEVEL, device=self.device)

    def exact(self, rel_tricks: torch.Tensor, vulnerable: torch.Tensor) -> torch.Tensor:
        """``rel_tricks (B,2,5)`` long, ``vulnerable (B,)`` -> scores ``(B,2,35)``."""
        taken = rel_tricks[:, :, self.table_strain]              # (B,2,35)
        rows = self.lookup[vulnerable.long()]                     # (B,35,14)
        rows = rows[:, None].expand(-1, rel_tricks.shape[1], -1, -1)
        return rows.gather(-1, taken.long().unsqueeze(-1)).squeeze(-1)

    def expected(self, trick_probs: torch.Tensor, vulnerable: torch.Tensor) -> torch.Tensor:
        """``trick_probs (B,2,5,14)`` -> differentiable expected scores ``(B,2,35)``."""
        per_contract = trick_probs[:, :, self.table_strain, :]   # (B,2,35,14)
        rows = self.lookup[vulnerable.long()]                     # (B,35,14)
        return (per_contract * rows[:, None]).sum(-1)

    def make_probability(self, trick_probs: torch.Tensor) -> torch.Tensor:
        """``P(tricks >= level + 6)`` for every declarer/contract ``(B,2,35)``."""
        per_contract = trick_probs[:, :, self.table_strain, :]
        tail = per_contract.flip(-1).cumsum(-1).flip(-1)          # P(T >= k)
        needed = (self.level + 6).view(1, 1, -1, 1).expand(*per_contract.shape[:3], 1)
        return tail.gather(-1, needed).squeeze(-1)

    @staticmethod
    def flat(scores: torch.Tensor) -> torch.Tensor:
        """``(B,2,35)`` -> ``(B,71)`` with Pass=0 appended."""
        batch = scores.shape[0]
        return torch.cat([scores.reshape(batch, -1), scores.new_zeros(batch, 1)], dim=-1)

    @staticmethod
    def ceiling(scores: torch.Tensor) -> torch.Tensor:
        return scores.reshape(scores.shape[0], -1).max(-1).values.clamp(min=0)
