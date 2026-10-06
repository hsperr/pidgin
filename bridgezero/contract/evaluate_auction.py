"""Greedy silent-opponent auctions on fixed rows: every deal x side x dealer x vulnerability."""

from __future__ import annotations

import torch

from .data import TorchDeals
from .prefixes import CoopBatch, decision_values, observe
from .targets import TorchScorer


@torch.no_grad()
def auction_rows(n_deals: int, device="cpu") -> tuple[torch.Tensor, ...]:
    """(deal, side, dealer, vul): 16 rows per deal, fixed order."""
    deal = torch.arange(n_deals, device=device).repeat_interleave(16)
    side = torch.arange(2, device=device).repeat_interleave(8).repeat(n_deals)
    dealer = torch.arange(4, device=device).repeat_interleave(2).repeat(2 * n_deals)
    vul = torch.arange(2, device=device).repeat(8 * n_deals)
    return deal, side, dealer, vul


@torch.no_grad()
def run_auctions(net, deals: TorchDeals, rows, scorer: TorchScorer, rule: str,
                 chunk: int = 32768, feature_transform=None,
                 observation: str = "intact", record: list | None = None) -> CoopBatch:
    """Greedy auctions; both active players use ``net`` and ``rule``."""
    return continue_auctions(net, deals, CoopBatch.start(*rows), scorer, rule, chunk,
                             feature_transform, observation, record)


@torch.no_grad()
def continue_auctions(net, deals: TorchDeals, batch: CoopBatch, scorer: TorchScorer, rule: str,
                      chunk: int = 32768, feature_transform=None,
                      observation: str = "intact", record: list | None = None) -> CoopBatch:
    """Finish ``batch`` in place with greedy calls.

    ``record``, when given, receives a copy of every live decision state before its call.

    The network sees the ``observation`` regime's features, then
    ``feature_transform(sub, feats, rows)`` may change them further; ``rows``
    index ``batch``. Legality and ``apply`` always use the real auction state.
    """
    was_training = net.training
    net.eval()
    while not bool(batch.ended.all()):
        alive = (~batch.ended).nonzero().squeeze(1)
        if record is not None:
            record.append(batch.subset(alive))
        action = torch.full((len(batch),), -1, dtype=torch.long, device=batch.deal.device)
        for i in range(0, len(alive), chunk):
            idx = alive[i:i + chunk]
            sub = batch.subset(idx)
            feats = observe(sub, observation)
            if feature_transform is not None:
                feats = feature_transform(sub, feats, idx)
            values = decision_values(net, deals, sub, scorer, rule, feats)
            action[idx] = values.masked_fill(~sub.legal(), -torch.inf).argmax(-1)
        batch.apply(action, ~batch.ended)
    net.train(was_training)
    return batch
