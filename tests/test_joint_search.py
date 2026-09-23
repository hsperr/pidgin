from pathlib import Path

import torch

from bridgezero.bridge.deals import load_dataset
from bridgezero.contract.data import TorchDeals
from bridgezero.contract.model import AuctionContractNet
from bridgezero.contract.prefixes import CoopBatch
from bridgezero.contract.targets import TorchScorer
from bridgezero.cooperative.joint_search import (
    build_joint_branches,
    joint_objective_from_logits,
    run_n_ply,
    sample_opening_roots,
)

SMOKE = Path(__file__).resolve().parents[1] / "data" / "smoke_128.npz"


def deals():
    return TorchDeals(*load_dataset(SMOKE))


def branches(n=2):
    data = deals()
    net = AuctionContractNet(16, 8, 1)
    roots = sample_opening_roots(data, n, torch.Generator().manual_seed(5))
    return data, build_joint_branches(net, data, roots, TorchScorer())


def test_joint_branch_table_enumerates_only_legal_calls():
    _, table = branches()
    assert len(table.root_row) == 36 * len(table.roots)
    assert bool(table.roots.legal()[table.root_row, table.root_action].all())
    assert bool(table.listeners.legal()[table.reply_listener_row, table.reply_action].all())
    # At an opening, none of the first calls ends the auction.  There are
    # 36 + sum(1..35) = 666 legal (opening, reply) pairs per root.
    assert not bool(torch.isfinite(table.root_edge_return).any())
    assert len(table.reply_action) == 666 * len(table.roots)
    assert table.branch_count == 666 * len(table.roots)


def test_joint_objective_matches_explicit_ragged_sum_and_trains_both_roles():
    _, table = branches()
    generator = torch.Generator().manual_seed(9)
    root_logits = torch.randn(len(table.roots), 36, generator=generator, requires_grad=True)
    listener_logits = torch.randn(
        len(table.listeners), 36, generator=generator, requires_grad=True)
    result = joint_objective_from_logits(
        root_logits, listener_logits, table, entropy_weight=0.0, temperature=0.7)

    root_policy = torch.softmax(
        (root_logits / 0.7).masked_fill(~table.roots.legal(), -1e9), -1)
    listener_policy = torch.softmax(
        (listener_logits / 0.7).masked_fill(~table.listeners.legal(), -1e9), -1)
    manual = root_logits.new_zeros(len(table.roots))
    for edge in range(len(table.root_row)):
        listener = edge  # opening roots make every root edge non-terminal
        replies = table.reply_listener_row == listener
        value = (listener_policy[listener, table.reply_action[replies]]
                 * table.reply_return[replies]).sum()
        manual[table.root_row[edge]] += root_policy[
            table.root_row[edge], table.root_action[edge]] * value
    assert torch.allclose(result["return"], manual.mean(), atol=1e-6)

    result["loss"].backward()
    assert root_logits.grad is not None and float(root_logits.grad.norm()) > 0
    assert listener_logits.grad is not None and float(listener_logits.grad.norm()) > 0


def test_pass_after_a_standing_contract_is_scored_without_a_listener():
    data = deals()
    net = AuctionContractNet(16, 8, 1)
    roots = CoopBatch.start(torch.tensor([0]), torch.tensor([0]),
                            torch.tensor([0]), torch.tensor([0]))
    roots.apply(torch.tensor([0]), torch.tensor([True]))  # 1C by opener
    table = build_joint_branches(net, data, roots, TorchScorer())
    pass_edge = table.root_action == 35
    assert int(pass_edge.sum()) == 1
    assert bool(torch.isfinite(table.root_edge_return[pass_edge]).all())
    assert not bool(torch.isfinite(table.root_edge_return[~pass_edge]).any())


def test_forced_stop_joint_search_needs_no_blueprint():
    data = deals()
    roots = sample_opening_roots(data, 2, torch.Generator().manual_seed(17))
    table = build_joint_branches(
        None, data, roots, TorchScorer(), continuation_rule="stop")
    assert len(table.reply_return) == 2 * 666
    assert bool(torch.isfinite(table.reply_return).all())


def test_one_and_two_ply_evaluators_terminate_normally():
    data = deals()
    net = AuctionContractNet(16, 8, 1)
    for depth in (1, 2):
        roots = sample_opening_roots(data, 8, torch.Generator().manual_seed(depth))
        final = run_n_ply(net, data, roots, TorchScorer(), depth)
        assert bool(final.ended.all())
        assert bool((final.k >= depth).all())
