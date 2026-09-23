"""Fixed held-out metrics and baselines for Stage B/C contract finders.

Every eval deal is scored for all four actor seats and both vulnerabilities, so
the row set is fully determined by the deal range. DDS tables are used only to
score selections and build labels; model inputs come from ``TorchDeals.inputs``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..bridge.scoring import IMP_LOWER_BOUNDS
from .data import TorchDeals, load_range
from .model import ContractNet, load_checkpoint
from .targets import (
    CONTRACT_BID_STRAIN,
    CONTRACT_LEVEL,
    CONTRACT_TABLE_STRAIN,
    N_CONTRACTS,
    PAIR_PASS,
    TorchScorer,
    pair_action_name,
)

IMP_BOUNDS = np.asarray(IMP_LOWER_BOUNDS)
HCP_WEIGHTS = torch.tensor([4.0, 3.0, 2.0, 1.0] + [0.0] * 9).repeat(4)
SINGLE_BUCKETS = ((0, 7), (8, 11), (12, 14), (15, 17), (18, 21), (22, 40))
PAIR_BUCKETS = ((0, 17), (18, 22), (23, 25), (26, 29), (30, 32), (33, 36), (37, 40))
CHUNK = 16384


def imps_array(points: np.ndarray) -> np.ndarray:
    return np.sign(points) * np.searchsorted(IMP_BOUNDS, np.abs(points), side="right")


def grid_rows(n_deals: int, device="cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All (deal, seat 0..3, vulnerable 0/1) rows, 8 per deal."""
    deal = torch.arange(n_deals, device=device).repeat_interleave(8)
    seat = torch.arange(4, device=device).repeat_interleave(2).repeat(n_deals)
    vul = torch.arange(2, device=device).repeat(4 * n_deals)
    return deal, seat, vul


@torch.no_grad()
def exact_rows(deals: TorchDeals, rows, scorer: TorchScorer):
    deal, seat, vul = rows
    flats, ceilings, rels = [], [], []
    for i in range(0, len(deal), CHUNK):
        sl = slice(i, i + CHUNK)
        rel = deals.rel_tricks(deal[sl], seat[sl])
        exact = scorer.exact(rel, vul[sl])
        flats.append(scorer.flat(exact))
        ceilings.append(scorer.ceiling(exact))
        rels.append(rel)
    return torch.cat(flats), torch.cat(ceilings), torch.cat(rels)


@torch.no_grad()
def predict(net: ContractNet, deals: TorchDeals, rows) -> tuple[torch.Tensor, torch.Tensor]:
    deal, seat, vul = rows
    was_training = net.training
    net.eval()
    probs, qs = [], []
    for i in range(0, len(deal), CHUNK):
        sl = slice(i, i + CHUNK)
        out = net(deals.inputs(deal[sl], seat[sl], net.inputs), vul[sl])
        probs.append(torch.softmax(out["trick_logits"], -1))
        qs.append(out["contract_q"])
    net.train(was_training)
    return torch.cat(probs), torch.cat(qs)


def choose(values: torch.Tensor, self_declarer_only: bool = False) -> torch.Tensor:
    if self_declarer_only:
        values = values.clone()
        values[:, N_CONTRACTS:PAIR_PASS] = -math.inf
    return values.argmax(-1)


def hand_features(deals: TorchDeals, rows, mode: str):
    """Actor/partnership HCP and longest suit (table strain) for diagnostics."""
    deal, seat, _ = rows
    weights = HCP_WEIGHTS.to(deals.hands.device)
    own = deals.hands[deal, seat % 4]
    partner = deals.hands[deal, (seat + 2) % 4]
    view = own if mode == "single" else own + partner
    lengths = view.view(-1, 4, 13).sum(-1)
    return {
        "hcp_self": (own @ weights).cpu().numpy(),
        "hcp_pair": ((own + partner) @ weights).cpu().numpy(),
        "longest": lengths.argmax(-1).cpu().numpy(),  # first max: S before H...
    }


# ---------------------------------------------------------------------------
# Baselines


@torch.no_grad()
def best_fixed_actions(train: TorchDeals, scorer: TorchScorer) -> dict:
    """Best unconditional pair action per vulnerability on the training range."""
    rows = grid_rows(train.n, train.hands.device)
    flat, _, _ = exact_rows(train, rows, scorer)
    vul = rows[2]
    actions, means = [], []
    for v in (0, 1):
        mean = flat[vul == v].mean(0)
        actions.append(int(mean.argmax()))
        means.append(float(mean.max()))
    return {"actions": actions, "names": [pair_action_name(a) for a in actions],
            "train_mean_score": means}


@torch.no_grad()
def trick_prior(train: TorchDeals) -> torch.Tensor:
    """Unconditional ``P(tricks | strain)`` ``(5,14)`` from the training range."""
    counts = torch.stack([
        torch.bincount(train.tricks[:, :, s].reshape(-1), minlength=14)
        for s in range(5)]).float()
    return (counts + 1.0) / (counts + 1.0).sum(-1, keepdim=True)


def contract_index(level: np.ndarray, bid_strain: np.ndarray) -> np.ndarray:
    return (level - 1) * 5 + bid_strain


@torch.no_grad()
def heuristic_choice(deals: TorchDeals, rows, mode: str) -> np.ndarray:
    """Crude HCP/longest-suit rule. Diagnostic only, not a bidding teacher."""
    deal, seat, _ = rows
    weights = HCP_WEIGHTS.to(deals.hands.device)
    own = deals.hands[deal, seat % 4]
    own_len = own.view(-1, 4, 13).sum(-1).cpu().numpy()
    own_hcp = (own @ weights).cpu().numpy()
    if mode == "single":
        balanced = (own_len.min(1) >= 2) & ((own_len == 2).sum(1) <= 1)
        suit_contract = contract_index(np.ones(len(own_hcp), dtype=np.int64),
                                       3 - own_len.argmax(1))
        choice = np.where(balanced, 4, suit_contract)  # 4 = 1NT
        return np.where(own_hcp < 12, PAIR_PASS, choice)

    partner = deals.hands[deal, (seat + 2) % 4]
    part_len = partner.view(-1, 4, 13).sum(-1).cpu().numpy()
    part_hcp = (partner @ weights).cpu().numpy()
    hcp = own_hcp + part_hcp
    lengths = own_len + part_len
    n = len(hcp)
    rows_idx = np.arange(n)
    best_major = lengths[:, :2].argmax(1)
    has_major = lengths[rows_idx, best_major] >= 8
    best_suit = lengths.argmax(1)
    has_fit = lengths[rows_idx, best_suit] >= 8
    fit_suit = np.where(has_major, best_major, best_suit)
    level = np.select([hcp >= 37, hcp >= 33, hcp >= 25], [7, 6, 0], default=2)
    nt = np.select([hcp >= 33, hcp >= 25], [~has_major, ~has_major], default=~has_fit)
    level = np.where(level == 0, np.where(nt, 3, 4), level)
    level = np.where((hcp < 25) & nt, 1, level)
    bid_strain = np.where(nt, 4, 3 - fit_suit)
    contract = contract_index(level, bid_strain)
    suit_declarer = (part_len[rows_idx, fit_suit] > own_len[rows_idx, fit_suit]).astype(np.int64)
    nt_declarer = (part_hcp > own_hcp).astype(np.int64)
    declarer = np.where(nt, nt_declarer, suit_declarer)
    return np.where(hcp < 18, PAIR_PASS, declarer * N_CONTRACTS + contract)


# ---------------------------------------------------------------------------
# Metrics


def paired_bootstrap(diff: np.ndarray, deal: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Mean paired difference with a deal-clustered 95% bootstrap interval."""
    n_deals = int(deal.max()) + 1
    per_deal = np.bincount(deal, weights=diff, minlength=n_deals) / np.bincount(
        deal, minlength=n_deals)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = per_deal[rng.integers(0, n_deals, n_deals)].mean()
    return {"mean": float(per_deal.mean()), "ci95": [float(np.percentile(means, 2.5)),
                                                     float(np.percentile(means, 97.5))]}


def selection_metrics(flat: np.ndarray, ceiling: np.ndarray, choice: np.ndarray,
                      rel: np.ndarray, strength: np.ndarray, longest: np.ndarray,
                      buckets) -> dict:
    n = len(choice)
    idx = np.arange(n)
    score = flat[idx, choice]
    regret = ceiling - score
    passed = choice == PAIR_PASS
    declarer = np.minimum(choice // N_CONTRACTS, 1)
    contract = choice % N_CONTRACTS
    level = np.where(passed, 0, CONTRACT_LEVEL[contract])
    table_strain = CONTRACT_TABLE_STRAIN[contract]
    margin = rel[idx, declarer, table_strain] - (level + 6)
    bids = ~passed
    counts = np.bincount(choice, minlength=PAIR_PASS + 1) / n
    nz = counts[counts > 0]
    suit_bids = bids & (table_strain < 4)
    corr = (float(np.corrcoef(level, strength)[0, 1])
            if level.std() > 0 and strength.std() > 0 else float("nan"))
    top = np.argsort(-counts)[:5]
    out = {
        "mean_score": float(score.mean()),
        "mean_ceiling": float(ceiling.mean()),
        "mean_regret": float(regret.mean()),
        "mean_imp_regret": float(imps_array(regret).mean()),
        "within_10": float((regret <= 10).mean()),
        "within_50": float((regret <= 50).mean()),
        "within_100": float((regret <= 100).mean()),
        "pass_rate": float(passed.mean()),
        "make_rate": float((margin[bids] >= 0).mean()) if bids.any() else float("nan"),
        "mean_trick_margin": float(margin[bids].mean()) if bids.any() else float("nan"),
        "level_share": {str(l): float((level == l).mean()) for l in range(8)},
        "strain_share": {s: float((CONTRACT_BID_STRAIN[contract[bids]] == i).mean())
                         if bids.any() else 0.0
                         for i, s in enumerate(("C", "D", "H", "S", "NT"))},
        "partner_declarer_share": float((declarer[bids] == 1).mean()) if bids.any() else 0.0,
        "distinct_choices": int((counts > 0).sum()),
        "choice_entropy_bits": float(-(nz * np.log2(nz)).sum()),
        "top_choices": {pair_action_name(int(a)): float(counts[a]) for a in top if counts[a] > 0},
        "level_strength_corr": corr,
        "suit_contract_in_longest_suit": float((table_strain[suit_bids] == longest[suit_bids]).mean())
        if suit_bids.any() else float("nan"),
        "by_strength": {},
    }
    for lo, hi in buckets:
        m = (strength >= lo) & (strength <= hi)
        if m.any():
            out["by_strength"][f"{lo}-{hi}"] = {
                "share": float(m.mean()), "pass_rate": float(passed[m].mean()),
                "mean_level": float(level[m].mean()), "mean_score": float(score[m].mean()),
                "mean_regret": float(regret[m].mean())}
    return out


def calibration(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(pred, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            gap = abs(pred[m].mean() - actual[m].mean())
            ece += m.mean() * gap
            table.append([float(edges[b]), float(m.mean()), float(pred[m].mean()),
                          float(actual[m].mean())])
    return {"ece": float(ece), "bins[lo,share,pred,actual]": table}


@torch.no_grad()
def trick_quality(probs, rel, flat, vul, scorer: TorchScorer, prior: torch.Tensor) -> dict:
    p_true = probs.gather(-1, rel.unsqueeze(-1)).squeeze(-1)
    one_hot = torch.nn.functional.one_hot(rel, 14).float()
    support = torch.arange(14, dtype=torch.float32, device=probs.device)
    prior_true = prior.to(probs.device)[torch.arange(5, device=probs.device), rel]
    exact = flat[:, :PAIR_PASS].view(-1, 2, N_CONTRACTS)
    expected = scorer.expected(probs, vul)
    make_p = scorer.make_probability(probs)
    made = (exact > 0).float()
    level = torch.as_tensor(CONTRACT_LEVEL, device=probs.device)
    by_level = {}
    for l in range(1, 8):
        m = level == l
        by_level[str(l)] = {"pred_make": float(make_p[:, :, m].mean()),
                            "actual_make": float(made[:, :, m].mean())}
    return {
        "trick_nll": float(-p_true.clamp(min=1e-12).log().mean()),
        "prior_trick_nll": float(-prior_true.log().mean()),
        "trick_brier": float(((probs - one_hot) ** 2).sum(-1).mean()),
        "expected_trick_mae": float(((probs * support).sum(-1) - rel).abs().mean()),
        "prior_expected_trick_mae": float(((prior.to(probs.device) * support).sum(-1)[
            torch.arange(5, device=probs.device)] - rel).abs().mean()),
        "score_mae_points": float((expected - exact).abs().mean()),
        "make_calibration": calibration(make_p.flatten().cpu().numpy(),
                                        made.flatten().cpu().numpy()),
        "make_rate_by_level": by_level,
    }


@torch.no_grad()
def quick_metrics(net: ContractNet, deals: TorchDeals, scorer: TorchScorer) -> dict:
    """Cheap validation metrics used during training."""
    rows = grid_rows(deals.n, deals.hands.device)
    flat, ceiling, rel = exact_rows(deals, rows, scorer)
    probs, q = predict(net, deals, rows)
    values = scorer.flat(scorer.expected(probs, rows[2]))
    target = (flat - ceiling[:, None]) / 100.0
    out = {
        "trick_nll": float(-probs.gather(-1, rel.unsqueeze(-1)).clamp(min=1e-12).log().mean()),
        "q_mae_points": float((q - target).abs().mean() * 100.0),
    }
    for name, choice in (("expected", choose(values)), ("q", choose(q))):
        score = flat.gather(1, choice[:, None]).squeeze(1)
        out[f"{name}_score"] = float(score.mean())
        out[f"{name}_regret"] = float((ceiling - score).mean())
        out[f"{name}_pass_rate"] = float((choice == PAIR_PASS).float().mean())
    return out


@torch.no_grad()
def evaluate_model(net: ContractNet, eval_deals: TorchDeals, train_deals: TorchDeals,
                   scorer: TorchScorer, seed: int = 0) -> tuple[dict, dict]:
    mode = net.inputs
    rows = grid_rows(eval_deals.n, eval_deals.hands.device)
    flat_t, ceiling_t, rel_t = exact_rows(eval_deals, rows, scorer)
    probs, q = predict(net, eval_deals, rows)
    vul_t = rows[2]
    values = scorer.flat(scorer.expected(probs, vul_t))
    fixed = best_fixed_actions(train_deals, scorer)
    prior = trick_prior(train_deals)

    flat = flat_t.cpu().numpy()
    ceiling = ceiling_t.cpu().numpy()
    rel = rel_t.cpu().numpy()
    vul = vul_t.cpu().numpy()
    deal = rows[0].cpu().numpy()
    feats = hand_features(eval_deals, rows, mode)
    strength = feats["hcp_self"] if mode == "single" else feats["hcp_pair"]
    buckets = SINGLE_BUCKETS if mode == "single" else PAIR_BUCKETS

    choices = {
        "model_expected": choose(values).cpu().numpy(),
        "model_q": choose(q).cpu().numpy(),
        "model_expected_self_declarer": choose(values, True).cpu().numpy(),
        "always_pass": np.full(len(deal), PAIR_PASS),
        "best_fixed": np.asarray(fixed["actions"])[vul],
        "heuristic": heuristic_choice(eval_deals, rows, mode),
        "dd_ceiling": flat.argmax(1),
    }
    assert np.array_equal(flat[np.arange(len(deal)), choices["dd_ceiling"]], ceiling)
    scores = {k: flat[np.arange(len(deal)), c] for k, c in choices.items()}

    selected_make = scorer.make_probability(probs).view(len(deal), -1).cpu().numpy()
    chosen = choices["model_expected"]
    bid = chosen != PAIR_PASS
    chosen_level = np.where(bid, CONTRACT_LEVEL[chosen % N_CONTRACTS], 0)
    chosen_margin = rel[np.arange(len(deal)), np.minimum(chosen // N_CONTRACTS, 1),
                        CONTRACT_TABLE_STRAIN[chosen % N_CONTRACTS]] - (chosen_level + 6)
    selected_calibration = {}
    for l in range(1, 8):
        m = bid & (chosen_level == l)
        if m.any():
            selected_calibration[str(l)] = {
                "share": float(m.mean()),
                "pred_make": float(selected_make[m, chosen[m]].mean()),
                "actual_make": float((chosen_margin[m] >= 0).mean())}

    report = {
        "inputs": mode,
        "eval_first_index": eval_deals.first_index,
        "eval_deals": eval_deals.n,
        "rows": int(len(deal)),
        "train_first_index": train_deals.first_index,
        "train_deals": train_deals.n,
        "best_fixed": fixed,
        "trick_quality": trick_quality(probs, rel_t, flat_t, vul_t, scorer, prior),
        "q_quality": {
            "q_mae_points": float((q - (flat_t - ceiling_t[:, None]) / 100.0).abs().mean() * 100),
            "q_vs_expected_argmax_agreement": float(
                (choices["model_q"] == choices["model_expected"]).mean()),
        },
        "selected_contract_calibration": selected_calibration,
        "policies": {k: selection_metrics(flat, ceiling, c, rel, strength, feats["longest"],
                                          buckets) for k, c in choices.items()},
        "paired_score_lift": {
            f"model_expected_minus_{b}": paired_bootstrap(
                scores["model_expected"] - scores[b], deal, seed=seed)
            for b in ("always_pass", "best_fixed", "heuristic", "model_q")},
    }
    arrays = {
        "deal_index": deal + eval_deals.first_index, "seat": rows[1].cpu().numpy(),
        "vulnerable": vul, "ceiling": ceiling,
        **{f"choice_{k}": v for k, v in choices.items()},
        **{f"score_{k}": v for k, v in scores.items()},
    }
    return report, arrays


def summary_lines(report: dict) -> list[str]:
    tq = report["trick_quality"]
    lines = [
        f"inputs={report['inputs']} eval_rows={report['rows']:,}",
        f"trick NLL {tq['trick_nll']:.4f} (prior {tq['prior_trick_nll']:.4f})  "
        f"E[tricks] MAE {tq['expected_trick_mae']:.3f} (prior {tq['prior_expected_trick_mae']:.3f})  "
        f"make ECE {tq['make_calibration']['ece']:.4f}",
        f"{'policy':<30}{'score':>8}{'regret':>8}{'IMPreg':>8}{'<=50':>7}"
        f"{'pass':>7}{'make':>7}{'lvl~hcp':>8}{'distinct':>9}",
    ]
    for name, m in report["policies"].items():
        lines.append(
            f"{name:<30}{m['mean_score']:8.1f}{m['mean_regret']:8.1f}"
            f"{m['mean_imp_regret']:8.2f}{m['within_50']:7.3f}{m['pass_rate']:7.3f}"
            f"{m['make_rate']:7.3f}{m['level_strength_corr']:8.3f}{m['distinct_choices']:9d}")
    for name, lift in report["paired_score_lift"].items():
        lines.append(f"{name}: {lift['mean']:+.1f} pts  CI95 "
                     f"[{lift['ci95'][0]:+.1f}, {lift['ci95'][1]:+.1f}]")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Stage B/C contract finder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="")
    parser.add_argument("--out", required=True, help="report JSON path; rows go next to it")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    net, ckpt = load_checkpoint(args.checkpoint, args.device)
    run = ckpt["args"]
    data = args.data or run["data"]
    train = load_range(data, run["train_start"], run["train_count"], args.device)
    held = load_range(data, run["eval_start"], run["eval_count"], args.device)
    report, arrays = evaluate_model(net, held, train, TorchScorer(args.device), run["seed"])
    report["checkpoint"] = args.checkpoint
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out.with_suffix(".rows.npz"), **arrays)
    print("\n".join(summary_lines(report)))


if __name__ == "__main__":
    main()
