"""Stage D evaluation: full greedy silent-opponent auctions on fixed held-out rows.

Rows are every eval deal x active side x dealer x vulnerability (16 per deal).
The frozen Stage-C baseline row for an auction row is the E3 choice of the first
active player (the opener) on the same deal, seat, and vulnerability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import TorchDeals, load_range
from .evaluate import PAIR_BUCKETS, HCP_WEIGHTS, imps_array, paired_bootstrap
from .model import load_checkpoint
from .prefixes import (
    OBSERVATION_NOTES,
    CoopBatch,
    decision_values,
    exact_endpoint,
    final_scores,
    mask_partner,
    observe,
    sample_prefixes,
)
from .targets import TorchScorer
from bridgezero.bridge.calls import STRAIN_PERM

RULES = ("expected", "q", "policy")


@torch.no_grad()
def trajectory_belief_metrics(net, deals: TorchDeals, states: list[CoopBatch],
                              observation: str = "intact", chunk: int = 32768) -> dict:
    """Partner-card belief along on-policy states, intact and history-masked."""
    if not states:
        return {}
    totals: dict[str, dict[int, dict[str, float]]] = {"intact": {}, "partner_masked": {}}
    for state in states:
        for start in range(0, len(state), chunk):
            rows = torch.arange(start, min(start + chunk, len(state)), device=state.deal.device)
            sub = state.subset(rows)
            hand = deals.hands[sub.deal, sub.actor_seat]
            partner = deals.hands[sub.deal, (sub.actor_seat + 2) % 4]
            base_features = observe(sub, observation)
            for regime, features in (("intact", base_features),
                                     ("partner_masked", mask_partner(sub, base_features))):
                out = net(hand, features)
                if "partner_belief_logits" not in out:
                    return {}
                unknown = 1.0 - hand
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    out["partner_belief_logits"], partner, reduction="none")
                top = out["partner_belief_logits"].masked_fill(
                    hand.bool(), -torch.inf).topk(13).indices
                depth = int(sub.k[0])
                if not bool((sub.k == depth).all()):
                    raise AssertionError("recorded auction state mixes decision depths")
                acc = totals[regime].setdefault(
                    depth, {"rows": 0, "bce_sum": 0.0, "top13_sum": 0.0,
                            "count_sum": 0.0})
                acc["rows"] += len(sub)
                acc["bce_sum"] += float((loss * unknown).sum() / 39.0)
                acc["top13_sum"] += float(partner.gather(1, top).sum() / 13.0)
                acc["count_sum"] += float(out["partner_probability"].sum())
    result = {}
    for regime, depths in totals.items():
        result[regime] = {
            str(depth): {"rows": int(acc["rows"]),
                         "bce": acc["bce_sum"] / acc["rows"],
                         "top13_recall": acc["top13_sum"] / acc["rows"],
                         "expected_count": acc["count_sum"] / acc["rows"]}
            for depth, acc in sorted(depths.items())
        }
    return result


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


def auction_metrics(batch: CoopBatch, score: np.ndarray, ceiling: np.ndarray,
                    declarer: np.ndarray, deals: TorchDeals) -> dict:
    last = batch.last.cpu().numpy()
    k = batch.k.cpu().numpy()
    history = batch.history.cpu().numpy()
    bid = last >= 0
    level = np.where(bid, last // 5 + 1, 0)
    regret = ceiling - score
    tricks = deals.rel_tricks(batch.deal, batch.a0).cpu().numpy()
    table_strain = np.asarray(STRAIN_PERM)[np.maximum(last, 0) % 5]
    margin = tricks[np.arange(len(last)), declarer, table_strain] - (level + 6)
    hands = deals.hands
    weights = HCP_WEIGHTS.to(hands.device)
    a0 = batch.a0
    hcp_open = (hands[batch.deal, a0] @ weights).cpu().numpy()
    hcp_pair = hcp_open + (hands[batch.deal, (a0 + 2) % 4] @ weights).cpu().numpy()

    def entropy_bits(values):
        counts = np.bincount(values, minlength=36) / max(len(values), 1)
        nz = counts[counts > 0]
        return float(-(nz * np.log2(nz)).sum())

    opening = history[:, 0]
    responded = (opening < 35) & (history[:, 1] >= 0)
    response_entropy = [entropy_bits(history[responded & (opening == o), 1])
                        * float((responded & (opening == o)).sum()) for o in range(35)]
    opening_level = np.where(opening < 35, opening // 5 + 1, 0)
    out = {
        "mean_score": float(score.mean()),
        "mean_ceiling": float(ceiling.mean()),
        "mean_regret": float(regret.mean()),
        "mean_imp_regret": float(imps_array(regret).mean()),
        "within_50": float((regret <= 50).mean()),
        "within_100": float((regret <= 100).mean()),
        "passout_rate": float((~bid).mean()),
        "make_rate": float((margin[bid] >= 0).mean()) if bid.any() else float("nan"),
        "mean_trick_margin": float(margin[bid].mean()) if bid.any() else float("nan"),
        "active_decisions": float(k.mean()),
        "active_decisions_max": int(k.max()),
        "partner_of_opener_declares": float((declarer[bid] == 1).mean()) if bid.any() else 0.0,
        "level_share": {str(l): float((level == l).mean()) for l in range(8)},
        "strain_share": {s: float((last[bid] % 5 == i).mean()) if bid.any() else 0.0
                         for i, s in enumerate(("C", "D", "H", "S", "NT"))},
        "opening_pass_rate": float((opening == 35).mean()),
        "opening_entropy_bits": entropy_bits(opening),
        "distinct_openings": int(len(np.unique(opening))),
        "response_entropy_bits": float(sum(response_entropy) / max(responded.sum(), 1)),
        "single_bid_auction_share": float(((k == 2) & bid).mean()),
        "opening_level_vs_opener_hcp_corr": float(np.corrcoef(opening_level, hcp_open)[0, 1])
        if opening_level.std() > 0 else float("nan"),
        "final_level_vs_pair_hcp_corr": float(np.corrcoef(level, hcp_pair)[0, 1])
        if level.std() > 0 else float("nan"),
        "by_pair_hcp": {},
    }
    for lo, hi in PAIR_BUCKETS:
        m = (hcp_pair >= lo) & (hcp_pair <= hi)
        if m.any():
            out["by_pair_hcp"][f"{lo}-{hi}"] = {
                "share": float(m.mean()), "passout": float((~bid[m]).mean()),
                "mean_level": float(level[m].mean()), "mean_regret": float(regret[m].mean())}
    return out


@torch.no_grad()
def prefix_metrics(net, deals: TorchDeals, scorer: TorchScorer, n: int, seed: int,
                   observation: str = "intact") -> dict:
    """Endpoint Q quality on uniformly generated prefixes (policy-independent)."""
    gen = torch.Generator().manual_seed(seed)
    batch, info = sample_prefixes(deals, n, gen, device=deals.hands.device)
    values, ceiling, rel = exact_endpoint(batch, deals, scorer)
    legal = batch.legal()
    target = (values - ceiling[:, None]) / 100.0
    feats = observe(batch, observation)
    was_training = net.training
    net.eval()
    out = {}
    hand = deals.hands[batch.deal, batch.actor_seat]
    outputs = net(hand, feats)
    probs = torch.softmax(outputs["trick_logits"], -1)
    out["trick_nll"] = float(-probs.gather(-1, rel.unsqueeze(-1)).clamp(min=1e-12).log().mean())
    q = decision_values(net, deals, batch, scorer, "q", feats)
    error = (q - target) * 100
    out["q_mae_points"] = float(error.abs()[legal].mean())
    out["q_rmse_points"] = float(error[legal].pow(2).mean().sqrt())
    out["q_pass_bias_points"] = float(error[:, -1][legal[:, -1]].mean())
    out["q_bid_bias_points"] = float(error[:, :-1][legal[:, :-1]].mean())
    if "partner_belief_logits" in outputs:
        partner = deals.hands[batch.deal, (batch.actor_seat + 2) % 4]
        unknown = 1.0 - hand
        per_card = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["partner_belief_logits"], partner, reduction="none")
        out["belief_bce"] = float((per_card * unknown).sum() / unknown.sum())
        probability = outputs["partner_probability"]
        out["belief_expected_count"] = float(probability.sum(-1).mean())
        top = outputs["partner_belief_logits"].masked_fill(hand.bool(), -torch.inf).topk(13).indices
        out["belief_top13_recall"] = float(partner.gather(1, top).sum(-1).float().mean() / 13)
    depth = info["depth"]
    for rule in RULES:
        v = decision_values(net, deals, batch, scorer, rule, feats)
        choice = v.masked_fill(~legal, -torch.inf).argmax(-1)
        regret = -target.gather(1, choice[:, None]).squeeze(1) * 100
        out[f"{rule}_endpoint_regret"] = float(regret.mean())
        out[f"{rule}_endpoint_regret_by_depth"] = {
            str(d): float(regret[depth == d].mean()) for d in range(int(depth.max()) + 1)
            if bool((depth == d).any())}
        out[f"{rule}_pass_rate"] = float((choice == 35).float().mean())
    net.train(was_training)
    return out


def align_frozen_rows(base, name: str, deal: np.ndarray, seat: np.ndarray, vul: np.ndarray,
                      first_index: int, ceiling: np.ndarray | None = None) -> np.ndarray:
    """Index of Stage B/C rows (ordered deal*8 + seat*2 + vul) for (deal, seat, vul).

    Raises unless the frozen rows hold the same absolute deals, seats,
    vulnerability, and (optionally) cooperative ceilings.
    """
    index = deal * 8 + seat * 2 + vul
    if len(index) and index.max() >= len(base["deal_index"]):
        raise ValueError(f"{name} rows do not cover the eval deals")
    for key, expect in (("deal_index", deal + first_index), ("seat", seat), ("vulnerable", vul)):
        if not np.array_equal(base[key][index], expect):
            raise ValueError(f"{name} rows do not align on {key}")
    if ceiling is not None and not np.array_equal(base["ceiling"][index], ceiling):
        raise ValueError(f"{name} ceilings differ from auction ceilings")
    return index


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen(run_dir: Path) -> dict:
    manifest = run_dir / "FROZEN.sha256"
    expected = dict(reversed(line.split()) for line in manifest.read_text().splitlines())
    for name, digest in expected.items():
        if sha256(run_dir / name) != digest:
            raise ValueError(f"frozen baseline file changed: {run_dir / name}")
    return expected


@torch.no_grad()
def evaluate_auctions(net, eval_deals: TorchDeals, scorer: TorchScorer,
                      baselines: dict[str, Path], seed: int = 0,
                      prefix_rows: int = 32000, observation: str = "intact",
                      stage: str = "D1") -> tuple[dict, dict]:
    rows = auction_rows(eval_deals.n, eval_deals.hands.device)
    deal_np = rows[0].cpu().numpy()
    report: dict = {"eval_first_index": eval_deals.first_index, "eval_deals": eval_deals.n,
                    "rows": int(len(deal_np)), "stage": stage, "observation": observation,
                    "observation_note": OBSERVATION_NOTES[observation],
                    "rules": {}, "paired_score_lift": {}, "baselines": {}}
    arrays: dict = {"deal_index": deal_np + eval_deals.first_index,
                    "side": rows[1].cpu().numpy(), "dealer": rows[2].cpu().numpy(),
                    "vulnerable": rows[3].cpu().numpy()}
    opener = None
    # Every belief architecture (full and residual) exposes ``belief_head``.
    belief = hasattr(net, "belief_head")
    states: dict[str, list[CoopBatch]] = {"q": [], "policy": []}
    for rule in RULES:
        record = states[rule] if belief and rule in states else None
        batch = continue_auctions(net, eval_deals, CoopBatch.start(*rows), scorer, rule,
                                  observation=observation, record=record)
        score_t, ceiling_t, declarer_t = final_scores(batch, eval_deals, scorer)
        score = score_t.cpu().numpy()
        ceiling = ceiling_t.cpu().numpy()
        report["rules"][rule] = auction_metrics(batch, score, ceiling,
                                                declarer_t.cpu().numpy(), eval_deals)
        arrays[f"score_{rule}"] = score
        arrays[f"contract_{rule}"] = batch.last.cpu().numpy()
        arrays[f"decisions_{rule}"] = batch.k.cpu().numpy()
        arrays["ceiling"] = ceiling
        opener = batch.a0.cpu().numpy()
    arrays["opener_seat"] = opener
    if states["q"]:
        report["belief_trajectory"] = trajectory_belief_metrics(
            net, eval_deals, states["q"], observation)
        report["belief_trajectory_policy"] = trajectory_belief_metrics(
            net, eval_deals, states["policy"], observation)

    for name, run_dir in baselines.items():
        frozen = verify_frozen(run_dir)
        base = np.load(run_dir / "eval_rows.npz")
        index = align_frozen_rows(base, name, deal_np, opener, arrays["vulnerable"],
                                  eval_deals.first_index, arrays["ceiling"])
        for variant in ("model_expected", "model_expected_self_declarer"):
            baseline_score = base[f"score_{variant}"][index]
            report["baselines"][f"{name}:{variant}"] = {
                "mean_score": float(baseline_score.mean()),
                "mean_regret": float((arrays["ceiling"] - baseline_score).mean()),
                "frozen_sha256_eval_rows": frozen["eval_rows.npz"]}
            for rule in RULES:
                report["paired_score_lift"][f"{rule}_minus_{name}:{variant}"] = paired_bootstrap(
                    arrays[f"score_{rule}"] - baseline_score, deal_np, seed=seed)
    report["prefix_endpoint"] = prefix_metrics(net, eval_deals, scorer, prefix_rows, seed + 99,
                                               observation)
    return report, arrays


def summary_lines(report: dict) -> list[str]:
    # Reports written before the stage was recorded carry no label.
    stage = report.get("stage", "D?")
    lines = [f"auction rows={report['rows']:,}",
             f"{'rule/baseline':<42}{'score':>8}{'regret':>8}{'IMPreg':>8}{'passout':>8}"
             f"{'make':>7}{'decis':>7}{'open-H':>7}{'resp-H':>7}"]
    for rule, m in report["rules"].items():
        lines.append(f"{stage + ' ' + rule:<42}{m['mean_score']:8.1f}{m['mean_regret']:8.1f}"
                     f"{m['mean_imp_regret']:8.2f}{m['passout_rate']:8.3f}{m['make_rate']:7.3f}"
                     f"{m['active_decisions']:7.2f}{m['opening_entropy_bits']:7.2f}"
                     f"{m['response_entropy_bits']:7.2f}")
    for name, m in report["baselines"].items():
        lines.append(f"{name:<42}{m['mean_score']:8.1f}{m['mean_regret']:8.1f}")
    for name, lift in report["paired_score_lift"].items():
        lines.append(f"{name}: {lift['mean']:+.1f} pts CI95 "
                     f"[{lift['ci95'][0]:+.1f}, {lift['ci95'][1]:+.1f}]")
    p = report["prefix_endpoint"]
    lines.append(f"prefix endpoint: trick NLL {p['trick_nll']:.4f} Q MAE {p['q_mae_points']:.1f} "
                 f"regret E/Q/P {p['expected_endpoint_regret']:.1f}/"
                 f"{p['q_endpoint_regret']:.1f}/{p['policy_endpoint_regret']:.1f}")
    if "belief_bce" in p:
        lines.append(f"prefix partner belief: BCE {p['belief_bce']:.4f} top13 "
                     f"{p['belief_top13_recall']:.3f} count {p['belief_expected_count']:.2f}")
    for regime, depths in report.get("belief_trajectory", {}).items():
        values = " ".join(
            f"d{depth}:{metric['top13_recall']:.3f}"
            for depth, metric in depths.items())
        lines.append(f"on-policy belief top13 {regime}: {values}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Stage D auction checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", action="append", default=[],
                        help="NAME=RUN_DIR of a frozen Stage B/C run")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    net, ckpt = load_checkpoint(args.checkpoint, args.device)
    run = ckpt["args"]
    held = load_range(run["data"], run["eval_start"], run["eval_count"], args.device)
    baselines = {k: Path(v) for k, v in (b.split("=", 1) for b in args.baseline)}
    # A checkpoint is always evaluated in the observation regime it was trained in.
    report, arrays = evaluate_auctions(net, held, TorchScorer(args.device), baselines, run["seed"],
                                       observation=ckpt["observation"], stage=ckpt["stage"])
    report["checkpoint"] = args.checkpoint
    report["checkpoint_step"] = ckpt.get("step")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    np.savez_compressed(out.with_suffix(".rows.npz"), **arrays)
    print("\n".join(summary_lines(report)))


if __name__ == "__main__":
    main()
