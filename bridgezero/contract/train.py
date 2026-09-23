"""Stage B (partnership hands) and Stage C (single hand) contract-finder trainer.

Example::

    OMP_NUM_THREADS=4 python -u -m bridgezero.contract.train \\
        --stage C --data data/deals.npz \\
        --train-count 16000 --out runs/E3_stageC_16k
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import STAGE_INPUTS, TorchDeals, dataset_size, load_range, ranges_overlap, resolve_range
from .evaluate import evaluate_model, quick_metrics, summary_lines
from .model import ContractNet, load_checkpoint, save_checkpoint
from .targets import TARGET_SCALE, TorchScorer


def batch_losses(net: ContractNet, deals: TorchDeals, idx: torch.Tensor, seat: torch.Tensor,
                 vul: torch.Tensor, scorer: TorchScorer) -> dict[str, torch.Tensor]:
    rel = deals.rel_tricks(idx, seat)
    out = net(deals.inputs(idx, seat, net.inputs), vul)
    with torch.no_grad():
        exact = scorer.exact(rel, vul)
        q_target = (scorer.flat(exact) - scorer.ceiling(exact)[:, None]) / TARGET_SCALE
    return {
        "trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "q_loss": F.huber_loss(out["contract_q"], q_target),
    }


def gradient_norms(net: ContractNet, losses: dict[str, torch.Tensor]) -> dict[str, float]:
    """Unweighted per-loss gradient norm on the shared encoder/trunk."""
    shared = [parameter for parameter in net.shared_parameters() if parameter.requires_grad]
    norms = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        norms[name] = float(torch.sqrt(sum((g ** 2).sum() for g in grads if g is not None)))
    return norms


def learning_rate(step: int, args: argparse.Namespace, start: int = 0) -> float:
    """Linear warmup then cosine to ``min_lr_frac``; a resumed run restarts at ``start``."""
    warm = min(1.0, (step - start) / max(args.warmup, 1))
    progress = min(1.0, (step - start) / max(args.steps - start, 1))
    return args.lr * warm * (args.min_lr_frac + (1 - args.min_lr_frac)
                             * 0.5 * (1 + math.cos(math.pi * progress)))


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = dataset_size(args.data)
    spans = {name: resolve_range(total, getattr(args, f"{name}_start"),
                                 getattr(args, f"{name}_count"))
             for name in ("train", "val", "eval")}
    for a in spans:
        for b in spans:
            if a < b and ranges_overlap(spans[a], spans[b]):
                raise ValueError(f"{a} range {spans[a]} overlaps {b} range {spans[b]}")
    train = load_range(args.data, args.train_start, args.train_count, device)
    val = load_range(args.data, args.val_start, args.val_count, device)
    held = load_range(args.data, args.eval_start, args.eval_count, device)
    scorer = TorchScorer(device)
    inputs = STAGE_INPUTS[args.stage]
    net = ContractNet(inputs, args.width, args.suit_width, args.depth).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    run_info = {**vars(args), "inputs": inputs, "dataset_size": total,
                "absolute_ranges": {k: list(v) for k, v in spans.items()},
                "parameters": sum(p.numel() for p in net.parameters())}
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    print(f"stage {args.stage} inputs={inputs} params={run_info['parameters']:,} "
          f"ranges={run_info['absolute_ranges']}", flush=True)

    fit = train.head(args.val_count)  # same size as val, for overfit comparison

    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    best_score = -math.inf
    start = time.time()
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args)
        idx = torch.randint(train.n, (args.batch,), generator=generator).to(device)
        seat = torch.randint(4, (args.batch,), generator=generator).to(device)
        vul = torch.randint(2, (args.batch,), generator=generator).to(device)
        net.train()
        losses = batch_losses(net, train, idx, seat, vul, scorer)
        loss = args.trick_weight * losses["trick_nll"] + args.q_weight * losses["q_loss"]
        logging = step == 1 or step % args.eval_every == 0 or step == args.steps
        norms = gradient_norms(net, losses) if logging else {}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
        optimizer.step()

        if logging:
            record = {
                "step": step, "seconds": round(time.time() - start, 1),
                "lr": optimizer.param_groups[0]["lr"], "loss": float(loss.detach()),
                **{k: float(v.detach()) for k, v in losses.items()},
                "grad_norm": total_norm, "grad_clipped": total_norm > args.grad_clip,
                "unweighted_shared_grad_norm": norms,
                "val": quick_metrics(net, val, scorer),
                "train_fit": quick_metrics(net, fit, scorer),
            }
            with log_path.open("a") as fh:
                fh.write(json.dumps(record) + "\n")
            v, t = record["val"], record["train_fit"]
            print(f"step {step:>6} {record['seconds']:>7.1f}s loss {record['loss']:.4f} "
                  f"nll tr/val {t['trick_nll']:.4f}/{v['trick_nll']:.4f} "
                  f"q {record['q_loss']:.4f} val score E/Q {v['expected_score']:+.1f}/"
                  f"{v['q_score']:+.1f} regret {v['expected_regret']:.1f} "
                  f"pass {v['expected_pass_rate']:.3f} gn {norms}", flush=True)
            if v["expected_score"] > best_score:
                best_score = v["expected_score"]
                save_checkpoint(out_dir / "best.pt", net, args.stage, step=step,
                                args=vars(args), val=v)
    save_checkpoint(out_dir / "last.pt", net, args.stage, step=args.steps,
                    args=vars(args))

    best, meta = load_checkpoint(out_dir / "best.pt", device, args.stage)
    report, arrays = evaluate_model(best, held, train, scorer, args.seed)
    report["checkpoint"] = str(out_dir / "best.pt")
    report["checkpoint_step"] = meta["step"]
    report["train_fit"] = quick_metrics(best, fit, scorer)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out_dir / "eval_rows.npz", **arrays)
    print(f"\nHELD-OUT EVAL (best val step {meta['step']})")
    print("\n".join(summary_lines(report)), flush=True)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=sorted(STAGE_INPUTS), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=16000)
    parser.add_argument("--val-start", type=int, default=16000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--eval-start", type=int, default=-10000,
                        help="negative counts from the end of the dataset")
    parser.add_argument("--eval-count", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--trick-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--suit-width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
