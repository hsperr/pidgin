"""Stage 1: grounding from random weights.

The net regresses the double-dummy trick distribution and score of every legal
endpoint on silent-opponent auction prefixes, and its policy head is pulled
toward the best endpoint. Nothing here plays against anyone.

Every ``--eval-every`` steps the greedy policy bids the validation deals with
silent opponents; that mean score (the "contract score") picks ``best.pt``.
Training stops when it has not improved for ``--patience`` steps, or after one
pass over the training pool.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .contract.data import block_starts, dataset_size, load_range, ranges_overlap, resolve_range, validate_training_pool
from .contract.evaluate_auction import auction_rows, run_auctions
from .contract.model import AuctionContractNet, CentralCritic, save_checkpoint
from .contract.prefixes import (exact_endpoint, expected_endpoint, final_scores, net_outputs,
                                observe, sample_prefixes)
from .contract.targets import TARGET_SCALE, TorchScorer

STAGE = "D4PG"      # checkpoint label the four-seat warm start reads


def repository_state() -> dict:
    """Commit identity and dirty flag of the code that runs."""
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(("git", "rev-parse", "HEAD"), cwd=root, check=True,
                                  capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(("git", "status", "--porcelain"), cwd=root, check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def grounding_losses(net, target_net, deals, batch, scorer, policy_temperature: float) -> dict:
    """Trick NLL and Huber Q on every legal endpoint of the prefix; the policy is pulled
    toward a softmax of the frozen target net's expected endpoint values."""
    values, ceiling, rel = exact_endpoint(batch, deals, scorer)
    legal = batch.legal()
    q_target = (values - ceiling[:, None]) / TARGET_SCALE
    feats = observe(batch)
    with torch.no_grad():
        target_out = net_outputs(target_net, deals, batch, feats)
        target_probs = torch.softmax(target_out["trick_logits"], -1)
        target_values = expected_endpoint(target_probs, batch, scorer) / TARGET_SCALE
        policy_target = torch.softmax(
            (target_values / policy_temperature).masked_fill(~legal, -torch.inf), -1)
    out = net_outputs(net, deals, batch, feats)
    log_policy = F.log_softmax(out["policy_logits"].masked_fill(~legal, -1e9), -1)
    return {"trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
            "q_loss": F.huber_loss(out["contract_q"][legal], q_target[legal]),
            "policy_loss": -(policy_target * log_policy).sum(-1).mean()}


@torch.no_grad()
def contract_score(actor, deals, scorer) -> dict:
    """Greedy silent-opponent auctions on every deal x side x dealer x vulnerability."""
    auctions = run_auctions(actor.eval(), deals, auction_rows(deals.n), scorer, "policy")
    score, ceiling, _ = final_scores(auctions, deals, scorer)
    return {"score": float(score.mean()), "regret": float((ceiling - score).mean()),
            "passout": float((auctions.last < 0).float().mean()),
            "decisions": float(auctions.k.float().mean())}


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    total = dataset_size(args.data)
    validate_training_pool(total, args.train_pool_start, args.train_pool_end,
                           args.train_block_size, args.train_block_every)
    val_span = resolve_range(total, args.val_start, args.val_count)
    eval_span = resolve_range(total, args.eval_start, args.eval_count)
    pool = (args.train_pool_start, args.train_pool_end)
    if ranges_overlap(val_span, eval_span):
        raise ValueError("validation overlaps held-out deals")
    if ranges_overlap(pool, val_span) or ranges_overlap(pool, eval_span):
        raise ValueError("training pool overlaps validation or held-out deals")
    blocks = block_starts(*pool, args.train_block_size, args.seed)
    max_steps = args.max_steps or len(blocks) * args.train_block_every
    if max_steps < 0 or max_steps > len(blocks) * args.train_block_every:
        raise ValueError("--max-steps runs past the last training block")
    if args.eval_every <= 0 or args.warmup <= 0 or args.target_every <= 0:
        raise ValueError("evaluation, warmup and target intervals must be positive")
    val = load_range(args.data, args.val_start, args.val_count)
    scorer = TorchScorer("cpu")

    actor = AuctionContractNet(args.width, args.suit_width, args.depth)
    # The critic is not trained here; stage 2 needs one in the checkpoint.
    critic = CentralCritic(args.width, args.suit_width, args.depth)
    target = copy.deepcopy(actor).eval()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)

    (out / "run.json").write_text(json.dumps(
        {**vars(args), "stage": "ground", "max_steps": max_steps,
         "absolute_ranges": {"train_pool": list(pool), "val": list(val_span),
                             "eval": list(eval_span)},
         "repository": repository_state()}, indent=2))
    log = (out / "train_log.jsonl").open("w")
    start, best, best_step, train = time.time(), -float("inf"), 0, None

    for step in range(0, max_steps + 1):
        if step:
            index = (step - 1) // args.train_block_every
            if (step - 1) % args.train_block_every == 0:
                train = load_range(args.data, blocks[index], args.train_block_size)
                print(f"step {step}: training block {index} = deals "
                      f"[{blocks[index]}, {blocks[index] + args.train_block_size})", flush=True)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * min(1.0, step / args.warmup)
            batch, _ = sample_prefixes(
                train, args.batch, generator, online=actor.eval(), target=target,
                max_depth=args.max_depth, opening_prob=args.opening_prob, window=args.window,
                epsilon=args.epsilon, temperature=1.0)
            actor.train()
            losses = grounding_losses(actor, target, train, batch, scorer,
                                      args.policy_temperature)
            loss = losses["trick_nll"] + losses["q_loss"] + args.policy_weight * losses["policy_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
            optimizer.step()
            if step % args.target_every == 0:
                target.load_state_dict(actor.state_dict())
        if step % args.eval_every and step != max_steps:
            continue
        val_score = contract_score(actor, val, scorer)
        record = {"step": step, "seconds": round(time.time() - start, 1), "validation": val_score,
                  **({"loss": float(loss.detach()), "grad_norm": grad,
                      **{k: float(v.detach()) for k, v in losses.items()}} if step else {})}
        record["best_step"] = step if val_score["score"] > best else best_step
        log.write(json.dumps(record) + "\n")
        log.flush()
        improved = val_score["score"] > best
        if improved:
            best, best_step = val_score["score"], step
            save_checkpoint(out / "best.pt", actor, STAGE, step=step, args=vars(args),
                            val=best, critic_config=critic.config, critic=critic.state_dict(),
                            phase="ground")
        print(f"step {step:>6} contract score {val_score['score']:+.1f} "
              f"(pass-out {val_score['passout']:.3f}) best {best:+.1f} @ {best_step}"
              f"{' *' if improved else ''} [{record['seconds']:.0f}s]", flush=True)
        if args.patience > 0 and step - best_step >= args.patience:
            print(f"EARLY STOP at step {step}: no new best for {args.patience} steps", flush=True)
            break
    log.close()
    result = {"best_step": best_step, "best_score": best, "last_step": step}
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-pool-start", type=int, default=3033000)
    p.add_argument("--train-pool-end", type=int, default=99990000)
    p.add_argument("--train-block-size", type=int, default=1000000)
    p.add_argument("--train-block-every", type=int, default=1000,
                   help="steps per block; 1,000 x 1,024 prefixes is about one prefix per deal")
    p.add_argument("--val-start", type=int, default=3028000)
    p.add_argument("--val-count", type=int, default=5000)
    p.add_argument("--eval-start", type=int, default=-10000)
    p.add_argument("--eval-count", type=int, default=10000)
    p.add_argument("--max-steps", type=int, default=0, help="0: one pass over the pool")
    p.add_argument("--patience", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--policy-weight", type=float, default=0.5)
    p.add_argument("--policy-temperature", type=float, default=0.5)
    p.add_argument("--target-every", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--opening-prob", type=float, default=0.25)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--suit-width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=6)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
