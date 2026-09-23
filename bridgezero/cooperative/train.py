"""Train a full-vocabulary cooperative bidder from random initialization.

One run starts with endpoint grounding, then may use complete-trajectory policy
gradient or two-ply joint policy improvement. No prior policy checkpoint is
accepted.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ..contract.data import dataset_size, load_range, ranges_overlap, resolve_range
from ..contract.evaluate_auction import (
    RULES,
    auction_rows,
    evaluate_auctions,
    run_auctions,
    summary_lines,
    verify_frozen,
)
from ..contract.model import (
    AuctionContractNet,
    ContinuationResidualAuctionNet,
    load_checkpoint,
    save_checkpoint,
)
from ..contract.prefixes import CoopBatch, final_scores, net_outputs, sample_prefixes
from ..contract.targets import TorchScorer
from ..contract.train import learning_rate
from ..contract.train_auction import d1_losses
from .actor_critic import CentralCritic, collect_trajectories, trajectory_losses
from .counterfactual import counterfactual_losses
from .joint_search import (
    build_joint_branches,
    joint_policy_loss,
    run_n_ply,
    run_two_ply,
    sample_opening_roots,
)

PG_STAGE = "D4PG"
JPS_STAGE = "D4JPS"
CF_STAGE = "D4CF"


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def repository_state() -> dict:
    """Commit identity and dirty flag recorded with every experiment."""
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=normal"), cwd=root,
            check=True, capture_output=True, text=True).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


@torch.no_grad()
def policy_closedness(actor, deals, states: list[CoopBatch], chunk: int = 32768) -> dict:
    """Cheap deployable-policy concentration metrics over visited states."""
    effective, top = [], []
    for state in states:
        for start in range(0, len(state), chunk):
            rows = torch.arange(start, min(start + chunk, len(state)),
                                device=state.deal.device)
            sub = state.subset(rows)
            legal = sub.legal()
            logits = net_outputs(actor, deals, sub)["policy_logits"]
            logp = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), -1)
            probability = logp.exp()
            entropy = -(probability * logp.masked_fill(~legal, 0.0)).sum(-1)
            effective.append(entropy.exp())
            top.append(probability.max(-1).values)
    effective_calls, top_probability = torch.cat(effective), torch.cat(top)
    return {
        "effective_calls_mean": float(effective_calls.mean()),
        "effective_calls_median": float(effective_calls.median()),
        "top_probability_median": float(top_probability.median()),
    }


@torch.no_grad()
def validation(actor, deals, scorer) -> dict:
    rows = auction_rows(deals.n, deals.hands.device)
    by_rule = {}
    for rule in RULES:
        states = [] if rule == "policy" else None
        auctions = run_auctions(actor, deals, rows, scorer, rule, record=states)
        score, ceiling, _ = final_scores(auctions, deals, scorer)
        by_rule[rule] = {
            "score": float(score.mean()),
            "regret": float((ceiling - score).mean()),
            "passout": float((auctions.last < 0).float().mean()),
            "decisions": float(auctions.k.float().mean()),
        }
        if states is not None:
            by_rule[rule].update(policy_closedness(actor, deals, states))
    one_ply = run_n_ply(actor, deals, CoopBatch.start(*rows), scorer, 1, "policy")
    one_ply_score, _, _ = final_scores(one_ply, deals, scorer)
    by_rule["one_ply_policy"] = {
        "score": float(one_ply_score.mean()),
        "passout": float((one_ply.last < 0).float().mean()),
    }
    two_ply = run_two_ply(actor, deals, CoopBatch.start(*rows), scorer, "policy")
    two_ply_score, _, _ = final_scores(two_ply, deals, scorer)
    by_rule["two_ply_policy"] = {
        "score": float(two_ply_score.mean()),
        "passout": float((two_ply.last < 0).float().mean()),
    }
    return by_rule


def run(args: argparse.Namespace) -> dict:
    if min(args.ground_steps, args.pg_steps, args.cf_steps,
           args.jps_stop_steps, args.jps_full_steps) < 0:
        raise ValueError("phase step counts must be non-negative")
    if args.pg_steps + args.cf_steps + args.jps_stop_steps + args.jps_full_steps < 1:
        raise ValueError("at least one learning phase after grounding is required")
    if args.cf_steps and (args.pg_steps or args.jps_stop_steps or args.jps_full_steps):
        raise ValueError("the first counterfactual experiment must not mix CF with PG or JPS")
    if args.batch < 1 or args.episodes < 1:
        raise ValueError("batch and episodes must be positive")
    if min(args.jps_roots, args.jps_inner_steps,
           args.jps_target_every, args.jps_eval_every) < 1:
        raise ValueError("joint-search sizes and intervals must be positive")
    if min(args.cf_roots, args.cf_target_every) < 1:
        raise ValueError("counterfactual sizes and intervals must be positive")
    if args.cf_temperature <= 0:
        raise ValueError("cf-temperature must be positive")
    if not 0.0 <= args.cf_support_floor < 1.0:
        raise ValueError("cf-support-floor must be in [0, 1)")
    if not 0.0 <= args.cf_explore < 1.0:
        raise ValueError("cf-explore must be in [0, 1)")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out = Path(args.out)
    _empty_output(out)

    total = dataset_size(args.data)
    spans = {name: resolve_range(total, getattr(args, f"{name}_start"),
                                 getattr(args, f"{name}_count"))
             for name in ("train", "val", "eval")}
    for left in spans:
        for right in spans:
            if left < right and ranges_overlap(spans[left], spans[right]):
                raise ValueError(f"{left} range overlaps {right} range")
    baselines = {name: Path(path) for name, path in
                 (item.split("=", 1) for item in args.baseline)}
    frozen = {name: verify_frozen(path) for name, path in baselines.items()}
    for name, path in baselines.items():
        info = json.loads((path / "run.json").read_text())
        ranges = info.get("absolute_ranges", {})
        if ranges.get("eval") != list(spans["eval"]):
            raise ValueError(f"baseline {name} used a different eval range")
        if ranges_overlap(tuple(ranges["train"]), spans["eval"]):
            raise ValueError(f"baseline {name} trained on this run's eval range")
    train = load_range(args.data, args.train_start, args.train_count, device)
    val = load_range(args.data, args.val_start, args.val_count, device)
    held = load_range(args.data, args.eval_start, args.eval_count, device)
    scorer = TorchScorer(device)

    actor_class = ContinuationResidualAuctionNet if args.cf_steps else AuctionContractNet
    # E52: only passed when on, so the counterfactual net (which has no dropout) is unchanged.
    extra = {"dropout": args.dropout} if args.dropout else {}
    actor = actor_class(args.width, args.suit_width, args.depth,
                        args.policy_logit_bound, **extra).to(device)
    critic = CentralCritic(args.width, args.suit_width, args.depth).to(device)
    target = copy.deepcopy(actor).eval()
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.ground_lr,
                                        weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr,
                                         weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    stage = (CF_STAGE if args.cf_steps else
             JPS_STAGE if args.jps_stop_steps + args.jps_full_steps else PG_STAGE)
    total_steps = (args.ground_steps + args.pg_steps
                   + args.cf_steps + args.jps_stop_steps + args.jps_full_steps)
    ground_schedule = argparse.Namespace(lr=args.ground_lr, warmup=args.warmup,
                                         steps=max(args.ground_steps, 1),
                                         min_lr_frac=args.min_lr_frac)
    run_info = {**vars(args), "stage": stage, "from_scratch": True,
                "action_vocabulary": "35 contracts + Pass; legal mask at every decision",
                "discount": 1.0,
                "absolute_ranges": {name: list(span) for name, span in spans.items()},
                "frozen_baselines": frozen,
                "repository": repository_state(),
                "actor_parameters": sum(p.numel() for p in actor.parameters()),
                "critic_parameters": sum(p.numel() for p in critic.parameters())}
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    log_path = out / "train_log.jsonl"
    log_path.write_text("")
    start = time.time()
    best_score = -math.inf

    def log_and_select(step: int, phase: str, metrics: dict) -> None:
        nonlocal best_score
        scores = validation(actor, val, scorer)
        record = {"step": step, "phase": phase, "seconds": round(time.time() - start, 1),
                  **metrics, "validation": scores}
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"step {step:>6} [{phase}] val E/Q/P "
              + "/".join(f"{scores[r]['score']:+.1f}" for r in RULES)
              + f" one/two {scores['one_ply_policy']['score']:+.1f}"
              + f"/{scores['two_ply_policy']['score']:+.1f}"
              + f" pass {scores['policy']['passout']:.3f} "
              f"dec {scores['policy']['decisions']:.2f} "
              f"eff {scores['policy']['effective_calls_mean']:.2f}", flush=True)
        score = scores["policy"]["score"]
        if score > best_score:
            best_score = score
            save_checkpoint(out / "best.pt", actor, stage, step=step, args=vars(args),
                            val=score, critic_config=critic.config,
                            critic=critic.state_dict(), phase=phase)

    log_and_select(0, "random_init", {})
    for step in range(1, args.ground_steps + 1):
        for group in actor_optimizer.param_groups:
            group["lr"] = learning_rate(step, ground_schedule)
        batch, _ = sample_prefixes(
            train, args.batch, generator, online=actor.eval(), target=target,
            max_depth=args.max_depth, opening_prob=args.opening_prob, window=args.window,
            epsilon=args.epsilon, temperature=args.behavior_temperature, device=device)
        actor.train()
        losses = d1_losses(actor, target, train, batch, scorer, args.policy_temperature,
                           policy_source="expected", q_loss="huber")
        loss = (args.ground_trick_weight * losses["trick_nll"]
                + args.ground_q_weight * losses["q_loss"]
                + args.ground_policy_weight * losses["policy_loss"])
        actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
        actor_optimizer.step()
        if step % args.target_every == 0:
            target.load_state_dict(actor.state_dict())
        if step == 1 or step % args.eval_every == 0 or step == args.ground_steps:
            log_and_select(step, "ground", {
                "loss": float(loss.detach()), "grad_norm": grad,
                **{name: float(value.detach()) for name, value in losses.items()}})
    save_checkpoint(out / "grounded.pt", actor, stage, step=args.ground_steps,
                    args=vars(args), critic_config=critic.config,
                    critic=critic.state_dict(), phase="ground")

    # Start policy-gradient optimization with fresh moments and its own lower
    # learning-rate schedule.  The grounding optimizer's moments describe a
    # different supervised objective and should not silently steer this phase.
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.pg_lr,
                                        weight_decay=args.weight_decay)
    pg_schedule = argparse.Namespace(lr=args.pg_lr, warmup=args.warmup,
                                     steps=args.pg_steps, min_lr_frac=args.min_lr_frac)
    for local_step in range(1, args.pg_steps + 1):
        step = args.ground_steps + local_step
        for group in actor_optimizer.param_groups:
            group["lr"] = learning_rate(local_step, pg_schedule)
        trajectories = collect_trajectories(
            actor, train, args.episodes, generator, scorer,
            args.behavior_temperature, device, args.max_decisions)
        actor.train()
        critic.train()
        losses = trajectory_losses(actor, critic, train, trajectories, args.entropy_weight,
                                   args.behavior_temperature)
        actor_loss = (args.policy_weight * losses["policy_objective"]
                      + args.q_weight * losses["q_loss"]
                      + args.trick_weight * losses["trick_nll"])
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        losses["critic_loss"].backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
        critic_grad = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip))
        actor_optimizer.step()
        critic_optimizer.step()
        if local_step == 1 or local_step % args.eval_every == 0 or local_step == args.pg_steps:
            log_and_select(step, "on_policy", {
                "actor_loss": float(actor_loss.detach()), "actor_grad_norm": actor_grad,
                "critic_grad_norm": critic_grad, "episodes": args.episodes,
                "decisions": len(trajectories),
                **{name: float(value.detach()) for name, value in losses.items()}})

    # Every sampled information state supplies a target for every legal call,
    # including calls to which the current policy assigns negligible mass.
    # The frozen target policy performs the continuation and is refreshed only
    # at explicit intervals, avoiding a moving target within an update.
    if args.cf_steps:
        target.load_state_dict(actor.state_dict())
        target.eval()
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(), lr=args.cf_lr, weight_decay=args.weight_decay)
        cf_schedule = argparse.Namespace(
            lr=args.cf_lr, warmup=args.warmup, steps=args.cf_steps,
            min_lr_frac=args.min_lr_frac)
        for local_step in range(1, args.cf_steps + 1):
            step = args.ground_steps + local_step
            for group in actor_optimizer.param_groups:
                group["lr"] = learning_rate(local_step, cf_schedule)
            trajectories = collect_trajectories(
                actor, train, args.episodes, generator, scorer,
                args.behavior_temperature, device, args.max_decisions,
                uniform_mix=args.cf_explore)
            count = min(args.cf_roots, len(trajectories.states))
            rows = torch.randperm(
                len(trajectories.states), generator=generator)[:count].to(device)
            roots = trajectories.states.subset(rows)
            actor.train()
            losses = counterfactual_losses(
                actor, target, train, roots, scorer,
                args.cf_temperature, args.cf_support_floor)
            loss = (args.cf_policy_weight * losses["policy_loss"]
                    + args.cf_residual_weight * losses["residual_loss"]
                    + args.cf_trick_weight * losses["trick_nll"])
            actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
            actor_optimizer.step()
            if local_step % args.cf_target_every == 0:
                target.load_state_dict(actor.state_dict())
                target.eval()
            if (local_step == 1 or local_step % args.eval_every == 0
                    or local_step == args.cf_steps):
                log_and_select(step, "all_action_cf", {
                    "loss": float(loss.detach()), "actor_grad_norm": grad,
                    "episodes": args.episodes, "roots": count,
                    **{name: float(value.detach()) for name, value in losses.items()},
                })

    # The joint phase begins from the policy learned earlier in this same
    # random-initialization run.  It never loads an external policy checkpoint.
    jps_steps = args.jps_stop_steps + args.jps_full_steps
    if jps_steps:
        save_checkpoint(out / "pre_jps.pt", actor, stage,
                        step=args.ground_steps + args.pg_steps, args=vars(args),
                        critic_config=critic.config, critic=critic.state_dict(),
                        phase="pre_joint")
        blueprint = copy.deepcopy(actor).eval()
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(), lr=args.jps_lr, weight_decay=args.weight_decay)
        jps_schedule = argparse.Namespace(
            lr=args.jps_lr, warmup=args.warmup, steps=jps_steps,
            min_lr_frac=args.min_lr_frac)
        for jps_local in range(1, jps_steps + 1):
            step = args.ground_steps + args.pg_steps + args.cf_steps + jps_local
            mode = "stop" if jps_local <= args.jps_stop_steps else "policy"
            first_full = jps_local == args.jps_stop_steps + 1
            if mode == "policy" and first_full:
                # The forced-stop phase may have learned a new two-call
                # language.  Full continuation must start from that blueprint,
                # not the policy that existed before joint training.
                blueprint.load_state_dict(actor.state_dict())
                blueprint.eval()
            for group in actor_optimizer.param_groups:
                group["lr"] = learning_rate(jps_local, jps_schedule)
            roots = sample_opening_roots(train, args.jps_roots, generator, device)
            branches = build_joint_branches(
                None if mode == "stop" else blueprint,
                train, roots, scorer, continuation_rule=mode)
            actor.train()
            for _ in range(args.jps_inner_steps):
                joint = joint_policy_loss(
                    actor, train, branches, args.jps_entropy_weight,
                    args.jps_temperature)
                actor_optimizer.zero_grad(set_to_none=True)
                joint["loss"].backward()
                grad = float(torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), args.grad_clip))
                actor_optimizer.step()
            if (mode == "policy" and not first_full
                    and (jps_local - args.jps_stop_steps) % args.jps_target_every == 0):
                blueprint.load_state_dict(actor.state_dict())
                blueprint.eval()
            if (jps_local == 1 or first_full or jps_local % args.jps_eval_every == 0
                    or jps_local == args.jps_stop_steps or jps_local == jps_steps):
                log_and_select(step, f"joint_{mode}", {
                    "joint_inner_steps": args.jps_inner_steps,
                    "joint_roots": len(roots),
                    "joint_branches": branches.branch_count,
                    "actor_grad_norm": grad,
                    **{f"joint_{name}": float(value.detach())
                       for name, value in joint.items()},
                })
            if args.jps_stop_steps and jps_local == args.jps_stop_steps:
                save_checkpoint(out / "joint_stop.pt", actor, stage, step=step,
                                args=vars(args), critic_config=critic.config,
                                critic=critic.state_dict(), phase="joint_stop")

    last_phase = ("all_action_cf" if args.cf_steps else
                  "joint_policy" if args.jps_full_steps else
                  "joint_stop" if args.jps_stop_steps else "on_policy")
    save_checkpoint(out / "last.pt", actor, stage, step=total_steps, args=vars(args),
                    critic_config=critic.config, critic=critic.state_dict(), phase=last_phase)
    torch.save({"step": total_steps, "actor": actor.state_dict(),
                "critic": critic.state_dict(), "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "generator": generator.get_state(), "torch_rng": torch.get_rng_state()},
               out / "last_state.pt")
    best, meta = load_checkpoint(out / "best.pt", device, stage)
    report, arrays = evaluate_auctions(best, held, scorer, baselines, args.seed, stage=stage)
    report.update(checkpoint=str(out / "best.pt"), checkpoint_step=meta["step"],
                  from_scratch=True, discount=1.0)
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "eval_rows.npz", **arrays)
    print(f"\nHELD-OUT EVAL (best policy step {meta['step']})")
    print("\n".join(summary_lines(report)), flush=True)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--train-start", type=int, default=2028000)
    parser.add_argument("--train-count", type=int, default=1000000)
    parser.add_argument("--val-start", type=int, default=3028000)
    parser.add_argument("--val-count", type=int, default=5000)
    parser.add_argument("--eval-start", type=int, default=-10000)
    parser.add_argument("--eval-count", type=int, default=10000)
    parser.add_argument("--ground-steps", type=int, default=6000)
    parser.add_argument("--pg-steps", type=int, default=6000)
    parser.add_argument("--cf-steps", type=int, default=0,
                        help="all-legal-call frozen-continuation updates")
    parser.add_argument("--jps-stop-steps", type=int, default=0,
                        help="joint two-ply updates that force Pass after the reply")
    parser.add_argument("--jps-full-steps", type=int, default=0,
                        help="joint two-ply updates with frozen-policy continuation")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--ground-lr", type=float, default=1e-3)
    parser.add_argument("--pg-lr", type=float, default=3e-4)
    parser.add_argument("--cf-lr", type=float, default=3e-4)
    parser.add_argument("--jps-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--ground-trick-weight", type=float, default=1.0)
    parser.add_argument("--ground-q-weight", type=float, default=1.0)
    parser.add_argument("--ground-policy-weight", type=float, default=0.5)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--trick-weight", type=float, default=0.2)
    parser.add_argument("--cf-policy-weight", type=float, default=1.0)
    parser.add_argument("--cf-residual-weight", type=float, default=1.0)
    parser.add_argument("--cf-trick-weight", type=float, default=0.2)
    parser.add_argument("--cf-temperature", type=float, default=0.5)
    parser.add_argument("--cf-support-floor", type=float, default=0.02,
                        help="total uniform legal mass in counterfactual policy targets")
    parser.add_argument("--cf-explore", type=float, default=0.02,
                        help="uniform legal behavior mixture while collecting CF states")
    parser.add_argument("--cf-roots", type=int, default=128)
    parser.add_argument("--cf-target-every", type=int, default=100)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--jps-entropy-weight", type=float, default=0.01)
    parser.add_argument("--jps-temperature", type=float, default=1.0)
    parser.add_argument("--jps-roots", type=int, default=32)
    parser.add_argument("--jps-inner-steps", type=int, default=4)
    parser.add_argument("--jps-target-every", type=int, default=50)
    parser.add_argument("--jps-eval-every", type=int, default=100)
    parser.add_argument("--policy-temperature", type=float, default=0.5)
    parser.add_argument("--behavior-temperature", type=float, default=1.0)
    parser.add_argument("--target-every", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-decisions", type=int, default=40)
    parser.add_argument("--opening-prob", type=float, default=0.25)
    parser.add_argument("--window", type=int, default=10,
                        help="structured prefix-history sampling window; action targets stay full")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--suit-width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="E52: dropout after every activation of the actor's suit net, "
                             "auction net and trunk (0: off, as every run before E52)")
    parser.add_argument("--policy-logit-bound", type=float, default=None,
                        help="positive symmetric affine bound; preserves greedy call ordering")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
