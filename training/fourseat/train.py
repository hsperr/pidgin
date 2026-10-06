"""Stages 2 and 3: four-seat self-play actor-critic.

All four seats bid with one shared network: 35 contracts, Pass, Double,
Redouble, and a Sacrifice gate. ``--silent-frac`` of the self-play episodes
force one random side to Pass, which keeps the cooperative game in the mix.

``--table-weight`` sets the reward. At 0 (stage 2) each side is rewarded for
its own contract. At 1 (stage 3) each side gets the real table result, which
includes the opponents' contract and doubles. ``--league-frac`` plays that
share of episodes against a random earlier snapshot of this run.

Every ``--eval-every`` steps the greedy policy bids the validation deals and the
``--select`` metric picks ``best.pt``. Training stops when that metric has not
improved for ``--patience`` steps, when a guard trips, or after ``--steps``
(default: one sweep through the training blocks, sampling within each block).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..contract.data import block_starts, dataset_size, load_range, ranges_overlap, resolve_range, validate_training_pool
from ..contract.targets import TorchScorer
from ..ground import repository_state
from .competitive import (
    competitive_trajectory_losses,
    competitive_trajectory_stats,
    competitive_validation,
    set_any_seat_double,
    set_opening_rule,
)
from .fast_rollout import FastCollector
from .model import (
    STAGE_TEXT,
    load_fourseat_checkpoint,
    save_fourseat_checkpoint,
    warm_start_competitive,
)
from .rollout import silent_validation, trajectory_stats
from .rulebots import RuleBot
from .state import set_own_down_doubled, set_table_down_doubled

ROOT = Path(__file__).resolve().parents[2]


def match_imps(args: argparse.Namespace, out: Path, step: int, checkpoint: Path) -> float:
    """Paired IMPs/board of ``checkpoint`` vs ``--imp-opponent`` on the validation deals."""
    mdir = out / "matches" / f"step{step}"
    subprocess.run([sys.executable, "-u", str(ROOT / "tools" / "match.py"),
                    "--a", f"four:{Path(checkpoint).resolve()}",
                    "--b", f"four:{Path(args.imp_opponent).resolve()}",
                    "--data", str(Path(args.data).resolve()), "--start", str(args.val_start),
                    "--deals", str(args.val_count), "--threads", str(args.threads),
                    "--replay-check", "50", "--out", str(mdir.resolve())],
                   check=True, stdout=subprocess.DEVNULL, cwd=ROOT)
    report = json.loads((mdir / "results.json").read_text())
    return float(report["imps_per_board"]["mean"])


def summary_line(silent: dict, four: dict) -> str:
    levels, spots = four["level_share"], four["spots_per_1000"]
    return (f"silent {silent['score']:+.1f} | 4seat own {four['own_score']:+.1f} "
            f"both-bid {four['both_sides_bid']:.2f} calls {four['calls']:.1f} "
            f"lvl1-3/4-5/6-7 {levels['1'] + levels['2'] + levels['3']:.2f}/"
            f"{levels['4'] + levels['5']:.2f}/{levels['6'] + levels['7']:.2f} "
            f"| X {four['double_rate']:.3f} XX {four['redouble_rate']:.3f} "
            f"SAC {four['sac_rate']:.4f} | spots/1k X {spots['x']:.0f} XX {spots['xx']:.1f} "
            f"SAC {spots['sac']:.0f}"
            + (f" | code words {four['simplicity']['code_words_per_100']:.1f}/100"
               if "simplicity" in four else ""))


def run(args: argparse.Namespace) -> dict:
    if args.select == "imp" and not args.imp_opponent:
        raise ValueError("--select imp needs --imp-opponent")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    set_any_seat_double(args.any_seat_double)
    set_opening_rule(args.opening_rule)
    set_own_down_doubled(args.own_down_doubled)
    set_table_down_doubled(args.table_down_doubled)
    out = Path(args.out)
    if args.resume and not (out / "last_state.pt").exists():
        raise ValueError(f"--resume needs {out / 'last_state.pt'}")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    total = dataset_size(args.data)
    validate_training_pool(total, args.train_pool_start, args.train_pool_end,
                           args.train_block_size, args.train_block_every)
    spans = {"train": (args.train_pool_start, args.train_pool_end),
             "val": resolve_range(total, args.val_start, args.val_count),
             "eval": resolve_range(total, args.eval_start, args.eval_count)}
    for left in spans:
        for right in spans:
            if left < right and ranges_overlap(spans[left], spans[right]):
                raise ValueError(f"{left} range overlaps {right} range")
    blocks = block_starts(args.train_pool_start, args.train_pool_end,
                          args.train_block_size, args.seed)
    steps = args.steps or len(blocks) * args.train_block_every
    if steps > len(blocks) * args.train_block_every:
        raise ValueError("--steps runs past the last training block")
    dev = torch.device(args.device)
    val = load_range(args.data, args.val_start, args.val_count, dev)
    held = load_range(args.data, args.eval_start, args.eval_count, dev)
    scorer = TorchScorer(dev)

    actor, critic, init_meta = warm_start_competitive(
        args.init, dev, fourseat=False, redouble=True, sacrifice=True,
        double_bias=args.double_bias, xx_bias=args.xx_bias, sac_bias=args.sac_bias)

    # League (fictitious self-play): a frozen copy that loads a uniformly random
    # earlier snapshot of this run before every rollout.
    pool, league = {}, None
    if args.league_frac > 0:
        league = {"net": copy.deepcopy(actor).eval().requires_grad_(False),
                  "rng": random.Random(args.seed + 9173),
                  "snapshots": [(0, {k: v.detach().clone()
                                     for k, v in actor.state_dict().items()})],
                  "pick": 0}
        pool[1] = (league["net"], args.league_frac)
    punisher = None
    if args.punisher_frac > 0:
        # A DD punisher: the starting policy (or, with --punisher-refresh, a recent copy of
        # the learner), except that it doubles exactly the failing contracts, so thin bids
        # and passes of a double are charged at the table.
        punisher = copy.deepcopy(actor).eval().requires_grad_(False)
        punisher.punisher_level, punisher.punisher_miss = args.punisher_level, args.punisher_miss
        pool[2] = (punisher, args.punisher_frac)
    if args.rule_frac > 0:
        styles = [s for s in args.rule_bots.split(",") if s]
        for i, style in enumerate(styles):
            pool[3 + i] = (RuleBot(style), args.rule_frac / len(styles))
    if args.fixed_frac > 0:
        # Fixed reference opponents (e.g. an earlier model), never updated.
        paths = [s for s in args.fixed_opponents.split(",") if s]
        if not paths:
            raise ValueError("--fixed-frac needs --fixed-opponents")
        for i, path in enumerate(paths):
            net, _ = load_fourseat_checkpoint(path, dev)
            pool[20 + i] = (net.eval().requires_grad_(False), args.fixed_frac / len(paths))

    # The detached Double/Redouble/Sacrifice value and gate heads get their own
    # learning rates, as a fixed multiple of the policy learning rate.
    head_groups = []
    for names, head_lr in ((("double_value_head", "double_gate_head"), args.double_value_lr),
                           (("redouble_value_head", "redouble_gate_head"), args.xx_value_lr),
                           (("sac_value_head", "sac_gate_head"), args.sac_value_lr)):
        params = [p for name in names for p in getattr(actor, name).parameters()]
        head_groups.append({"params": params, "scale": head_lr / args.pg_lr})
    head_ids = {id(p) for group in head_groups for p in group["params"]}
    groups = [{"params": [p for p in actor.parameters() if id(p) not in head_ids], "scale": 1.0},
              *head_groups]
    actor_optimizer = torch.optim.AdamW(groups, lr=args.pg_lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr,
                                         weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    objective, features = STAGE_TEXT[actor.stage]
    run_info = {**vars(args), "steps": steps, "stage": actor.stage, "objective": objective,
                "features": features, **init_meta,
                "absolute_ranges": {name: list(span) for name, span in spans.items()},
                "repository": repository_state(),
                "actor_parameters": sum(p.numel() for p in actor.parameters())}
    extra = {**init_meta, "args": vars(args), "any_seat_double": bool(args.any_seat_double),
             "opening_rule": int(args.opening_rule)}
    log_path = out / "train_log.jsonl"
    start = time.time()
    best, best_step, best_own = -math.inf, 0, -math.inf
    level5_start = code_words_start = math.nan
    first_step = 1
    if args.resume:
        # map to CPU so a run saved on a GPU box resumes anywhere; load_state_dict moves it back
        state = torch.load(out / "last_state.pt", map_location="cpu", weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        first_step = state["step"] + 1
        best, best_step, best_own = state["best"], state["best_step"], state["best_own"]
        level5_start = state["level5_start"]
        code_words_start = state.get("code_words_start", math.nan)
        if league is not None:
            league["snapshots"], league["rng"] = state["league_snapshots"], state["league_rng"]
        (out / f"run_resume_step{state['step']}.json").write_text(json.dumps(run_info, indent=2))
        print(f"RESUMED from step {state['step']} (best {best:+.3f} at {best_step})", flush=True)
    else:
        (out / "run.json").write_text(json.dumps(run_info, indent=2))
        log_path.write_text("")

    def save_state(step: int) -> None:
        save_fourseat_checkpoint(out / "last.pt", actor, critic, step=step, **extra)
        torch.save({"step": step, "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "best": best, "best_step": best_step, "best_own": best_own,
                    "level5_start": level5_start, "code_words_start": code_words_start,
                    **({"league_snapshots": league["snapshots"], "league_rng": league["rng"]}
                       if league is not None else {})},
                   out / "last_state.pt.tmp")
        (out / "last_state.pt.tmp").replace(out / "last_state.pt")

    def guard(four: dict) -> str | None:
        """Why this policy must not be kept, or None."""
        if four["double_rate"] > args.max_double_rate:
            return f"double rate {four['double_rate']:.3f} > {args.max_double_rate}"
        if four["sac_rate"] > args.max_sac_rate:
            return f"sacrifice rate {four['sac_rate']:.4f} > {args.max_sac_rate}"
        if four["level5_share"] > level5_start + args.max_level5_rise:
            return (f"level-5+ share {four['level5_share']:.3f} rose more than "
                    f"{args.max_level5_rise} above its start {level5_start:.3f}")
        if four["own_score"] < best_own - args.max_own_drop:
            return (f"own-contract score {four['own_score']:.1f} fell more than "
                    f"{args.max_own_drop} below its best {best_own:.1f}")
        return None

    def evaluate(step: int, metrics: dict) -> tuple[dict, str | None]:
        """Validate, log, and keep the policy as best.pt unless a guard trips."""
        nonlocal best, best_step
        silent = silent_validation(actor, val, scorer)
        four = competitive_validation(actor, val, scorer)
        if args.select == "imp":
            snapshot = out / f"ckpt_step{step}.pt"
            save_fourseat_checkpoint(snapshot, actor, critic, step=step, **extra)
            four["imps_vs_opponent"] = match_imps(args, out, step, snapshot.resolve())
        metric = {"objective": four["objective"], "own": four["own_score"],
                  "imp": four.get("imps_vs_opponent")}[args.select]
        record = {"step": step, "seconds": round(time.time() - start, 1), **metrics,
                  "select": args.select, "select_metric": metric,
                  "validation": {"silent": silent, "fourseat": four}}
        tripped = guard(four) if step else None
        # Keep the measured starting policy if training only produces weaker
        # policies or trips a guard. A policy that got much less simple is not kept,
        # but training goes on.
        code_words = four.get("simplicity", {}).get("code_words_per_100", math.nan)
        too_coded = step and code_words > code_words_start + args.max_code_words_rise
        improved = metric > best and tripped is None and not too_coded
        if improved:
            best, best_step = metric, step
            save_fourseat_checkpoint(out / "best.pt", actor, critic, step=step,
                                     val_fourseat=four, val_silent=silent, **extra)
        record.update(best_step=best_step, accepted=bool(improved), guard=tripped)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"step {step:>6} {summary_line(silent, four)} | {args.select} {metric:+.3f} "
              f"best {best:+.3f} @ {best_step}{' *' if improved else ''} "
              f"[{record['seconds']:.0f}s]", flush=True)
        return four, tripped

    if not args.resume:
        start_four, _ = evaluate(0, {"phase": "warm_start"})
        best_own = start_four["own_score"]
        level5_start = start_four["level5_share"]
        code_words_start = start_four.get("simplicity", {}).get("code_words_per_100", math.nan)
    collector = FastCollector(actor, args.episodes, dev, True, args.behavior_temperature, pool)
    stopped, block, train, step = None, -1, None, first_step - 1
    for step in range(first_step, steps + 1):
        if (step - 1) // args.train_block_every != block:
            block = (step - 1) // args.train_block_every
            train = load_range(args.data, blocks[block], args.train_block_size, dev)
            print(f"step {step}: training block {block} = deals "
                  f"[{blocks[block]}, {blocks[block] + args.train_block_size})", flush=True)
        for group in actor_optimizer.param_groups:
            group["lr"] = args.pg_lr * min(1.0, step / args.warmup) * group["scale"]
        if league is not None:
            league["pick"], weights = league["rng"].choice(league["snapshots"])
            league["net"].load_state_dict(weights)
        trajectories = collector.collect(train, generator, scorer, args.silent_frac)
        actor.train()
        critic.train()
        losses = competitive_trajectory_losses(
            actor, critic, train, trajectories, args.entropy_weight,
            args.behavior_temperature, args.double_tau, args.xx_tau, args.sac_tau,
            policy_cf=step > args.double_policy_start, gate_pg=args.gate_pg,
            all_spots=False, table_weight=args.table_weight,
            code_word_penalty=args.code_word_penalty, light_open_penalty=args.light_open_penalty)
        actor_loss = (losses["policy_objective"]
                      + args.q_weight * losses["q_loss"]
                      + args.trick_weight * losses["trick_nll"]
                      + args.double_value_weight * losses["double_value_loss"]
                      + args.double_cf_weight * losses["double_ce"]
                      + args.xx_value_weight * losses["redouble_value_loss"]
                      + args.xx_cf_weight * losses["redouble_ce"]
                      + args.sac_value_weight * losses["sac_value_loss"]
                      + args.sac_cf_weight * losses["sac_ce"])
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        losses["critic_loss"].backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
        critic_grad = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip))
        actor_optimizer.step()
        critic_optimizer.step()
        if (step == 1 and args.select != "imp") or step % args.eval_every == 0 or step == steps:
            four, stopped = evaluate(step, {
                "phase": "on_policy", "lr": actor_optimizer.param_groups[0]["lr"],
                "train_block_index": block, "train_block_start": blocks[block],
                **({"league_size": len(league["snapshots"]),
                    "league_opponent_step": league["pick"]} if league is not None else {}),
                "actor_loss": float(actor_loss.detach()), "actor_grad_norm": actor_grad,
                "critic_grad_norm": critic_grad, **trajectory_stats(trajectories),
                **competitive_trajectory_stats(trajectories),
                **{name: float(value.detach()) for name, value in losses.items()}})
            if not stopped and args.patience > 0 and step - best_step >= args.patience:
                stopped = f"no new best {args.select} for {args.patience} steps"
            best_own = max(best_own, four["own_score"])
            if stopped:
                print(f"EARLY STOP at step {step}: {stopped}", flush=True)
                break
        if punisher is not None and args.punisher_refresh and step % args.punisher_refresh == 0:
            punisher.load_state_dict(actor.state_dict())
        if league is not None and step % args.league_every == 0:
            league["snapshots"].append(
                (step, {k: v.detach().clone() for k, v in actor.state_dict().items()}))
        if step % args.snapshot_every == 0:
            save_state(step)
            if not (out / f"ckpt_step{step}.pt").exists():
                save_fourseat_checkpoint(out / f"ckpt_step{step}.pt", actor, critic,
                                         step=step, **extra)
    save_state(step)
    if not (out / "best.pt").exists():
        raise RuntimeError("no accepted checkpoint; the run has no best.pt")

    # Held-out report for the chosen checkpoint.
    net, meta = load_fourseat_checkpoint(out / "best.pt", dev)
    report = {"checkpoint": str(out / "best.pt"), "checkpoint_step": meta["step"],
              "select": args.select, "select_metric": best, "early_stop": stopped,
              "last_step": step, "eval_range": list(spans["eval"]),
              "silent": silent_validation(net, held, scorer),
              "fourseat": competitive_validation(net, held, scorer)}
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"\nHELD-OUT (best step {meta['step']}): "
          f"{summary_line(report['silent'], report['fourseat'])}", flush=True)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu", help="cpu or cuda (mps works but is slower)")
    p.add_argument("--description", default="",
                   help="what this run is; shown on the dashboard (saved in run.json)")
    p.add_argument("--init", required=True,
                   help="stage-1 checkpoint (stage 2) or four-seat checkpoint (stage 3)")
    p.add_argument("--resume", action="store_true", help="continue --out from last_state.pt")
    # data
    p.add_argument("--train-pool-start", type=int, default=3033000)
    p.add_argument("--train-pool-end", type=int, default=99990000)
    p.add_argument("--train-block-size", type=int, default=1000000)
    p.add_argument("--train-block-every", type=int, default=2000,
                   help="steps per block; 2,000 x 512 episodes is about one episode per deal")
    p.add_argument("--val-start", type=int, default=3028000)
    p.add_argument("--val-count", type=int, default=5000)
    p.add_argument("--eval-start", type=int, default=-10000)
    p.add_argument("--eval-count", type=int, default=10000)
    # schedule and selection
    p.add_argument("--steps", type=int, default=0,
                   help="0: one sweep through the pool blocks, with replacement within each block")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=1000)
    p.add_argument("--select", choices=("own", "imp", "objective"), default="own",
                   help="best.pt metric: own-contract score, paired IMPs vs --imp-opponent "
                        "on the validation deals, or own score plus X/XX/SAC credit")
    p.add_argument("--imp-opponent", default="")
    p.add_argument("--patience", type=int, default=5000,
                   help="stop when the --select metric has no new best for this many steps; 0 disables")
    p.add_argument("--max-double-rate", type=float, default=0.40)
    p.add_argument("--max-sac-rate", type=float, default=0.20)
    p.add_argument("--max-level5-rise", type=float, default=0.05,
                   help="stop when the level-5+ contract share rises this much above step 0")
    p.add_argument("--max-code-words-rise", type=float, default=math.inf,
                   help="best.pt only takes policies whose code words per 100 auctions are at "
                        "most this far above step 0 (training continues either way)")
    p.add_argument("--max-own-drop", type=float, default=math.inf,
                   help="stop when the own-contract score falls this far below its best")
    # game and reward
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--silent-frac", type=float, default=0.25)
    p.add_argument("--table-weight", type=float, default=0.0,
                   help="0: own-contract reward; 1: real table result")
    p.add_argument("--own-down-doubled", type=float, default=0.0,
                   help="own-contract reward: share of the doubled penalty a failing own "
                        "contract costs (0 = undoubled, as before; 1 = always doubled)")
    p.add_argument("--table-down-doubled", nargs="?", const="both", default="",
                   choices=("", "both", "own"),
                   help="table result with perfect doublers (use with --table-weight > 0): "
                        "'both' (bare flag) scores every failing contract doubled; 'own' only "
                        "a side's own failing contracts, so doubling the opponents still pays")
    p.add_argument("--league-frac", type=float, default=0.0)
    p.add_argument("--league-every", type=int, default=1000)
    p.add_argument("--punisher-frac", type=float, default=0.0,
                   help="share of episodes against a DD punisher: a frozen copy of --init "
                        "whose Double is decided by the deal's double-dummy tricks")
    p.add_argument("--punisher-level", type=int, default=1,
                   help="the punisher doubles failing contracts at this level or higher")
    p.add_argument("--punisher-refresh", type=int, default=0,
                   help="copy the learner into the punisher every this many steps (0: keep --init)")
    p.add_argument("--punisher-miss", type=float, default=0.0,
                   help="chance the punisher passes a failing contract anyway")
    p.add_argument("--rule-frac", type=float, default=0.0,
                   help="share of episodes against rule bidders (split evenly over --rule-bots)")
    p.add_argument("--rule-bots", default="sayc,weakclub,happy",
                   help="comma-separated training/fourseat/rulebots.py styles")
    p.add_argument("--fixed-frac", type=float, default=0.0,
                   help="share of episodes against --fixed-opponents (split evenly)")
    p.add_argument("--fixed-opponents", default="",
                   help="comma-separated four-seat checkpoints played as frozen opponents")
    p.add_argument("--any-seat-double", action="store_true",
                   help="X/XX legal at every seat, not only where Pass would end the auction")
    p.add_argument("--behavior-temperature", type=float, default=1.0)
    # optimizer and loss weights
    p.add_argument("--pg-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--q-weight", type=float, default=0.5)
    p.add_argument("--code-word-penalty", type=float, default=0.0,
                   help="cost (/100 points) on each call that is a code word "
                        "(tools/simplicity.py rule); 0 = off")
    p.add_argument("--light-open-penalty", type=float, default=0.0,
                   help="cost (/100 points) on each opening with LIGHT_HCP (7) HCP or fewer, "
                        "natural preempts (2+ level, 6+ card suit) excepted; 0 = off")
    p.add_argument("--opening-rule", type=int, default=0,
                   help="hard rule: in 1st/2nd seat a 1-level opening needs HCP + two longest "
                        "suits >= this (18 = rule of 18), for every net at the table; 0 = off")
    p.add_argument("--trick-weight", type=float, default=0.2)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    # Double / Redouble / Sacrifice heads
    p.add_argument("--double-bias", type=float, default=-3.0)
    p.add_argument("--xx-bias", type=float, default=-6.0)
    p.add_argument("--sac-bias", type=float, default=-6.0)
    p.add_argument("--double-tau", type=float, default=0.5)
    p.add_argument("--xx-tau", type=float, default=0.1)
    p.add_argument("--sac-tau", type=float, default=0.5)
    p.add_argument("--double-value-weight", type=float, default=0.5)
    p.add_argument("--xx-value-weight", type=float, default=0.5)
    p.add_argument("--sac-value-weight", type=float, default=0.5)
    p.add_argument("--double-value-lr", type=float, default=1e-3)
    p.add_argument("--xx-value-lr", type=float, default=1e-3)
    p.add_argument("--sac-value-lr", type=float, default=1e-3)
    p.add_argument("--double-cf-weight", type=float, default=1.0)
    p.add_argument("--xx-cf-weight", type=float, default=1.0)
    p.add_argument("--sac-cf-weight", type=float, default=1.0)
    p.add_argument("--double-policy-start", type=int, default=250,
                   help="steps of value-only Double training before its policy loss")
    p.add_argument("--gate-pg", action="store_true",
                   help="let the policy gradient also train the XX/SAC gate logits")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=8)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
