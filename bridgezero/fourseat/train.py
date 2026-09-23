"""Four-seat own-bid actor-critic, warm-started from a cooperative checkpoint.

All four seats bid with one shared network (35 contracts + Pass; ``--doubles``
adds Double, never Redouble). Each side's calls receive that side's own-bid
return (not zero-sum). A ``--silent-frac`` share of episodes forces one random
side to Pass. With ``--doubles`` see ``double.py`` for the Double credit and the
counterfactual {Pass, X} loss.

    OMP_NUM_THREADS=4 python -u -m bridgezero.fourseat.train \
      --init runs/E15d_scratch_cooperative_pg_seed1_1m/best.pt \
      --baseline E3=runs/E3_stageC_16k --seed 1 \
      --out runs/E18_fourseat_ownbid_from_E15d

    # E19: add --doubles --min-own-score 100 --out runs/E19_fourseat_ownbid_double_from_E15d
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ..contract.data import dataset_size, load_range, ranges_overlap, resolve_range
from ..contract.evaluate import paired_bootstrap
from ..contract.evaluate_auction import evaluate_auctions, summary_lines, verify_frozen
from ..contract.model import load_checkpoint
from ..contract.targets import TorchScorer
from ..contract.train import learning_rate
from ..cooperative.actor_critic import trajectory_losses
from ..cooperative.train import repository_state
from .competitive import (
    set_any_seat_double,
    collect_competitive_trajectories,
    competitive_trajectory_losses,
    competitive_trajectory_stats,
    competitive_validation,
)
from .double import double_trajectory_losses
from .fast_rollout import FastCollector
from .model import (
    STAGE_TEXT,
    SilentView,
    load_fourseat_checkpoint,
    save_fourseat_checkpoint,
    sha256,
    warm_start,
    warm_start_competitive,
    warm_start_from_fourseat,
)
from .rollout import collect_trajectories, fourseat_validation, silent_validation, trajectory_stats

DATA = "data/deals.npz"   # override with --data; see README "Getting deals"
DEFAULT_INIT = "runs/E15d_scratch_cooperative_pg_seed1_1m/best.pt"
ROOT = Path(__file__).resolve().parents[2]


def block_start(args: argparse.Namespace, index: int) -> int:
    """Seeded start of training block ``index`` inside [train_pool_start, train_pool_end)."""
    rng = np.random.default_rng([args.seed, index])
    return int(rng.integers(args.train_pool_start,
                            args.train_pool_end - args.train_block_size + 1))


def launch_match(args: argparse.Namespace, out: Path, step: int, checkpoint: Path) -> None:
    """Background IMP match of ``checkpoint`` vs ``--match-opponent``; one line to matches.log."""
    mdir = out / "matches" / f"step{step}"
    mdir.mkdir(parents=True, exist_ok=True)
    cmd = (f"OMP_NUM_THREADS={args.match_threads} {sys.executable} -u "
           f"{ROOT}/experiments/match/match.py --a 'four:{checkpoint}' "
           f"--b 'four:{args.match_opponent}' --deals {args.match_deals} "
           f"--threads {args.match_threads} --replay-check 50 --data {args.data} "
           f"--out {mdir} > {mdir}/stdout.txt 2>&1; "
           f"echo \"step {step}: $(sed -n 2p {mdir}/stdout.txt) vs {args.match_opponent}\" "
           f">> {out}/matches.log")
    subprocess.Popen(cmd, shell=True, cwd=ROOT, start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _OwnStepZero(dict):
    """Guard reference without a log: every step compares to this run's step-0 silent."""

    def get(self, step, default=None):
        return super().get(0, default)


class _LogReference(dict):
    """Silent val score by step from a (possibly still growing) reference train_log.jsonl."""

    def __init__(self, path: str):
        super().__init__()
        self.path = Path(path)

    def get(self, step, default=None):
        if step not in self and self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self[rec["step"]] = rec["validation"]["silent"]["score"]
        return super().get(step, default)


def silent_guard_reference(args: argparse.Namespace) -> dict:
    """{step: silent val score} of arm A (``--silent-guard-reference`` train_log.jsonl, re-read
    as it grows, so arms may run concurrently; steps A has not logged yet are not guarded)."""
    if args.silent_guard is None:
        return {}
    if not args.silent_guard_reference:
        return _OwnStepZero()
    return _LogReference(args.silent_guard_reference)


def table_weight_at(step: int, args: argparse.Namespace) -> float:
    """Lambda of the real table result at ``step`` (E42's --table-weight, E49's ramp).

    With ``--table-weight-steps 0`` the weight is constant, which is every run up to E48. A
    ramp lets a policy learn to bid its own contracts under a mostly own-score return before
    the return becomes the whole table's business, which is the recovery path if training
    straight at lambda 1 collapses.
    """
    if args.table_weight_steps <= 0:
        return args.table_weight
    frac = min(1.0, max(0.0, (step - 1) / args.table_weight_steps))
    return args.table_weight_start + (args.table_weight - args.table_weight_start) * frac


def shared_grad_norms(actor, args: argparse.Namespace, losses: dict, belief: dict | None) -> dict:
    """L2 norm of each weighted loss's gradient on the shared encoder (suit/auction/trunk)."""
    shared = [p for p in actor.shared_parameters() if p.requires_grad]
    terms = {"policy": args.policy_weight * losses["policy_objective"],
             "q": args.q_weight * losses["q_loss"], "trick": args.trick_weight * losses["trick_nll"]}
    if args.doubles and args.double_value:
        terms["double_value"] = args.double_value_weight * losses["double_value_loss"]
        terms["double_ce"] = args.double_cf_weight * losses["double_ce"]
    if getattr(args, "redouble", False):
        terms["redouble"] = (args.xx_value_weight * losses["redouble_value_loss"]
                             + args.xx_cf_weight * losses["redouble_ce"])
    if getattr(args, "sacrifice", False):
        terms["sacrifice"] = (args.sac_value_weight * losses["sac_value_loss"]
                              + args.sac_cf_weight * losses["sac_ce"])
    if belief is not None and args.belief == "shared":
        terms["belief"] = args.belief_weight * belief["loss"]
    out = {}
    for name, loss in terms.items():
        if not loss.requires_grad:
            out[name] = 0.0
            continue
        grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
        out[name] = float(sum(float(g.pow(2).sum()) for g in grads if g is not None) ** 0.5)
    return out


def init_belief_head(args: argparse.Namespace, actor, device):
    """``(head, how)`` for ``--belief detached|shared``; ``(None, None)`` when off.

    The head comes from ``--init`` when that checkpoint carries ``belief_head`` with the same
    config (continuing a belief run); otherwise it is fresh with a zero-init output (NLL ln 3),
    e.g. warm-starting from an E28 snapshot. Built under a forked RNG so the policy run's
    stream is unchanged. ``--resume`` later overwrites it from last_state.pt.
    """
    if args.belief == "off":
        return None, None
    from .belief_model import BeliefHead
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 7919)
        head = BeliefHead(actor.config["width"], args.belief_width).to(device)
    init = Path(args.init)
    if init.exists():
        ck = torch.load(init, map_location=device, weights_only=False)
        if "belief_head" in ck and ck.get("belief_config") == head.config:
            head.load_state_dict(ck["belief_head"])
            return head, f"checkpoint ({ck.get('belief_mode', '?')})"
    return head, "fresh"


def run(args: argparse.Namespace) -> dict:
    if args.steps < 1 or args.episodes < 1:
        raise ValueError("steps and episodes must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if getattr(args, "cooperative", False):
        if args.pool or args.league_frac > 0:
            raise ValueError("--cooperative cannot be combined with --pool or --league-frac")
        args.silent_frac = 1.0
        args.table_weight = args.table_weight_start = 0.0
        args.table_weight_steps = 0
    set_any_seat_double(args.any_seat_double)
    device = torch.device(args.device)
    out = Path(args.out)
    if args.resume and not (out / "last_state.pt").exists():
        raise ValueError(f"--resume needs {out / 'last_state.pt'}")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise ValueError(f"output directory is not empty: {out}")
    blocks = args.train_block_every > 0
    out.mkdir(parents=True, exist_ok=True)

    total = dataset_size(args.data)
    spans = {name: resolve_range(total, getattr(args, f"{name}_start"),
                                 getattr(args, f"{name}_count"))
             for name in (("val", "eval") if blocks else ("train", "val", "eval"))}
    if blocks:
        if not 0 <= args.train_pool_start < args.train_pool_end - args.train_block_size <= total:
            raise ValueError("invalid training block pool")
        spans["train"] = (args.train_pool_start, args.train_pool_end)
    for left in spans:
        for right in spans:
            if left < right and ranges_overlap(spans[left], spans[right]):
                raise ValueError(f"{left} range overlaps {right} range")
    baselines = {name: Path(path) for name, path in
                 (item.split("=", 1) for item in args.baseline)}
    frozen = {name: verify_frozen(path) for name, path in baselines.items()}
    train = None if blocks else load_range(args.data, args.train_start, args.train_count, device)
    block = {"index": -1, "start": args.train_start}
    val = load_range(args.data, args.val_start, args.val_count, device)
    held = load_range(args.data, args.eval_start, args.eval_count, device)
    scorer = TorchScorer(device)

    competitive = args.redouble or args.sacrifice      # stage D5OWN4XC
    if args.init_fourseat or competitive:
        args.double_gate = True
    args.double_value = args.double_value or args.double_gate
    args.doubles = args.doubles or args.double_value
    if competitive:
        actor, critic, init_meta = warm_start_competitive(
            args.init, device, args.init_fourseat, args.redouble, args.sacrifice,
            args.double_bias, args.xx_bias, args.sac_bias, args.belief_summary)
    elif args.init_fourseat:
        actor, critic, init_meta = warm_start_from_fourseat(args.init, device, args.double_bias)
    else:
        actor, critic, init_meta = warm_start(args.init, device, args.doubles, args.double_bias,
                                              args.double_value, args.double_gate)
    pool, pool_names, pool_meta = {}, {}, {}
    for code, item in enumerate(args.pool, 1):
        name, spec = item.split("=", 1)
        kind, path, frac = spec.split(":")
        if kind == "brl":
            raise ValueError("brl opponents are not part of this repository")
        net = (load_fourseat_checkpoint(path, device)[0] if kind == "four"
               else load_checkpoint(path, device)[0])
        net.eval().requires_grad_(False)
        pool[code] = (net, float(frac))
        pool_names[code] = name
        pool_meta[name] = {"kind": kind, "path": path, "fraction": float(frac),
                           "sha256": sha256(path)}
    # E46 league (fictitious self-play, as harukaki/brl): a frozen copy of the actor that loads a
    # uniformly random earlier snapshot of THIS run before every rollout; share --league-frac.
    league = None
    if args.league_frac > 0:
        if not competitive:
            raise ValueError("--league-frac needs a D5OWN4XC actor (--redouble/--sacrifice)")
        league_net = copy.deepcopy(actor).eval().requires_grad_(False)
        code = max(pool, default=0) + 1
        league = {"net": league_net, "code": code, "rng": random.Random(args.seed + 9173),
                  "snapshots": [(0, {k: v.detach().clone() for k, v in actor.state_dict().items()})],
                  "pick": 0}
        pool[code] = (league_net, args.league_frac)
        pool_names[code] = "league"
        pool_meta["league"] = {"kind": "league", "fraction": args.league_frac,
                               "every": args.league_every, "max": args.league_max}
    stage = actor.stage
    objective, features = STAGE_TEXT[stage]
    head_groups = []
    for names, head_lr in ((("double_value_head", "double_gate_head"), args.double_value_lr),
                           (("redouble_value_head", "redouble_gate_head"), args.xx_value_lr),
                           (("sac_value_head", "sac_gate_head"), args.sac_value_lr)):
        params = [p for name in names if hasattr(actor, name)
                  for p in getattr(actor, name).parameters()]
        if params:
            head_groups.append({"params": params, "scale": head_lr / args.pg_lr})
    value_ids = {id(p) for group in head_groups for p in group["params"]}
    groups = [{"params": [p for p in actor.parameters() if id(p) not in value_ids], "scale": 1.0},
              *head_groups]
    actor_optimizer = torch.optim.AdamW(groups, lr=args.pg_lr, weight_decay=args.weight_decay)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr,
                                         weight_decay=args.weight_decay)
    head, belief_init = init_belief_head(args, actor, device)
    init_meta = {**init_meta, **({"belief_init": belief_init} if head is not None else {})}
    belief_optimizer = (torch.optim.AdamW(head.parameters(), lr=args.belief_lr,
                                          weight_decay=args.weight_decay)
                        if head is not None else None)
    silent_reference = silent_guard_reference(args)
    latest: dict = {}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    schedule = argparse.Namespace(lr=args.pg_lr, warmup=args.warmup, steps=args.steps,
                                  min_lr_frac=args.min_lr_frac)
    run_info = {**vars(args), "stage": stage, "objective": objective, "features": features,
                **init_meta, "discount": 1.0,
                "action_vocabulary": (("35 contracts + Pass + Double"
                                       + (" + Redouble" if args.redouble else "; no XX")
                                       + (" + Sacrifice gate over the cheapest bid per strain"
                                          if args.sacrifice else "") + " at all four seats")
                                      if competitive else
                                      "35 contracts + Pass + Double at all four seats; no XX"
                                      if args.doubles else
                                      "35 contracts + Pass at all four seats; no X/XX"),
                "absolute_ranges": {name: list(span) for name, span in spans.items()},
                "frozen_baselines": frozen, "repository": repository_state(),
                "opponent_pool": pool_meta,
                "episode_mix": ("self-play share = 1 - sum(pool fractions); silent_frac applies "
                                "only inside self-play; pool episodes: one random side frozen "
                                "(greedy), only learner decisions are trained"),
                "actor_parameters": sum(p.numel() for p in actor.parameters())}
    log_path = out / "train_log.jsonl"
    start = time.time()
    best = -math.inf
    best_own = -math.inf
    level5_start = math.nan
    first_step = 1
    if args.resume:
        state = torch.load(out / "last_state.pt", map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        # map_location moves the RNG ByteTensors to --device; set_state needs CPU tensors.
        generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        first_step = state["step"] + 1
        best = state.get("best", best)
        best_own = state.get("best_own", best_own)
        level5_start = state.get("level5_start", level5_start)
        if head is not None:
            head.load_state_dict(state["belief_head"])
            belief_optimizer.load_state_dict(state["belief_optimizer"])
        (out / f"run_resume_step{state['step']}.json").write_text(json.dumps(run_info, indent=2))
        print(f"RESUMED from step {state['step']} (best objective {best:+.2f})", flush=True)
    else:
        (out / "run.json").write_text(json.dumps(run_info, indent=2))
        log_path.write_text("")

    def save_state(step: int) -> None:
        save_fourseat_checkpoint(out / "last.pt", actor, critic, step=step, **extra,
                                 **belief_extra())
        torch.save({"step": step, "actor": actor.state_dict(), "critic": critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "generator": generator.get_state(), "torch_rng": torch.get_rng_state(),
                    "best": best, "best_own": best_own,
                    **({"level5_start": level5_start} if competitive else {}),
                    **({"belief_head": head.state_dict(),
                        "belief_optimizer": belief_optimizer.state_dict()} if head else {})},
                   out / "last_state.pt.tmp")
        (out / "last_state.pt.tmp").replace(out / "last_state.pt")

    def lr_at(step: int) -> float:
        if args.lr_schedule == "constant":
            return args.pg_lr * min(1.0, step / max(args.warmup, 1))
        return learning_rate(step, schedule)
    extra = {**init_meta, "args": vars(args), "any_seat_double": bool(args.any_seat_double)}

    def belief_extra() -> dict:
        return ({"belief_head": head.state_dict(), "belief_config": head.config,
                 "belief_mode": args.belief} if head is not None else {})

    def log_and_select(step: int, metrics: dict) -> dict:
        nonlocal best
        silent = silent_validation(actor, val, scorer)
        if competitive:
            four = competitive_validation(actor, val, scorer)
            vs = {pool_names[code]: competitive_validation(actor, val, scorer, frozen_net=net)
                  for code, (net, _) in pool.items() if pool_names[code] != "league"
                            and not getattr(net, "is_brl", False)}
        else:
            four = fourseat_validation(actor, val, scorer, args.doubles)
            vs = {pool_names[code]: fourseat_validation(actor, val, scorer, True, frozen_net=net)
                  for code, (net, _) in pool.items()}
        latest["silent"] = silent["score"]
        if step == 0 and isinstance(silent_reference, _OwnStepZero):
            silent_reference[0] = silent["score"]
        record = {"step": step, "seconds": round(time.time() - start, 1), **metrics,
                  "validation": {"silent": silent, "fourseat": four, "pool": vs}}
        if head is not None:
            from .train_belief import policy_belief_validation
            record["validation"]["belief"] = policy_belief_validation(actor, head, val,
                                                                      args.belief_val_deals)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        levels = four["level_share"]
        line = (f"step {step:>5} silent {silent['score']:+.1f} (pass {silent['passout']:.3f})"
                f" | 4seat own {four['own_score']:+.1f} ns/ew {four['own_ns']:+.1f}/"
                f"{four['own_ew']:+.1f} both-bid {four['both_sides_bid']:.2f} "
                f"calls {four['calls']:.1f} lvl1-3/4-5/6-7 "
                f"{levels['1'] + levels['2'] + levels['3']:.2f}/"
                f"{levels['4'] + levels['5']:.2f}/{levels['6'] + levels['7']:.2f}")
        if args.doubles:
            line += (f" | X rate {four['double_rate']:.3f} acc {four['double_accuracy']:.2f} "
                     f"delta/X {four['delta_per_double']:+.0f} pX(final) "
                     f"{four['p_double_final']:.3f} obj {four['objective']:+.1f}")
            if "double_value" in four:
                c = four["double_value"]
                line += (f" | qX mse {c['mse']:.2f} q>0 {c['q_positive_share']:.3f} "
                         f"acc {c['q_positive_accuracy']:.2f} top-decile pred/real "
                         f"{c['deciles'][-1]['pred']:+.2f}/{c['deciles'][-1]['real']:+.2f}")
        if competitive:
            line += competitive_line(four)
        for name, m in vs.items():
            line += (f" | vs {name}: X {m['double_rate']:.3f} acc {m['double_accuracy']:.2f} "
                     f"d/X {m['delta_per_double']:+.0f} own {m['learner_own_score']:+.1f} "
                     f"table {m['learner_table_score']:+.1f}")
            if competitive:
                line += (f" XX {m['redoubles']} SAC {m['sac_rate']:.4f} "
                         f"cr/SAC {m['sac_credit_per_sac']:+.0f}")
        if head is not None:
            b = record["validation"]["belief"]
            line += (f" | belief nll {b['nll']:.4f} partner bits {b['partner_bits']:.2f} "
                     f"ece {b['ece']:.3f}")
        print(line + f" [{record['seconds']:.0f}s]", flush=True)
        if four["objective"] > best:
            best = four["objective"]
            save_fourseat_checkpoint(out / "best.pt", actor, critic, step=step,
                                     val_fourseat=four, val_silent=silent, **extra,
                                     **belief_extra())
        return four

    def competitive_line(four: dict) -> str:
        spots = four["spots_per_1000"]
        text = (f" | lvl5+ {four['level5_share']:.3f} | spots/1k boards X {spots['x']:.0f} "
                f"XX {spots['xx']:.1f} SAC {spots['sac']:.0f}")
        if args.redouble:
            xv = four.get("redouble_value", {})
            text += (f" | XX rate {four['redouble_rate']:.3f} n {four['redoubles']} "
                     f"d/XX {four['delta_per_redouble']:+.0f} acc {four['redouble_accuracy']:.2f} "
                     f"qXX mse {xv.get('mse', math.nan):.2f} prof {xv.get('profitable_share', math.nan):.2f}")
        if args.sacrifice:
            sv = four.get("sac_value", {})
            text += (f" | SAC rate {four['sac_rate']:.4f} n {four['sacs']} "
                     f"cr/SAC {four['sac_credit_per_sac']:+.0f} pos {four['sac_positive_share']:.2f} "
                     f"lvl5 {four['sac_level5_share']:.2f} qSAC mse {sv.get('mse', math.nan):.2f} "
                     f"prof {sv.get('profitable_share', math.nan):.2f}")
        return text

    if not args.resume:
        start_four = log_and_select(0, {"phase": "warm_start"})
        best_own = start_four["own_score"]
        level5_start = start_four.get("level5_share", math.nan)
    # --fast-rollout: same policy and episode draws, different sampling stream (see fast_rollout.py);
    # a D5OWN4XC actor gets the competitive rollout (X/XX/SAC at pass-out seats, SAC fire record)
    if args.compile_rollout and not args.fast_rollout:
        raise ValueError("--compile-rollout needs --fast-rollout")
    collector = (FastCollector(actor, args.episodes, device, args.doubles,
                               args.behavior_temperature, pool, compile=args.compile_rollout)
                 if args.fast_rollout else None)
    stopped = None
    last_step = first_step - 1
    for step in range(first_step, args.steps + 1):
        last_step = step
        if blocks and (step - 1) // args.train_block_every != block["index"]:
            block["index"] = (step - 1) // args.train_block_every
            block["start"] = block_start(args, block["index"])
            train = None                      # release the previous block before loading
            train = load_range(args.data, block["start"], args.train_block_size, device)
            print(f"step {step}: training block {block['index']} = deals "
                  f"[{block['start']}, {block['start'] + args.train_block_size})", flush=True)
        for group in actor_optimizer.param_groups:
            group["lr"] = lr_at(step) * group["scale"]
        if league is not None:
            league["pick"], weights = league["rng"].choice(league["snapshots"])
            league["net"].load_state_dict(weights)
        if collector is not None:
            trajectories = collector.collect(train, generator, scorer, args.silent_frac)
        elif competitive:
            trajectories = collect_competitive_trajectories(
                actor, train, args.episodes, generator, scorer, args.behavior_temperature,
                args.silent_frac, device, pool)
        else:
            trajectories = collect_trajectories(
                actor, train, args.episodes, generator, scorer, args.behavior_temperature,
                args.silent_frac, device, args.doubles, pool)
        actor.train()
        critic.train()
        if competitive:
            losses = competitive_trajectory_losses(
                actor, critic, train, trajectories, args.entropy_weight,
                args.behavior_temperature, args.double_tau, args.xx_tau, args.sac_tau,
                policy_cf=step > args.double_policy_start, gate_pg=args.gate_pg,
                all_spots=args.all_spots, table_weight=table_weight_at(step, args))
        elif args.doubles:
            losses = double_trajectory_losses(
                actor, critic, train, trajectories, args.entropy_weight,
                args.behavior_temperature, args.double_cf_entropy, args.double_cf_nonfinal,
                args.double_tau, args.double_cf_final_only,
                policy_cf=step > args.double_policy_start)
        else:
            losses = trajectory_losses(actor, critic, train, trajectories, args.entropy_weight,
                                       args.behavior_temperature)
        actor_loss = (args.policy_weight * losses["policy_objective"]
                      + args.q_weight * losses["q_loss"]
                      + args.trick_weight * losses["trick_nll"])
        if args.doubles:
            if args.double_value:
                actor_loss = (actor_loss + args.double_value_weight * losses["double_value_loss"]
                              + args.double_cf_weight * losses["double_ce"])
            else:
                actor_loss = actor_loss + args.double_cf_weight * losses["double_cf_loss"]
        if competitive:
            if args.redouble:
                actor_loss = (actor_loss + args.xx_value_weight * losses["redouble_value_loss"]
                              + args.xx_cf_weight * losses["redouble_ce"])
            if args.sacrifice:
                actor_loss = (actor_loss + args.sac_value_weight * losses["sac_value_loss"]
                              + args.sac_cf_weight * losses["sac_ce"])
            if args.belief_summary:
                actor_loss = (actor_loss
                              + args.belief_summary_weight * losses["belief_summary_loss"])
        log_now = step == 1 or step % args.eval_every == 0 or step == args.steps
        belief_step: dict = {}
        belief = None
        if head is not None:
            from .train_belief import policy_belief_loss
            belief = policy_belief_loss(actor, head, train, trajectories.states, args.belief,
                                        args.belief_count_weight)
            if args.belief == "shared":
                actor_loss = actor_loss + args.belief_weight * belief["loss"]
            losses = {**losses, "belief_nll": belief["nll"],
                      "belief_count_loss": belief["count_loss"]}
            belief_optimizer.zero_grad(set_to_none=True)
        if log_now and args.log_shared_grad_norms:
            belief_step["shared_grad_norms"] = shared_grad_norms(actor, args, losses, belief)
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        losses["critic_loss"].backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip))
        critic_grad = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip))
        actor_optimizer.step()
        critic_optimizer.step()
        if head is not None:
            if args.belief == "detached":
                belief["loss"].backward()          # graph touches only the head
            for group in belief_optimizer.param_groups:
                group["lr"] = lr_at(step) / args.pg_lr * args.belief_lr
            # clipped on its own so the actor's clip scale is unchanged
            belief_step["belief_grad_norm"] = float(
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip))
            belief_optimizer.step()
        if log_now:
            four = log_and_select(step, {
                "phase": "on_policy", **({"league_size": len(league["snapshots"]), "league_opponent_step": league["pick"]} if league is not None else {}), "lr": actor_optimizer.param_groups[0]["lr"],
                "train_block_index": block["index"], "train_block_start": block["start"],
                "actor_loss": float(actor_loss.detach()), "actor_grad_norm": actor_grad,
                "critic_grad_norm": critic_grad, **trajectory_stats(trajectories),
                **(competitive_trajectory_stats(trajectories) if competitive else {}),
                "opponent_weight_norm": float(actor.auction_net[0].opponent.weight.detach().norm()),
                **{name: float(value.detach()) for name, value in losses.items()},
                **belief_step})
            if four.get("double_rate", 0.0) > args.max_double_rate:
                stopped = f"double rate {four['double_rate']:.3f} > {args.max_double_rate}"
            elif competitive and four["sac_rate"] > args.max_sac_rate:
                stopped = f"sacrifice rate {four['sac_rate']:.4f} > {args.max_sac_rate}"
            elif competitive and four["level5_share"] > level5_start + args.max_level5_rise:
                stopped = (f"level-5+ share {four['level5_share']:.3f} rose more than "
                           f"{args.max_level5_rise} above its start {level5_start:.3f}")
            elif four["own_score"] < args.min_own_score:
                stopped = f"own-bid score {four['own_score']:.1f} < {args.min_own_score}"
            elif four["own_score"] < best_own - args.max_own_drop:
                stopped = (f"own-bid score {four['own_score']:.1f} fell more than "
                           f"{args.max_own_drop} below its best {best_own:.1f}")
            elif (args.silent_guard is not None and silent_reference.get(step) is not None
                  and latest["silent"] < silent_reference.get(step) - args.silent_guard):
                stopped = (f"silent {latest['silent']:+.1f} < reference "
                           f"{silent_reference.get(step):+.1f} - {args.silent_guard}")
            best_own = max(best_own, four["own_score"])
            if stopped:
                print(f"EARLY STOP at step {step}: {stopped}", flush=True)
                break
        if league is not None and step % args.league_every == 0:
            league["snapshots"].append((step, {k: v.detach().clone() for k, v in actor.state_dict().items()}))
            if args.league_max and len(league["snapshots"]) > args.league_max:
                league["snapshots"].pop(0)
        if args.state_every and step % args.state_every == 0:
            save_state(step)
        if args.snapshot_every and step % args.snapshot_every == 0:
            snapshot = out / f"ckpt_step{step}.pt"
            save_fourseat_checkpoint(snapshot, actor, critic, step=step, **extra,
                                     **belief_extra())
            if args.match_opponent:
                launch_match(args, out, step, snapshot.resolve())

    save_state(last_step)

    net, meta = load_fourseat_checkpoint(out / "best.pt", device)
    report, arrays = evaluate_auctions(SilentView(net), held, scorer, baselines, args.seed,
                                       stage=stage)
    if competitive:
        held_four = competitive_validation(net, held, scorer)
        held_pool = {pool_names[code]: competitive_validation(net, held, scorer, frozen_net=pnet)
                     for code, (pnet, _) in pool.items() if pool_names[code] != "league"
                               and not getattr(pnet, "is_brl", False)}
    else:
        held_four = fourseat_validation(net, held, scorer, args.doubles)
        held_pool = {pool_names[code]: fourseat_validation(net, held, scorer, True, frozen_net=pnet)
                     for code, (pnet, _) in pool.items() if not getattr(pnet, "is_brl", False)}
    report.update(checkpoint=str(out / "best.pt"), checkpoint_step=meta["step"], discount=1.0,
                  early_stop=stopped, fourseat=held_four, fourseat_pool=held_pool)
    compare = {"init": Path(args.init).parent / "eval_rows.npz",
               **{k: Path(v) for k, v in (c.split("=", 1) for c in args.compare_rows)}}
    for name, rows_path in compare.items():
        if not rows_path.exists():
            continue
        base = np.load(rows_path)
        if all(k in base and np.array_equal(base[k], arrays[k])
               for k in ("deal_index", "side", "dealer", "vulnerable")):
            key = ("paired_silent_policy_lift_vs_init" if name == "init"
                   else f"paired_silent_policy_lift_vs_{name}")
            report[key] = paired_bootstrap(
                arrays["score_policy"] - base["score_policy"], arrays["deal_index"]
                - arrays["deal_index"].min(), seed=args.seed)
    (out / "eval.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(out / "eval_rows.npz", **arrays)
    print(f"\nHELD-OUT EVAL (best step {meta['step']})")
    print("best.pt holds the best four-seat VALIDATION objective, which is not the same as the "
          "strongest policy: rank this run's ckpt_step*.pt by a paired match before using one "
          "(experiments/LESSONS.md, experiments/match/watch_vs_fsp.py).")
    print("\n".join(summary_lines(report)))
    f = report["fourseat"]
    print(f"four-seat own-bid {f['own_score']:+.1f} (ns {f['own_ns']:+.1f} ew {f['own_ew']:+.1f})"
          f" both-bid {f['both_sides_bid']:.3f} calls {f['calls']:.2f}"
          + (f" X rate {f['double_rate']:.3f} acc {f['double_accuracy']:.2f} "
             f"delta/X {f['delta_per_double']:+.0f}" if args.doubles else "")
          + (competitive_line(f) if competitive else ""))
    for key in sorted(k for k in report if k.startswith("paired_silent_policy_lift_vs_")):
        lift = report[key]
        print(f"silent policy lift {key.removeprefix('paired_silent_policy_lift_')}: "
              f"{lift['mean']:+.2f} [{lift['ci95'][0]:+.2f}, {lift['ci95'][1]:+.2f}]", flush=True)
    for name, m in report["fourseat_pool"].items():
        print(f"vs {name}: X rate {m['double_rate']:.3f} acc {m['double_accuracy']:.2f} "
              f"delta/X {m['delta_per_double']:+.0f} learner own {m['learner_own_score']:+.1f} "
              f"table {m['learner_table_score']:+.1f}", flush=True)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", default=DEFAULT_INIT)
    parser.add_argument("--baseline", action="append", default=[])
    parser.add_argument("--train-start", type=int, default=2028000)
    parser.add_argument("--train-count", type=int, default=1000000)
    parser.add_argument("--val-start", type=int, default=3028000)
    parser.add_argument("--val-count", type=int, default=5000)
    parser.add_argument("--eval-start", type=int, default=-10000)
    parser.add_argument("--eval-count", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--silent-frac", type=float, default=0.25)
    parser.add_argument("--pg-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-lr-frac", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--trick-weight", type=float, default=0.2)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--behavior-temperature", type=float, default=1.0)
    parser.add_argument("--doubles", action="store_true", help="add Double (stage D5OWN4X)")
    parser.add_argument("--double-bias", type=float, default=-3.0,
                        help="initial Double policy logit bias")
    parser.add_argument("--double-cf-weight", type=float, default=1.0)
    parser.add_argument("--double-cf-entropy", type=float, default=0.01)
    parser.add_argument("--double-cf-nonfinal", action="store_true",
                        help="apply the counterfactual {Pass,X} loss at every legal-X decision")
    parser.add_argument("--double-value", action="store_true",
                        help="D5OWN4XV: double_value head + two-action CE toward it (implies --doubles)")
    parser.add_argument("--resume", action="store_true",
                        help="continue --out from its last_state.pt (same arguments otherwise)")
    parser.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine",
                        help="constant: linear warmup then --pg-lr forever")
    parser.add_argument("--train-block-every", type=int, default=0,
                        help=">0: rotate seeded training blocks every N steps (replaces --train-start)")
    parser.add_argument("--train-block-size", type=int, default=1000000)
    parser.add_argument("--train-pool-start", type=int, default=3033000)
    parser.add_argument("--train-pool-end", type=int, default=99990000)
    parser.add_argument("--state-every", type=int, default=1000,
                        help="write last.pt and last_state.pt every N steps (0: only at the end)")
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--match-opponent", default="",
                        help="four-seat checkpoint for a background IMP match at each snapshot")
    parser.add_argument("--match-deals", type=int, default=10000)
    parser.add_argument("--match-threads", type=int, default=2)
    parser.add_argument("--init-fourseat", action="store_true",
                        help="--init is a D5OWN4 checkpoint; warm start a D5OWN4XD net from it")
    parser.add_argument("--pool", action="append", default=[],
                        help="NAME=four|zero:PATH:FRACTION frozen opponent share of episodes")
    parser.add_argument("--compare-rows", action="append", default=[],
                        help="NAME=eval_rows.npz for an extra paired silent lift")
    parser.add_argument("--double-gate", action="store_true",
                        help="D5OWN4XD: X gate/value/Q read detached trunk features (implies --double-value)")
    parser.add_argument("--double-tau", type=float, default=0.5)
    parser.add_argument("--double-value-weight", type=float, default=0.5)
    parser.add_argument("--double-value-lr", type=float, default=1e-3)
    parser.add_argument("--double-policy-start", type=int, default=250,
                        help="value-mode: steps of value-only training before the policy CE")
    parser.add_argument("--double-cf-final-only", action="store_true",
                        help="value-mode: only spots where Pass would end the auction")
    parser.add_argument("--max-double-rate", type=float, default=0.40,
                        help="stop early when the val double rate exceeds this")
    parser.add_argument("--min-own-score", type=float, default=-math.inf,
                        help="stop early when the 4-seat own-bid val drops below this")
    parser.add_argument("--max-own-drop", type=float, default=math.inf,
                        help="stop early when the 4-seat own-bid val falls this far below its best")
    parser.add_argument("--fast-rollout", action="store_true",
                        help="fixed-shape rollout, CUDA graph on GPU, inverse-CDF sampling from "
                             "CPU-generator uniforms: same policy, different random stream")
    parser.add_argument("--compile-rollout", action="store_true",
                        help="with --fast-rollout: torch.compile the rollout round (~20%% faster "
                             "on GPU, slower start-up)")
    parser.add_argument("--redouble", action="store_true",
                        help="D5OWN4XC: add Redouble with a detached XX gate/value (implies the X gate)")
    parser.add_argument("--sacrifice", action="store_true",
                        help="D5OWN4XC: add the detached Sacrifice gate/value (implies the X gate)")
    parser.add_argument("--xx-bias", type=float, default=-6.0, help="initial XX gate bias (< 0)")
    parser.add_argument("--xx-tau", type=float, default=0.1)
    parser.add_argument("--xx-value-weight", type=float, default=0.5)
    parser.add_argument("--xx-value-lr", type=float, default=1e-3)
    parser.add_argument("--xx-cf-weight", type=float, default=1.0)
    parser.add_argument("--sac-bias", type=float, default=-6.0, help="initial SAC gate bias (< 0)")
    parser.add_argument("--sac-tau", type=float, default=0.5)
    parser.add_argument("--sac-value-weight", type=float, default=0.5)
    parser.add_argument("--sac-value-lr", type=float, default=1e-3)
    parser.add_argument("--sac-cf-weight", type=float, default=1.0)
    parser.add_argument("--gate-pg", action="store_true",
                        help="D5OWN4XC: let the policy gradient train the XX/SAC gate logits "
                             "(default: detached, gates learn only from their BCE)")
    parser.add_argument("--all-spots", action="store_true",
                        help="D5OWN4XC: XX/SAC value and gate losses at every spot (default: only "
                             "spots where every earlier opponent call was their greedy top call)")
    parser.add_argument("--league-frac", type=float, default=0.0,
                        help="E46: share of episodes against a random earlier snapshot of this run")
    parser.add_argument("--league-every", type=int, default=1000,
                        help="E46: add the actor to the league every N steps (step 0 = init)")
    parser.add_argument("--league-max", type=int, default=0,
                        help="E46: keep only the newest N league snapshots (0 = all)")
    parser.add_argument("--any-seat-double", action="store_true",
                        help="E45b: X/XX legal at every seat for every player (auction continues); "
                             "frozen competitive pool nets play their full policy")
    parser.add_argument("--table-weight", type=float, default=0.0,
                        help="E42: lambda of the real table result in the team return (D5OWN4XC only)")
    parser.add_argument("--table-weight-start", type=float, default=0.0,
                        help="E49: the lambda at step 1 when --table-weight-steps > 0")
    parser.add_argument("--table-weight-steps", type=int, default=0,
                        help="E49: ramp --table-weight-start to --table-weight over this many "
                             "steps (0: --table-weight applies from step 1, as before E49)")
    parser.add_argument("--cooperative", action="store_true",
                        help="cooperative stage only: forces --silent-frac 1.0 and --table-weight 0, "
                             "and refuses --pool/--league-frac. Without it the run is adversarial, "
                             "which is how the released model was trained")
    parser.add_argument("--max-sac-rate", type=float, default=0.20,
                        help="stop early when greedy val sacrifices per board exceed this")
    parser.add_argument("--max-level5-rise", type=float, default=0.05,
                        help="stop early when the val level-5+ contract share rises this much above step 0")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--belief", choices=("off", "detached", "shared"), default="off",
                        help="E27a: full-deal belief head on the actor trunk (detached reads "
                             "hidden.detach(); shared also trains the trunk)")
    parser.add_argument("--belief-weight", type=float, default=0.1,
                        help="shared mode: weight of the belief loss in the actor loss")
    parser.add_argument("--belief-width", type=int, default=512)
    parser.add_argument("--belief-count-weight", type=float, default=0.05)
    parser.add_argument("--belief-lr", type=float, default=1e-3)
    parser.add_argument("--belief-val-deals", type=int, default=256,
                        help="val deals (x16 greedy rows, half silent) for belief metrics")
    parser.add_argument("--belief-summary", action="store_true",
                        help="E54: bolt an HCP+length-per-suit belief head onto the actor "
                             "(not per-card) and feed its detached guess into the Q/policy "
                             "heads; supervised loss never reaches the guess through them")
    parser.add_argument("--belief-summary-weight", type=float, default=0.3,
                        help="weight of the belief-summary MSE loss in the actor loss")
    parser.add_argument("--silent-guard", type=float, default=None,
                        help="stop when silent val < reference silent at the same step - this")
    parser.add_argument("--silent-guard-reference", default="",
                        help="train_log.jsonl of the reference arm (default: own step 0)")
    parser.add_argument("--log-shared-grad-norms", action="store_true",
                        help="log each loss's gradient norm on the shared encoder at eval steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
