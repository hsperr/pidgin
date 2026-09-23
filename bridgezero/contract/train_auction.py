"""Stage D1/D2 trainer: cooperative silent-opponent auction prefixes.

D1 is endpoint grounding; D2 adds continuation roots finished by the frozen
target policy (see README.md). Example::

    OMP_NUM_THREADS=4 python -u -m bridgezero.contract.train_auction \\
        --data data/deals.npz --out runs/E4_stageD1_16k \\
        --baseline E3=runs/E3_stageC_16k
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import dataset_size, load_range, ranges_overlap, resolve_range
from .environment import CooperativeAuction
from .evaluate import imps_array
from .evaluate_auction import (
    RULES,
    auction_metrics,
    auction_rows,
    continue_auctions,
    evaluate_auctions,
    prefix_metrics,
    run_auctions,
    summary_lines,
    trajectory_belief_metrics,
    verify_frozen,
)
from .environment import N_COOP_ACTIONS
from .model import (
    Q_LOSSES,
    STAGE_POLICY_SOURCES,
    AuctionContractNet,
    BeliefAuctionNet,
    ResidualBeliefAuctionNet,
    default_policy_source,
    load_checkpoint,
    objective_for,
    save_checkpoint,
)
from .prefixes import (
    OBSERVATION_NOTES,
    OBSERVATIONS,
    CoopBatch,
    exact_endpoint,
    expected_endpoint,
    final_scores,
    net_outputs,
    observe,
    sample_prefixes,
)
from .targets import TARGET_SCALE, TorchScorer
from .train import gradient_norms, learning_rate

STAGE = "D1"
# stage -> validation/checkpoint-selection decision rule
SELECT_RULES = {"D1": "expected", "D2": "policy", "D4": "policy"}
# The D2 checkpoint objective names the greedy target-net policy, so it is the
# only continuation rule the trainer accepts.
CONTINUATION_RULES = ("policy", "q")


def state_digest(net) -> str:
    """Short sha256 of a network's weights, to identify a frozen target version."""
    h = hashlib.sha256()
    for name, tensor in net.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def continuation_values(target_net, deals, batch, scorer, rule: str = "policy",
                        observation: str = "intact") -> torch.Tensor:
    """``(B,36)`` exact partnership score after each legal root call (0 where illegal).

    Every legal call is applied to a copy of the root, then both active players
    finish the auction greedily with the frozen ``target_net`` and ``rule`` while
    opponents Pass. A call that ends the auction scores its endpoint.
    """
    legal = batch.legal()
    row, action = legal.nonzero(as_tuple=True)
    roots = batch.subset(row)
    roots.apply(action, torch.ones_like(row, dtype=torch.bool))
    done = continue_auctions(target_net, deals, roots, scorer, rule, observation=observation)
    values = torch.zeros(legal.shape, device=legal.device)
    values[row, action] = final_scores(done, deals, scorer)[0]
    return values


@torch.no_grad()
def signal_nats(listener, deals, batch, observation: str = "intact") -> torch.Tensor:
    """``(B,36)`` listener's partner-card BCE in nats, summed over its unknown cards.

    Every legal non-terminal root call is applied to a copy of the root; the next
    active player (the root actor's partner) predicts the root actor's hand from
    the new public auction. Lower is a more informative call. A Pass that ends the
    auction has no listener, so it receives the row's mean non-terminal cost and is
    therefore neutral under this auxiliary. Illegal slots are zero.
    """
    legal = batch.legal()
    row, action = legal.nonzero(as_tuple=True)
    forks = batch.subset(row)
    forks.apply(action, torch.ones_like(row, dtype=torch.bool))
    listener_hand = deals.hands[forks.deal, forks.actor_seat]
    speaker_hand = deals.hands[forks.deal, batch.actor_seat[row]]
    out = net_outputs(listener, deals, forks, observation=observation)
    card = F.binary_cross_entropy_with_logits(out["partner_belief_logits"], speaker_hand,
                                              reduction="none")
    nats = torch.zeros(legal.shape, device=legal.device)
    nats[row, action] = (card * (1.0 - listener_hand)).sum(-1)
    terminal_pass = torch.zeros_like(legal)
    terminal_pass[:, -1] = batch.pass_ends()
    heard = legal & ~terminal_pass
    neutral = ((nats * heard).sum(-1)
               / heard.sum(-1).clamp(min=1)).unsqueeze(-1)
    nats = torch.where(terminal_pass & legal, neutral, nats)
    return nats


def belief_losses(out, deals, batch) -> dict:
    """Partner-card BCE over unknown cards and the expected-13 count penalty."""
    hand = deals.hands[batch.deal, batch.actor_seat]
    partner = deals.hands[batch.deal, (batch.actor_seat + 2) % 4]
    unknown = 1.0 - hand
    card_loss = F.binary_cross_entropy_with_logits(
        out["partner_belief_logits"], partner, reduction="none")
    expected_count = (out["partner_probability"] * unknown).sum(-1)
    return {"belief_loss": (card_loss * unknown).sum() / unknown.sum().clamp(min=1),
            "belief_count_loss": ((expected_count - 13.0) ** 2).mean() / 169.0}


def d1_losses(net, target_net, deals, batch, scorer, policy_temperature: float,
              observation: str = "intact", continuation=None, policy_source: str = "expected",
              q_loss: str = "huber", signal=None, signal_weight: float = 0.0,
              stats: dict | None = None):
    """D1/D2/D4 losses.

    ``continuation=(rows, values)`` replaces the endpoint Q targets of ``rows``
    with continuation ``values`` (points). ``policy_source`` is ``expected``
    (D1: target-net expected endpoint), ``q`` (target-net contract Q), ``rollout``
    (softmax of this deal's exact values), or ``exact_pg`` (maximize the policy's
    expected exact value over all legal calls, plus temperature entropy and an
    optional ``signal`` (B,36) listener-nats cost weighted by ``signal_weight``).
    ``stats``, when given, receives detached diagnostics.
    """
    values, ceiling, rel = exact_endpoint(batch, deals, scorer)
    if continuation is not None:
        rows, cont = continuation
        values = torch.where(rows[:, None], cont, values)
    legal = batch.legal()
    q_target = (values - ceiling[:, None]) / TARGET_SCALE
    feats = observe(batch, observation)
    policy_target = None
    if policy_source != "exact_pg":
        with torch.no_grad():
            target_out = net_outputs(target_net, deals, batch, feats)
            if policy_source == "expected":
                target_probs = torch.softmax(target_out["trick_logits"], -1)
                target_values = expected_endpoint(target_probs, batch, scorer) / TARGET_SCALE
            elif policy_source == "q":
                target_values = target_out["contract_q"]
            elif policy_source == "rollout":
                target_values = q_target
            else:
                raise ValueError(f"unknown policy source {policy_source!r}")
            policy_target = torch.softmax(
                (target_values / policy_temperature).masked_fill(~legal, -torch.inf), -1)
    out = net_outputs(net, deals, batch, feats)
    log_policy = F.log_softmax(out["policy_logits"].masked_fill(~legal, -1e9), -1)
    policy = log_policy.exp()
    entropy = -(policy * log_policy.masked_fill(~legal, 0.0)).sum(-1)
    if policy_target is not None:
        policy_loss = -(policy_target * log_policy).sum(-1).mean()
    else:
        value = (policy * q_target.masked_fill(~legal, 0.0)).sum(-1)
        objective = value + policy_temperature * entropy
        if signal is not None and signal_weight:
            objective = objective - signal_weight * (policy * signal).sum(-1)
        policy_loss = -objective.mean()
    q_fn = {"huber": F.huber_loss, "mse": F.mse_loss}[q_loss]
    losses = {
        "trick_nll": F.cross_entropy(out["trick_logits"].reshape(-1, 14), rel.reshape(-1)),
        "q_loss": q_fn(out["contract_q"][legal], q_target[legal]),
        "policy_loss": policy_loss,
    }
    if "partner_belief_logits" in out:
        losses.update(belief_losses(out, deals, batch))
    if stats is not None:
        with torch.no_grad():
            error = (out["contract_q"] - q_target) * TARGET_SCALE
            pass_slot = torch.zeros_like(legal)
            pass_slot[:, -1] = True
            q_choice = out["contract_q"].masked_fill(~legal, -torch.inf).argmax(-1)
            pi_choice = log_policy.argmax(-1)
            best_value = q_target.masked_fill(~legal, -torch.inf).max(-1).values
            stats.update({
                "q_rmse_points": float(error[legal].pow(2).mean().sqrt()),
                "q_bias_points": float(error[legal].mean()),
                "q_pass_bias_points": float(error[legal & pass_slot].mean()),
                "q_bid_bias_points": float(error[legal & ~pass_slot].mean()),
                "policy_entropy_bits": float(entropy.mean() / math.log(2)),
                "policy_q_argmax_agree": float((q_choice == pi_choice).float().mean()),
                # Hindsight (per-deal) points left on the table by the policy's mean call.
                "policy_hindsight_regret_points": float(
                    ((best_value - (policy * q_target.masked_fill(~legal, 0)).sum(-1))
                     * TARGET_SCALE).mean()),
                "policy_pass_prob": float(policy[:, -1].mean()),
            })
            if signal is not None:
                stats["signal_nats_under_policy"] = float((policy * signal).sum(-1).mean())
                stats["signal_nats_spread"] = float(
                    (signal.masked_fill(~legal, -torch.inf).max(-1).values
                     - signal.masked_fill(~legal, torch.inf).min(-1).values).mean())
    return losses


def check_invariants(batch, deals, n: int = 64) -> int:
    """Replay prefixes through the reference four-seat wrapper; raise on violation.

    Always checks the real public features, whatever the observation regime.
    """
    legal = batch.legal()
    if legal.shape[1] != 36:
        raise AssertionError("cooperative legal mask must have exactly 36 slots")
    features = batch.features().cpu().numpy()
    for row, calls in enumerate(batch.call_lists()[:n]):
        game = CooperativeAuction.new(int(batch.dealer[row]), int(batch.side[row]),
                                      bool(batch.vul[row]))
        for call in calls:
            game = game.apply(call)
        if not game.inactive_calls_are_passes() or any(c >= 36 for c in game.state.calls):
            raise AssertionError("inactive seat bid or X/XX appeared in a prefix")
        if game.actor != int(batch.actor_seat[row]):
            raise AssertionError("batch actor differs from reference auction")
        mask = game.legal_mask()
        if mask[36:].any() or not np.array_equal(mask[:36], legal[row].cpu().numpy()):
            raise AssertionError("legal mask differs from reference auction")
        owners = deals.hands[int(batch.deal[row])].argmax(0).cpu().numpy()   # seat per card
        hand, feats = game.observation(owners)
        if not np.array_equal(feats, features[row]):
            raise AssertionError("observation features differ from reference auction")
    return min(n, len(batch))


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"output directory {out_dir} is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    baselines = {k: Path(v) for k, v in (b.split("=", 1) for b in args.baseline)}
    frozen = {k: verify_frozen(v) for k, v in baselines.items()}

    total = dataset_size(args.data)
    spans = {name: resolve_range(total, getattr(args, f"{name}_start"),
                                 getattr(args, f"{name}_count"))
             for name in ("train", "val", "eval")}
    for a in spans:
        for b in spans:
            if a < b and ranges_overlap(spans[a], spans[b]):
                raise ValueError(f"{a} range {spans[a]} overlaps {b} range {spans[b]}")
    for name, run_dir in baselines.items():
        base_run = json.loads((run_dir / "run.json").read_text())
        if base_run["absolute_ranges"]["eval"] != list(spans["eval"]):
            raise ValueError(f"baseline {name} used a different eval range")
        if ranges_overlap(tuple(base_run["absolute_ranges"]["train"]), spans["eval"]):
            raise ValueError(f"baseline {name} trained on the eval range")
    obs = args.observation
    stage = args.stage
    if stage == "D1" and args.continuation_frac:
        raise ValueError("continuation roots are a Stage D2 objective")
    if not 0 <= args.continuation_frac <= 1:
        raise ValueError("--continuation-frac must be in [0, 1]")
    if stage == "D2" and args.continuation_rule != "policy":
        raise ValueError("D2 continuation must use the greedy target-net policy")
    if stage == "D4":
        if args.architecture not in ("belief", "residual_belief"):
            raise ValueError("D4 requires a belief architecture")
        if args.continuation_frac != 1 or args.continuation_rule != "policy":
            raise ValueError("D4 requires --continuation-frac 1 --continuation-rule policy")
        if args.endpoint_warmup_steps < 1:
            raise ValueError("D4 requires a positive endpoint warm-up")
    elif args.architecture != "standard":
        raise ValueError("the belief architecture is reserved for D4")
    args.policy_source = args.policy_source or default_policy_source(stage)
    if args.signal_weight and (args.policy_source != "exact_pg" or stage != "D4"):
        raise ValueError("--signal-weight needs --stage D4 --policy-source exact_pg")
    objective = objective_for(stage, args.policy_source, args.q_loss)  # rejects a forbidden source
    policy_source, select_rule = args.policy_source, SELECT_RULES[stage]
    resume_state, resume_info = load_resume(args, device) if args.resume else (None, None)
    train = load_range(args.data, args.train_start, args.train_count, device)
    val = load_range(args.data, args.val_start, args.val_count, device)
    held = load_range(args.data, args.eval_start, args.eval_count, device)
    scorer = TorchScorer(device)
    architectures = {"standard": AuctionContractNet, "belief": BeliefAuctionNet,
                     "residual_belief": ResidualBeliefAuctionNet}
    model_class = architectures[args.architecture]
    model_args = ((args.width, args.suit_width, args.depth) if args.architecture == "standard"
                  else (args.width, args.suit_width, args.depth, args.belief_width))
    net = model_class(*model_args).to(device)
    init_info = None
    if args.init:
        init_path = Path(args.init)
        if not (init_path.parent / "FROZEN.sha256").exists():
            raise ValueError(f"init checkpoint run {init_path.parent} is not frozen")
        init_files = verify_frozen(init_path.parent)
        if init_path.name not in init_files:
            raise ValueError(f"init checkpoint {init_path} is not listed in its FROZEN.sha256")
        check_lineage_ranges(init_path.parent, spans)
        init_net, init_ckpt = load_checkpoint(init_path, device)
        if init_ckpt["observation"] != obs:
            raise ValueError("init checkpoint observation regime differs")
        if isinstance(net, ResidualBeliefAuctionNet) and isinstance(init_net, AuctionContractNet):
            net.load_base(init_net)
        else:
            if init_ckpt["model_config"] != net.config:
                raise ValueError("init checkpoint architecture differs")
            net.load_state_dict(init_net.state_dict())
        init_info = {"path": str(init_path), "sha256": init_files[init_path.name],
                     "stage": init_ckpt["stage"], "step": init_ckpt.get("step")}
    if args.freeze_base:
        if not isinstance(net, ResidualBeliefAuctionNet):
            raise ValueError("--freeze-base requires --architecture residual_belief")
        net.freeze_base()
    target_net = copy.deepcopy(net).eval()
    # Frozen target policy identity: version 0 is the init (or fresh) weights; each
    # refresh copies the learner after ``learner_step`` updates.
    target = {"version": 0, "learner_step": 0, "sha256_16": state_digest(target_net),
              "init_step": init_info["step"] if init_info else None}
    optimizer = torch.optim.AdamW(
        [parameter for parameter in net.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    first_step, samples, invariant_rows = 0, 0, 0
    depth_counts = np.zeros(args.max_depth + 1, dtype=np.int64)
    if resume_state is not None:
        net.load_state_dict(resume_state["net"])
        target_net.load_state_dict(resume_state["target_net"])
        optimizer.load_state_dict(resume_state["optimizer"])
        generator.set_state(resume_state["generator"])
        torch.set_rng_state(resume_state["torch_rng"])
        target = resume_state["target"]
        first_step, samples = resume_state["step"], resume_state["samples"]
        invariant_rows = resume_state["invariant_rows"]
        depth_counts = np.asarray(resume_state["depth_counts"], dtype=np.int64)

    run_info = {**vars(args), "stage": stage, "dataset_size": total,
                "observation_note": OBSERVATION_NOTES[obs],
                "absolute_ranges": {k: list(v) for k, v in spans.items()},
                "frozen_baselines": frozen,
                "parameters": sum(p.numel() for p in net.parameters())}
    if stage != "D1":
        run_info.update(objective=objective, policy_source=policy_source,
                        select_rule=select_rule, init_checkpoint=init_info)
    if resume_info:
        run_info.update(resume_checkpoint=resume_info, schedule_start=first_step)
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    best_score = -math.inf
    if resume_info:
        # Selection covers the whole lineage: the ancestor's best stays best.pt unless beaten.
        (out_dir / "best.pt").write_bytes((Path(args.resume) / "best.pt").read_bytes())
        best_score = resume_info["ancestral_best"]["val"]
    print(f"stage {stage} observation={obs} params={run_info['parameters']:,} "
          f"ranges={run_info['absolute_ranges']}"
          + (f" continuation_frac={args.continuation_frac} policy_source={policy_source} "
             f"init={init_info}" if stage != "D1" else "")
          + (f" resume={resume_info}" if resume_info else ""), flush=True)

    val_rows = auction_rows(val.n, device)
    # Same-size fixed slice of the training range: train-vs-val auction gap shows overfitting.
    train_probe = train.head(val.n)
    # Fixed policy-independent validation prefixes for Q fit (MSE and Pass/bid bias).
    q_probe, _ = sample_prefixes(val, args.val_q_prefixes, torch.Generator().manual_seed(
        args.seed + 13), max_depth=args.max_depth, device=device)
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")
    start = time.time()
    cont_stats = ContinuationStats()
    # Per-row validation scores of the first record: paired IMPs against the start.
    reference: dict = {}

    def log_and_select(step: int, record: dict, used_target: dict) -> None:
        nonlocal best_score
        record.update(validation(net, val, train_probe, val_rows, scorer, select_rule, obs,
                                 args.seed, q_probe=q_probe,
                                 continuation=stage in ("D2", "D4"), reference=reference))
        if stage != "D1":
            record["val_auction_rule"] = select_rule
            record["policy_source"] = policy_source
            # Target policy that produced this step's continuation and policy targets.
            record["target_policy"] = used_target
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(progress_line(record, select_rule), flush=True)
        if record["val_auction_score"] > best_score:
            best_score = record["val_auction_score"]
            save_checkpoint(out_dir / "best.pt", net, stage, policy_source, args.q_loss, step=step,
                            args=vars(args), val=record["val_auction_score"], observation=obs,
                            **checkpoint_extra(stage, select_rule, init_info, used_target))

    if init_info or resume_state is not None:
        # The warm-start weights, before any update, are eligible for selection.
        record = {"step": first_step, "seconds": 0.0, "samples": samples, "before_update": True}
        log_and_select(first_step, record, dict(target))
    for step in range(first_step + 1, args.steps + 1):
        belief_pretraining = stage == "D4" and step <= args.belief_pretrain_steps
        step_batch = (args.belief_pretrain_batch if belief_pretraining
                      and args.belief_pretrain_batch else args.batch)
        for group in optimizer.param_groups:
            group["lr"] = (args.belief_pretrain_lr if belief_pretraining else
                           learning_rate(step, args, max(first_step, args.belief_pretrain_steps)))
        batch, info = sample_prefixes(
            train, step_batch, generator, online=net.eval(), target=target_net,
            max_depth=args.max_depth, opening_prob=args.opening_prob, window=args.window,
            epsilon=args.epsilon, temperature=args.behavior_temperature, device=device,
            observation=obs)
        depth_counts += np.bincount(info["depth"].cpu().numpy(), minlength=args.max_depth + 1)
        continuation = None
        warming_up = stage == "D4" and step <= args.endpoint_warmup_steps
        n_cont = 0 if warming_up else round(len(batch) * args.continuation_frac)
        if belief_pretraining:
            n_cont = 0
        if n_cont:
            # Prefix rows are i.i.d., so the first n_cont rows are a random subset.
            rows = torch.arange(len(batch), device=device) < n_cont
            cont = torch.zeros(len(batch), N_COOP_ACTIONS, device=device)
            cont[:n_cont] = continuation_values(target_net, train, batch.subset(rows), scorer,
                                                args.continuation_rule, obs)
            continuation = (rows, cont)
            cont_stats.add(cont[rows], exact_endpoint(batch.subset(rows), train, scorer)[0],
                           batch.subset(rows).legal(), target)
        net.train()
        logging = step == first_step + 1 or step % args.eval_every == 0 or step == args.steps
        batch_stats: dict = {}
        if belief_pretraining:
            # Only partner reconstruction trains here; skip target, endpoint, and value heads.
            losses = belief_losses(net_outputs(net, train, batch, observation=obs), train, batch)
            loss = losses["belief_loss"] + args.belief_count_weight * losses["belief_count_loss"]
        else:
            # A listener that has not learned yet rewards noise, so signal credit waits
            # for the end of endpoint warm-up.
            signal = (signal_nats(target_net, train, batch, obs)
                      if args.signal_weight and not warming_up else None)
            losses = d1_losses(net, target_net, train, batch, scorer, args.policy_temperature,
                               obs, continuation, policy_source, args.q_loss, signal,
                               args.signal_weight, batch_stats if logging else None)
            loss = (args.trick_weight * losses["trick_nll"] + args.q_weight * losses["q_loss"]
                    + args.policy_weight * losses["policy_loss"])
            if "belief_loss" in losses:
                loss = (loss + args.belief_weight * losses["belief_loss"]
                        + args.belief_count_weight * losses["belief_count_loss"])
        norms = gradient_norms(net, losses) if logging else {}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        total_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip))
        optimizer.step()
        samples += len(batch)
        used_target = dict(target)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(out_dir / "snapshots" / f"step_{step}.pt", net, stage, policy_source,
                            args.q_loss, step=step, args=vars(args), observation=obs,
                            **checkpoint_extra(stage, select_rule, init_info, used_target))
        if step % args.target_every == 0:
            target_net.load_state_dict(net.state_dict())
            target = {**target, "version": target["version"] + 1, "learner_step": step,
                      "sha256_16": state_digest(target_net)}

        if logging:
            invariant_rows += check_invariants(batch, train)
            record = {
                "step": step, "seconds": round(time.time() - start, 1),
                "lr": optimizer.param_groups[0]["lr"], "loss": float(loss.detach()),
                **{k: float(v.detach()) for k, v in losses.items()},
                "grad_norm": total_norm, "grad_clipped": total_norm > args.grad_clip,
                "unweighted_shared_grad_norm": norms, "samples": samples,
                "depth_share": (depth_counts / depth_counts.sum()).round(4).tolist(),
                "invariant_rows_checked": invariant_rows,
                "phase": "belief_pretrain" if belief_pretraining else "joint",
                "train_batch": batch_stats,
            }
            if continuation is not None:
                record["continuation_roots"] = int(continuation[0].sum())
                record["continuation_since_last_log"] = cont_stats.flush()
            log_and_select(step, record, used_target)
    extra = checkpoint_extra(stage, select_rule, init_info, used_target)
    save_checkpoint(out_dir / "last.pt", net, stage, policy_source, args.q_loss, step=args.steps,
                    args=vars(args), observation=obs, **extra)
    # Everything needed to extend this run with a new learning-rate segment (--resume).
    torch.save({"step": args.steps, "args": vars(args), "net": net.state_dict(),
                "target_net": target_net.state_dict(), "target": target,
                "optimizer": optimizer.state_dict(), "generator": generator.get_state(),
                "torch_rng": torch.get_rng_state(), "samples": samples,
                "invariant_rows": invariant_rows, "depth_counts": depth_counts.tolist()},
               out_dir / "last_state.pt")

    for name, prefix, label in (("best.pt", "eval", "best val step"),
                                ("last.pt", "last_eval", "DIAGNOSTIC final step")):
        ckpt_net, meta = load_checkpoint(out_dir / name, device, stage)
        result, arrays = evaluate_auctions(ckpt_net, held, scorer, baselines, args.seed,
                                           observation=meta["observation"], stage=stage)
        result["checkpoint"] = str(out_dir / name)
        result["checkpoint_step"] = meta["step"]
        result["policy_source"] = meta.get("policy_source")
        (out_dir / f"{prefix}.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(out_dir / f"{prefix}_rows.npz", **arrays)
        print(f"\nHELD-OUT AUCTION EVAL ({label} {meta['step']}, observation={obs})")
        print("\n".join(summary_lines(result)), flush=True)
        if name == "best.pt":
            report = result
    return report


@torch.no_grad()
def validation(net, val, train_probe, val_rows, scorer, select_rule: str, observation: str,
               seed: int, q_probe=None, continuation: bool = False,
               reference: dict | None = None) -> dict:
    """Validation auctions under every decision rule; top-level keys use ``select_rule``.

    ``val_key`` holds the headline numbers of the selection rule: mean score, gap to
    the cooperative DD par (points and IMPs), and paired points/IMPs against the
    first validation record (``reference`` keeps its per-row scores).
    """
    by_rule = {}
    states: list[CoopBatch] = []
    belief = hasattr(net, "belief_head")
    for rule in RULES:
        record = states if belief and rule == select_rule else None
        auctions = continue_auctions(net, val, CoopBatch.start(*val_rows), scorer, rule,
                                     observation=observation, record=record)
        score, ceiling, declarer = final_scores(auctions, val, scorer)
        by_rule[rule] = {"score": float(score.mean()), "regret": float((ceiling - score).mean()),
                         "passout": float((auctions.last < 0).float().mean()),
                         "decisions": float(auctions.k.float().mean())}
        if rule == select_rule:
            chosen_scores = score.cpu().numpy()
            metrics = auction_metrics(auctions, chosen_scores, ceiling.cpu().numpy(),
                                      declarer.cpu().numpy(), val)
    train_auctions = run_auctions(net, train_probe, val_rows, scorer, select_rule,
                                  observation=observation)
    chosen = by_rule[select_rule]
    key = {"score": metrics["mean_score"], "gap_to_dd_par": metrics["mean_regret"],
           "imp_gap_to_dd_par": metrics["mean_imp_regret"], "make_rate": metrics["make_rate"],
           "passout": metrics["passout_rate"], "decisions": metrics["active_decisions"],
           "within_100": metrics["within_100"]}
    if reference is not None:
        reference.setdefault("scores", chosen_scores)
        diff = chosen_scores - reference["scores"]
        key["points_vs_start"] = float(diff.mean())
        key["imps_vs_start"] = float(imps_array(diff).mean())
    out = {
        "val_key": key,
        "val_auction_score": chosen["score"], "val_auction_regret": chosen["regret"],
        "val_auction_passout": chosen["passout"], "val_auction_decisions": chosen["decisions"],
        "val_auction_by_rule": by_rule,
        "val_auction_metrics": metrics,
        "val_q_minus_expected_score": by_rule["q"]["score"] - by_rule["expected"]["score"],
        "train_probe_auction_score": float(
            final_scores(train_auctions, train_probe, scorer)[0].mean()),
        "val_prefix": prefix_metrics(net, val, scorer, 8000, seed + 7, observation),
    }
    if q_probe is not None:
        out["val_q_fit"] = q_fit(net, val, q_probe, scorer, observation, continuation)
    if states:
        by_depth = trajectory_belief_metrics(net, val, states, observation)
        out["val_belief"] = {regime: belief_summary(depths) for regime, depths in by_depth.items()}
        out["val_belief"]["by_depth"] = by_depth
    return out


def belief_summary(depths: dict) -> dict:
    """Row-weighted mean of per-depth on-policy belief metrics."""
    rows = sum(m["rows"] for m in depths.values())
    return {name: sum(m[name] * m["rows"] for m in depths.values()) / max(rows, 1)
            for name in ("bce", "top13_recall", "expected_count")}


@torch.no_grad()
def q_fit(net, deals, batch, scorer, observation: str, continuation: bool) -> dict:
    """Q error on fixed prefixes against endpoint and (optionally) continuation targets.

    MSE is in the training units ((points / 100) squared); RMSE and biases are points.
    Pass and bid biases separate the Pass-to-bid offset from the spread of bids.
    The continuation target finishes every legal call with the current greedy policy.
    """
    was_training = net.training
    net.eval()
    endpoint, ceiling, _ = exact_endpoint(batch, deals, scorer)
    legal = batch.legal()
    out = net_outputs(net, deals, batch, observation=observation)
    q = out["contract_q"] * TARGET_SCALE
    pass_slot = torch.zeros_like(legal)
    pass_slot[:, -1] = True
    targets = {"endpoint": endpoint}
    if continuation:
        targets["continuation"] = continuation_values(net, deals, batch, scorer, "policy",
                                                      observation)
    result = {}
    for name, values in targets.items():
        error = q - (values - ceiling[:, None])
        result[name] = {
            "q_mse": float((error / TARGET_SCALE)[legal].pow(2).mean()),
            "q_rmse_points": float(error[legal].pow(2).mean().sqrt()),
            "q_bias_points": float(error[legal].mean()),
            "pass_bias_points": float(error[legal & pass_slot].mean()),
            "bid_bias_points": float(error[legal & ~pass_slot].mean()),
        }
    expected = expected_endpoint(torch.softmax(out["trick_logits"], -1), batch, scorer)
    result["endpoint"]["expected_rule_rmse_points"] = float(
        (expected - endpoint)[legal].pow(2).mean().sqrt())
    q_choice = out["contract_q"].masked_fill(~legal, -torch.inf).argmax(-1)
    pi_choice = out["policy_logits"].masked_fill(~legal, -torch.inf).argmax(-1)
    result["policy_q_argmax_agree"] = float((q_choice == pi_choice).float().mean())
    net.train(was_training)
    return result


def progress_line(record: dict, rule: str) -> str:
    """One console line with the headline training and validation numbers."""
    key = record["val_key"]
    head = f"step {record['step']:>6} {record['seconds']:>6.1f}s "
    if "loss" in record:
        head += f"[{record['phase']}] loss {record['loss']:.3f}"
        if "q_loss" in record:
            head += f" q {record['q_loss']:.3f} pi {record['policy_loss']:+.3f}"
        if "belief_loss" in record:
            head += f" belief {record['belief_loss']:.4f}"
    else:
        head += "[before update]"
    parts = [head,
             f"val {rule}: score {key['score']:+.1f} gap {key['gap_to_dd_par']:.1f} "
             f"IMPgap {key['imp_gap_to_dd_par']:.2f} vs start {key.get('points_vs_start', 0):+.1f}pts "
             f"{key.get('imps_vs_start', 0):+.3f}IMP pass {key['passout']:.3f} "
             f"make {key['make_rate']:.3f} dec {key['decisions']:.2f}",
             "E/Q/P " + "/".join(f"{record['val_auction_by_rule'][r]['score']:+.1f}"
                                 for r in RULES)]
    if "val_q_fit" in record:
        fit = record["val_q_fit"]
        target = fit.get("continuation", fit["endpoint"])
        parts.append(f"Q rmse {target['q_rmse_points']:.0f} bias pass/bid "
                     f"{target['pass_bias_points']:+.0f}/{target['bid_bias_points']:+.0f} "
                     f"pi=Q {fit['policy_q_argmax_agree']:.2f}")
    if "val_belief" in record:
        belief = record["val_belief"]
        parts.append(f"belief top13 {belief['intact']['top13_recall']:.3f} "
                     f"(masked {belief['partner_masked']['top13_recall']:.3f})")
    return " | ".join(parts)


def check_lineage_ranges(run_dir: Path, spans: dict) -> None:
    """Raise if an init ancestor trained on this run's validation or eval deals."""
    info = json.loads((run_dir / "run.json").read_text())
    trained = info.get("absolute_ranges", {}).get("train")
    if trained is None:
        raise ValueError(f"{run_dir / 'run.json'} records no absolute train range")
    for name in ("val", "eval"):
        if ranges_overlap(tuple(trained), spans[name]):
            raise ValueError(f"init lineage {run_dir} trained on this run's {name} range")
    if info.get("init"):
        check_lineage_ranges(Path(info["init"]).parent, spans)


# Options a resumed run may change: output, schedule segment, logging cadence, threads.
RESUME_FREE_KEYS = {"out", "steps", "lr", "warmup", "min_lr_frac", "eval_every", "threads", "resume"}


def load_resume(args: argparse.Namespace, device) -> tuple[dict, dict]:
    """Final training state of a frozen run whose options match ``args``."""
    run_dir = Path(args.resume)
    if not (run_dir / "FROZEN.sha256").exists():
        raise ValueError(f"resume run {run_dir} is not frozen")
    files = verify_frozen(run_dir)
    for name in ("run.json", "best.pt", "last.pt", "last_state.pt"):
        if name not in files:
            raise ValueError(f"{run_dir / name} is not listed in the run's FROZEN.sha256")
    previous = json.loads((run_dir / "run.json").read_text())
    # The extension continues the ancestor's --init lineage; naming another init is an error.
    args.init = args.init or previous.get("init", "")
    differ = args_differ(previous, vars(args), RESUME_FREE_KEYS)
    if differ:
        raise ValueError(f"resumed run options differ from {run_dir}: {differ}")
    # RNG states must stay CPU ByteTensors; weights and optimizer state move on load.
    state = torch.load(run_dir / "last_state.pt", map_location="cpu", weights_only=False)
    if state["step"] != previous["steps"] or args.steps <= state["step"]:
        raise ValueError(f"--steps {args.steps} must extend {run_dir} beyond step {state['step']}")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    return state, {"run": str(run_dir), "step": state["step"],
                   "last_state_sha256": files["last_state.pt"], "last_sha256": files["last.pt"],
                   "ancestral_best": {"step": best["step"], "val": best["val"],
                                      "sha256": files["best.pt"]},
                   "previous_resume": previous.get("resume_checkpoint")}


def args_differ(a: dict, b: dict, free: set[str]) -> list[str]:
    """Trainer options (after ``normalize_run_args``) that differ outside ``free``."""
    a, b = normalize_run_args(a), normalize_run_args(b)
    options = {x.dest for x in build_parser()._actions if x.dest != "help"}
    return sorted(k for k in options if k not in free and a.get(k) != b.get(k))


def checkpoint_extra(stage: str, select_rule: str, init_info, target: dict) -> dict:
    if stage == "D1":
        return {}
    return {"val_rule": select_rule, "init": init_info, "target_policy": target}


class ContinuationStats:
    """Continuation-versus-endpoint target statistics accumulated between log records."""

    def __init__(self):
        self.flush()

    def add(self, cont: torch.Tensor, endpoint: torch.Tensor, legal: torch.Tensor,
            target: dict) -> None:
        masked = lambda v: v.masked_fill(~legal, -torch.inf)
        best_cont, best_end = masked(cont).max(1), masked(endpoint).max(1)
        self.roots += len(cont)
        self.actions += int(legal.sum())
        self.diff_sum += float((cont - endpoint)[legal].sum())
        self.differs += int((cont != endpoint)[legal].sum())
        self.argmax_differs += int((best_cont.indices != best_end.indices).sum())
        self.best_gain_sum += float((best_cont.values - best_end.values).sum())
        key = f"{target['version']}@{target['learner_step']}"
        self.by_target[key] = self.by_target.get(key, 0) + len(cont)

    def flush(self) -> dict:
        out = {}
        if getattr(self, "roots", 0):
            out = {"roots": self.roots, "legal_actions": self.actions,
                   "continuation_minus_endpoint_points": self.diff_sum / self.actions,
                   "continuation_differs_share": self.differs / self.actions,
                   "best_call_differs_share": self.argmax_differs / self.roots,
                   "best_continuation_minus_best_endpoint_points": self.best_gain_sum / self.roots,
                   "roots_by_target_version_at_learner_step": self.by_target}
        self.roots = self.actions = self.differs = self.argmax_differs = 0
        self.diff_sum = self.best_gain_sum = 0.0
        self.by_target = {}
        return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline", action="append", default=[],
                        help="NAME=RUN_DIR of a frozen Stage C run (read-only)")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=16000)
    parser.add_argument("--val-start", type=int, default=16000)
    parser.add_argument("--val-count", type=int, default=2000)
    parser.add_argument("--eval-start", type=int, default=-10000)
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
    parser.add_argument("--policy-weight", type=float, default=0.5)
    parser.add_argument("--belief-weight", type=float, default=0.2)
    parser.add_argument("--belief-count-weight", type=float, default=0.05)
    parser.add_argument("--belief-pretrain-steps", type=int, default=0,
                        help="D4 steps that train only partner reconstruction")
    parser.add_argument("--belief-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--belief-pretrain-batch", type=int, default=0,
                        help="optional larger D4 reconstruction-only batch")
    parser.add_argument("--policy-temperature", type=float, default=0.5,
                        help="softmax temperature in units of 100 points")
    parser.add_argument("--q-loss", choices=Q_LOSSES, default="huber",
                        help="Q regression loss; Huber (delta 100 points) estimates a "
                             "median-like value and overvalues Pass, MSE the mean")
    parser.add_argument("--signal-weight", type=float, default=0.0,
                        help="D4 exact_pg: points/100 per nat of partner-card uncertainty "
                             "left in the listener after each call")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also keep snapshots/step_N.pt every N steps (0: off)")
    parser.add_argument("--val-q-prefixes", type=int, default=2048,
                        help="fixed validation prefixes for Q MSE and Pass/bid bias")
    parser.add_argument("--target-every", type=int, default=250)
    parser.add_argument("--endpoint-warmup-steps", type=int, default=0,
                        help="D4 endpoint-grounding steps before full rollout targets")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--opening-prob", type=float, default=0.25)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--behavior-temperature", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--suit-width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--architecture", choices=("standard", "belief", "residual_belief"),
                        default="standard")
    parser.add_argument("--belief-width", type=int, default=192)
    parser.add_argument("--freeze-base", action="store_true",
                        help="train only belief/residual modules of residual_belief")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--observation", choices=tuple(OBSERVATIONS), default="intact",
                        help="auction features every network input sees")
    parser.add_argument("--stage", choices=tuple(SELECT_RULES), default=STAGE)
    parser.add_argument("--init", default="",
                        help="frozen checkpoint to start from (its run dir needs FROZEN.sha256)")
    parser.add_argument("--continuation-frac", type=float, default=0.0,
                        help="share of prefix roots with continuation Q targets; D4 requires 1")
    parser.add_argument("--continuation-rule", choices=CONTINUATION_RULES,
                        default="policy", help="greedy rule the frozen target net finishes with")
    parser.add_argument("--policy-source", default="",
                        choices=("", *sorted({s for v in STAGE_POLICY_SOURCES.values() for s in v})),
                        help="target-net values the policy is distilled from; empty is the stage "
                             "default (D1 expected, D2 q, D4 exact rollout). D2 also allows expected")
    parser.add_argument("--resume", default="",
                        help="frozen run dir to extend from its last_state.pt with a new "
                             "learning-rate segment (warmup + cosine to --steps)")
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def normalize_run_args(info: dict) -> dict:
    """``run.json`` with options added after it was written filled with their defaults.

    Every option added to this trainer defaults to the behavior that existed
    before it, so a legacy run (for example frozen E4, written before
    ``--observation`` and the D2 options) ran with those defaults. Keys present
    in ``info`` are never changed, except an empty ``policy_source`` (the stage
    default), which is filled with that default.
    """
    parser = build_parser()
    defaults = {a.dest: a.default for a in parser._actions
                if a.dest != "help" and not a.required}
    out = {**defaults, **info}
    out.setdefault("observation_note", OBSERVATION_NOTES[out["observation"]])
    out["policy_source"] = out["policy_source"] or default_policy_source(out.get("stage", STAGE))
    return out


if __name__ == "__main__":
    run(parse_args())
