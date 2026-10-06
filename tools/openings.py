"""Print the opening bids a four-seat model makes.

An opening call sees only the opener's hand, the seat (how many passes came
before), and its side's vulnerability, so no auction has to be played: each
held-out hand is asked directly. The model plays its greedy call, as in a match.

    python tools/openings.py runs/training/4_D/best.pt
    python tools/openings.py CKPT --seat 3 --vul --deals 50000 --examples 5

``brl:WEIGHTS.npz`` loads the brl port from ``brl_player.py`` next to the weights file
(an outside bot, for comparison only; never training data).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.bridge.calls import ACTION_NAMES, PASS, STRAIN_PERM  # noqa: E402
from training.contract.data import load_range  # noqa: E402
from training.fourseat.competitive import apply_opening_rule, competitive_features  # noqa: E402
from training.fourseat.model import load_fourseat_checkpoint, policy_log_probs  # noqa: E402
from training.fourseat.state import features_from_history  # noqa: E402
from training.simplicity import BALANCED_SHAPES, LIGHT_HCP, opening_numbers  # noqa: E402

SYMBOL = {"S": "♠", "H": "♥", "D": "♦", "C": "♣", "NT": "NT"}
CARD_SUITS = "SHDC"                                   # hand layout: suit * 13 + rank
RANKS = "AKQJT98765432"
BALANCED = BALANCED_SHAPES


def call_name(action: int) -> str:
    name = ACTION_NAMES[action]
    if action == PASS:
        return "Pass"
    level, strain = name[0], name[1:]
    return level + SYMBOL[strain]


def hand_text(hand: np.ndarray) -> str:
    cards = hand.reshape(4, 13)
    return " ".join(SYMBOL[CARD_SUITS[s]] + ("".join(RANKS[r] for r in range(13) if cards[s, r])
                                             or "-") for s in range(4))


@torch.no_grad()
def opening_probs(net, hands: torch.Tensor, seat: int, vul: bool, rule: int = 0) -> np.ndarray:
    """``(n, 36)`` policy over the 35 bids and Pass after ``seat - 1`` passes."""
    n = len(hands)
    history = torch.full((n, seat - 1), PASS, dtype=torch.long)
    dealer = torch.zeros(n, dtype=torch.long)
    actor = torch.full((n,), seat - 1, dtype=torch.long)
    v = torch.full((n,), vul)
    if getattr(net, "redouble", None) is not None:              # D5OWN4XC
        feats = competitive_features(history, dealer, v, v, actor)
    else:
        feats = features_from_history(history, dealer, v, v, actor, net.n_actions > PASS + 1)
    out = []
    for i in range(0, n, 20000):
        o = net(hands[i:i + 20000], feats[i:i + 20000])
        legal = torch.zeros(len(o["policy_logits"]), net.n_actions, dtype=torch.bool)
        legal[:, :PASS + 1] = True                              # X/XX/SAC are illegal here
        legal = apply_opening_rule(legal, hands[i:i + 20000], torch.full((len(legal),), -1),
                                   seat - 1, rule)
        out.append(policy_log_probs(o, legal).exp()[:, :PASS + 1])
    return torch.cat(out).numpy()


def load_opener(spec: str):
    """``(probs(hands, seat, vul) -> (n, 36), meta)`` for a checkpoint or ``brl:WEIGHTS.npz``."""
    if spec.startswith("brl:"):
        weights = Path(spec[4:]).resolve()
        sys.path[:0] = [str(weights.parent), str(ROOT / "server" / "emergent")]
        from brl_player import _OURS_TO_PGX, BrlNet, encode, legal_pgx

        brl = BrlNet(weights).eval()

        @torch.no_grad()
        def brl_probs(hands: torch.Tensor, seat: int, vul: bool) -> np.ndarray:
            out = []
            for i in range(0, len(hands), 20000):
                h = hands[i:i + 20000]
                n = len(h)
                v = torch.full((n,), vul)
                seat_t = torch.full((n,), seat - 1, dtype=torch.long)
                obs = encode(h, torch.full((n, seat - 1), PASS, dtype=torch.long),
                             torch.zeros(n, dtype=torch.long), v, v, seat_t)
                legal = legal_pgx(torch.full((n,), -1), torch.zeros(n), torch.zeros(n), seat_t)
                logp = torch.log_softmax(brl(obs).masked_fill(~legal, -torch.inf), -1)
                out.append(logp[:, _OURS_TO_PGX][:, :PASS + 1].exp())
            return torch.cat(out).numpy()

        return brl_probs, {"stage": "brl (outside bot, diagnostic)", "step": None}
    net, ck = load_fourseat_checkpoint(spec)
    net.eval()
    rule = int(ck.get("opening_rule", 0))
    return (lambda hands, seat, vul: opening_probs(net, hands, seat, vul, rule)), ck


def hand_features(hands: np.ndarray) -> dict:
    cards = hands.reshape(-1, 4, 13).astype(int)
    lengths = cards.sum(2)
    shape = np.array(["".join(map(str, row)) for row in -np.sort(-lengths, 1)])
    return {"hcp": (cards[:, :, :4] * np.array([4, 3, 2, 1])).sum((1, 2)),
            "len": lengths, "shape": shape, "bal": np.isin(shape, BALANCED)}


def suit_length(f: dict, action: int) -> np.ndarray | None:
    strain = action % 5
    return None if strain == 4 else f["len"][:, STRAIN_PERM[strain]]


def pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", help="four-seat checkpoint, or brl:WEIGHTS.npz")
    p.add_argument("--data", default="data/dds_results_100M.npy")
    p.add_argument("--deals", type=int, default=25000, help="last N (held-out) deals, 4 hands each")
    p.add_argument("--seat", type=int, default=1, choices=(1, 2, 3, 4),
                   help="seat of the detailed tables (1 = dealer)")
    p.add_argument("--vul", action="store_true", help="detailed tables when vulnerable")
    p.add_argument("--examples", type=int, default=3, help="example hands per call")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    opener, ck = load_opener(args.checkpoint)
    deals = load_range(args.data, -args.deals, args.deals)
    hands_t = deals.hands.reshape(-1, 52)
    hands = hands_t.numpy().astype(np.uint8)
    f = hand_features(hands)
    hcp = f["hcp"]
    print(f"Opening bids of {args.checkpoint}  (stage {ck.get('stage')}, step {ck.get('step')})")
    print(f"{len(hands):,} held-out hands; the model's greedy call. "
          "Seat 1 = dealer, seat 3 = after two passes.\n")

    # -- how often it opens, every seat and vulnerability
    greedy = {}
    print("HOW OFTEN IT OPENS   (half of the hands with this many HCP or more open)")
    print("          not vulnerable        vulnerable")
    for seat in range(1, 5):
        cells = []
        for vul in (False, True):
            a = opener(hands_t, seat, vul).argmax(1)
            greedy[seat, vul] = a
            opens = a != PASS
            point = next((x for x in range(38) if (hcp == x).any() and opens[hcp == x].mean() >= .5),
                         None)
            cells.append(f"{pct(opens.mean())}  ({point} HCP)")
        print(f"  seat {seat}  {cells[0]:<20}  {cells[1]}")

    # -- simplicity numbers (training/simplicity.py, the same ones the dashboard charts)
    print(f"\nOPENING SIMPLICITY  (light = 0-{LIGHT_HCP} HCP; spread = HCP 5%-95% range of a call,"
          " averaged over openings)")
    print("          light openings   unbalanced 1NT/2NT   HCP spread")
    rows = [(f"seat {seat}, {'vul' if vul else 'non-vul'}", [greedy[seat, vul]])
            for seat in range(1, 5) for vul in (False, True)]
    rows.append(("all seats", [greedy[k] for k in sorted(greedy)]))
    for label, parts in rows:
        k = len(parts)
        o = opening_numbers(np.concatenate(parts), np.tile(hcp, k), np.tile(f["bal"], k))
        print(f"  {label:<17} {pct(o['light_open_share'])}           {pct(o['unbalanced_nt_share'])}"
              f"         {o['opening_hcp_spread']:4.1f}")

    probs = opener(hands_t, args.seat, args.vul)
    a = probs.argmax(1)
    sure = probs.max(1)
    where = f"seat {args.seat}, {'vulnerable' if args.vul else 'not vulnerable'}"

    # -- every opening call
    print(f"\nEVERY OPENING CALL  ({where})")
    print("  call    share   HCP 5%/50%/95%   suit length (4+/5+/6+)   balanced  sure  "
          "common shapes")
    calls = [act for act, _ in Counter(a).most_common()]
    for act in sorted(calls, key=lambda c: (c == PASS, c)):
        m = a == act
        lo, mid, hi = np.percentile(hcp[m], [5, 50, 95]).astype(int)
        length = suit_length(f, act)
        if length is None:
            ltxt = "-"
        else:
            ln = length[m]
            ltxt = f"{ln.mean():.1f} ({pct((ln >= 4).mean()).strip()}/" \
                   f"{pct((ln >= 5).mean()).strip()}/{pct((ln >= 6).mean()).strip()})"
        shapes = " ".join(s for s, _ in Counter(f["shape"][m]).most_common(3))
        print(f"  {call_name(act):<6} {pct(m.mean())}   {lo:>3} {mid:>3} {hi:>3}      "
              f"{ltxt:<24} {pct(f['bal'][m].mean())} {sure[m].mean():.2f}  {shapes}"
              + ("   (few hands)" if m.sum() < 30 else ""))

    # -- call by HCP
    main_calls = [c for c in sorted(set(calls)) if (a == c).mean() >= 0.005]
    print(f"\nCALL BY HCP  ({where}; % of hands with that HCP; calls under 0.5% left out)")
    print("  HCP       n " + "".join(f"{call_name(c):>6}" for c in main_calls))
    for x in range(0, 31):
        m = hcp == x
        if m.sum() < 50:
            continue
        row = "".join(f"{100 * (a[m] == c).mean():6.0f}" if (a[m] == c).any() else "     ."
                      for c in main_calls)
        print(f"  {x:>3} {m.sum():>7} {row}")

    # -- example hands, most typical (highest probability) and a few random ones
    rng = np.random.default_rng(0)
    print(f"\nEXAMPLE HANDS  ({where})")
    for act in sorted(calls, key=lambda c: (c == PASS, c)):
        idx = np.nonzero(a == act)[0]
        if act == PASS:
            idx = idx[hcp[idx] >= 10]                 # the passes worth looking at
            if not len(idx):
                continue
        pick = rng.choice(idx, min(args.examples, len(idx)), replace=False)
        label = "Pass with 10+ HCP" if act == PASS else call_name(act)
        print(f"  {label}")
        for i in pick:
            print(f"    {hand_text(hands[i]):<32} {hcp[i]:>2} HCP  p={probs[i, act]:.2f}")

    # -- special hands: high-level openings and big hands
    print(f"\nSPECIAL HANDS  ({where})")
    high = [c for c in calls if c != PASS and c >= 5]
    for act in sorted(high):
        m = a == act
        print(f"  {call_name(act)} opening: {m.sum()} hand{'s' * (m.sum() != 1)}, "
              f"HCP {hcp[m].min()}-{hcp[m].max()}")
    for label, m in (("20-22 HCP", (hcp >= 20) & (hcp <= 22)), ("23+ HCP", hcp >= 23)):
        if m.sum():
            top = ", ".join(f"{call_name(c)} {100 * n / m.sum():.0f}%"
                            for c, n in Counter(a[m]).most_common(4))
            print(f"  {label} ({m.sum()} hands): {top}")
    for s, suit in enumerate(CARD_SUITS):
        m = (f["len"][:, s] >= 7) & (hcp <= 10)
        if m.sum() >= 20:
            top = ", ".join(f"{call_name(c)} {100 * n / m.sum():.0f}%"
                            for c, n in Counter(a[m]).most_common(3))
            print(f"  7+ {SYMBOL[suit]}, 10 HCP or less ({m.sum()} hands): {top}")


if __name__ == "__main__":
    main()
