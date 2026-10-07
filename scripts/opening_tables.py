"""Side-by-side opening tables for several bidders, as text for docs/simplicity.md.

Every bidder is asked for its opening call on the same held-out hands, in one seat and
vulnerability (default: second seat, after one pass, not vulnerable). A bidder is
NAME=SPEC, where SPEC is a four-seat checkpoint, ``brl:WEIGHTS.npz`` (see
tools/openings.py), or ``calls:FILE.npy``: one precomputed call id per hand, in the
order the hands are read, for a bidder that cannot be loaded here.

    python scripts/opening_tables.py \\
        "Pidgin V1=server/models/D_cw_s75k.pt" \\
        "Pidgin V2=server/models/pidginv2_bid_s40000.pt" \\
        "BRL=brl:server/models/brl_fsp_weights.npz" \\
        "SAYC=calls:sayc_seat2_nonvul.npy"
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

from tools.openings import call_name, hand_features, load_opener, suit_length  # noqa: E402
from training.bridge.calls import PASS  # noqa: E402
from training.contract.data import load_range  # noqa: E402
from training.simplicity import LIGHT_HCP, opening_numbers  # noqa: E402


def load_calls(spec: str, hands_t: torch.Tensor, seat: int, vul: bool) -> np.ndarray:
    if spec.startswith("calls:"):
        calls = np.load(spec[6:]).astype(int)
        if calls.shape != (len(hands_t),):
            raise SystemExit(f"{spec}: expected {len(hands_t)} calls, got {calls.shape}")
        return calls
    opener, _ = load_opener(spec)
    return opener(hands_t, seat, vul).argmax(1)


def artificial(calls: np.ndarray, f: dict) -> np.ndarray:
    """Openings flagged by the code-word rule: a suit bid with fewer than 4 cards, or a
    2♣ opening with fewer than 5 clubs or 20+ HCP."""
    out = np.zeros(len(calls), bool)
    for act in np.unique(calls[calls != PASS]):
        m = calls == act
        length = suit_length(f, act)
        if length is not None:
            out[m] |= length[m] < 4
        if act == 5:                                              # 2♣
            out[m] |= (f["len"][m, 3] < 5) | (f["hcp"][m] >= 20)
    return out


def summary(bidders: dict, f: dict) -> list[str]:
    hcp = f["hcp"]
    head = (f"{'':<11}{'opens':>7}{'half open':>11}{'0-' + str(LIGHT_HCP) + ' HCP':>9}"
            f"{'calls':>7}{'artificial':>12}{'1NT HCP':>10}")
    lines = [head, f"{'':<11}{'':>7}{'from':>11}{'opened':>9}{'used':>7}{'openings':>12}"
             f"{'5-95%':>10}"]
    for name, a in bidders.items():
        opens = a != PASS
        half = next(x for x in range(38) if (hcp == x).any() and opens[hcp == x].mean() >= .5)
        o = opening_numbers(a, hcp, f["bal"])
        used = sum(1 for c, n in Counter(a[opens]).items() if n >= 0.001 * len(a))
        nt = a == 4
        lo, hi = np.percentile(hcp[nt], [5, 95]).astype(int) if nt.any() else ("-", "-")
        art = artificial(a, f)[opens].mean()
        lines.append(f"{name:<11}{100 * opens.mean():6.1f}%{half:>7} HCP"
                     f"{100 * o['light_open_share']:8.1f}%{used:>7}{100 * art:11.1f}%"
                     f"{f'{lo}-{hi}':>10}")
    return lines


def opened_by_hcp(bidders: dict, hcp: np.ndarray) -> list[str]:
    lines = [f"{'HCP':>4}" + "".join(f"{n:>11}" for n in bidders)]
    for x in range(0, 23):
        m = hcp == x
        if not m.any():
            continue
        row = "".join(f"{100 * (a[m] != PASS).mean():10.0f}%" for a in bidders.values())
        lines.append(f"{x:>4}{row}")
    return lines


def every_call(a: np.ndarray, f: dict) -> list[str]:
    hcp = f["hcp"]
    lines = ["call    share   HCP 5% 50% 95%   suit length  5+ cards  balanced   common shapes"]
    for act in sorted(set(a.tolist()), key=lambda c: (c == PASS, c)):
        m = a == act
        if m.mean() < 0.001:
            continue
        lo, mid, hi = np.percentile(hcp[m], [5, 50, 95]).astype(int)
        length = suit_length(f, act)
        if length is None or act == PASS:
            ltxt = f"{'-':>11}{'-':>10}"
        else:
            ltxt = f"{length[m].mean():11.1f}{100 * (length[m] >= 5).mean():9.0f}%"
        shapes = " ".join(s for s, _ in Counter(f["shape"][m]).most_common(3))
        lines.append(f"{call_name(act):<6}{100 * m.mean():6.1f}%      {lo:>3} {mid:>3} {hi:>3}"
                     f"{ltxt}{100 * f['bal'][m].mean():9.0f}%   {shapes}")
    return lines


def call_by_hcp(a: np.ndarray, hcp: np.ndarray) -> list[str]:
    calls = [c for c in sorted(set(a.tolist())) if (a == c).mean() >= 0.005]
    calls = [c for c in calls if c != PASS] + [PASS]
    lines = [f"{'HCP':>4}" + "".join(f"{call_name(c):>6}" for c in calls)]
    for x in range(0, 25):
        m = hcp == x
        if m.sum() < 50:
            continue
        lines.append(f"{x:>4}" + "".join(f"{100 * (a[m] == c).mean():6.0f}"
                                         if (a[m] == c).any() else "     ."
                                         for c in calls))
    return lines


def block(title: str, lines: list[str]) -> str:
    return f"### {title}\n\n```text\n" + "\n".join(lines) + "\n```\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("bidders", nargs="+", help="NAME=SPEC (checkpoint, brl:W.npz, calls:F.npy)")
    p.add_argument("--data", default="data/dds_results_100M.npy")
    p.add_argument("--deals", type=int, default=25000, help="last N (held-out) deals, 4 hands each")
    p.add_argument("--seat", type=int, default=2, choices=(1, 2, 3, 4))
    p.add_argument("--vul", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    hands_t = load_range(args.data, -args.deals, args.deals).hands.reshape(-1, 52)
    f = hand_features(hands_t.numpy().astype(np.uint8))
    bidders = {}
    for item in args.bidders:
        name, spec = item.split("=", 1)
        bidders[name] = load_calls(spec, hands_t, args.seat, args.vul)

    where = f"seat {args.seat}, {'vulnerable' if args.vul else 'not vulnerable'}"
    print(f"<!-- {len(hands_t):,} held-out hands, {where} -->\n")
    print(block("Summary", summary(bidders, f)))
    print(block("Share of hands opened, by HCP", opened_by_hcp(bidders, f["hcp"])))
    for name, a in bidders.items():
        print(block(f"{name}: every opening call", every_call(a, f)))
        print(block(f"{name}: call by HCP (% of hands with that HCP)", call_by_hcp(a, f["hcp"])))


if __name__ == "__main__":
    main()
