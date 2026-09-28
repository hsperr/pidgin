"""Cards, calls, deals, their names and link codes: what every page and API shares."""
import base64

import numpy as np

from bridgezero.bridge.calls import CONTRACTS, DOUBLE, PASS, REDOUBLE

RANKS = "AKQJT98765432"   # index 0 = ace ... 12 = two
SUITS = "SHDC"            # index 0 = spades ... 3 = clubs; card = suit * 13 + rank
NCARDS = 52
SEAT_NAMES = ["North", "East", "South", "West"]
STRAINS = ["S", "H", "D", "C", "NT"]      # trump order: the card suits, then notrump
TRUMP_TO_BID_STRAIN = (3, 2, 1, 0, 4)     # S H D C NT -> bids' C D H S NT (its own inverse)
NAMES = [c[0] for c in CONTRACTS]         # bid index -> "1C" ... "7NT"
N_CALLS = REDOUBLE + 1                    # 35 contracts + Pass + X + XX
HCP_W = np.array([4, 3, 2, 1] + [0] * 9)


def card_name(card):
    return SUITS[card // 13] + RANKS[card % 13]


def call_name(c):
    return "Pass" if c == PASS else "X" if c == DOUBLE else "XX" if c == REDOUBLE else NAMES[c]


def call_token(c):
    """A call as the self-play corpus keys spell it: P, X, XX, 1C ..."""
    return "P" if c == PASS else "X" if c == DOUBLE else "XX" if c == REDOUBLE else NAMES[c]


def trick_best(cards, trump):
    """Index into `cards` (play order) of the card winning so far; `trump` 4 = notrump."""
    if not cards:
        return None
    best = 0
    for i, c in enumerate(cards):
        b = cards[best]
        if c // 13 == b // 13:
            if c % 13 < b % 13:         # rank 0 is the ace, so a lower index wins
                best = i
        elif trump < 4 and c // 13 == trump and b // 13 != trump:
            best = i
    return best


def beats(card, best_card, trump):
    """Would `card` be winning the trick if it were played now?"""
    if best_card is None:
        return True
    if card // 13 == best_card // 13:
        return card % 13 < best_card % 13
    if trump < 4 and card // 13 == trump and best_card // 13 != trump:
        return True
    return False


def deal_owners(rng):
    """Return array of length 52, value = seat (0=N,1=E,2=S,3=W) holding that card."""
    owners = np.repeat(np.arange(4), 13)
    rng.shuffle(owners)
    return owners


def owners_to_pbn(owners):
    """PBN string, north first."""
    hands = []
    for seat in range(4):
        idx = np.flatnonzero(owners == seat)
        suits = []
        for s in range(4):
            ranks = [RANKS[i % 13] for i in idx if i // 13 == s]
            suits.append("".join(ranks))
        hands.append(".".join(suits))
    return "N:" + " ".join(hands)


def owners_to_bitmaps(owners):
    """(4, 52) uint8 one-hot-per-seat card ownership."""
    out = np.zeros((4, NCARDS), dtype=np.uint8)
    out[owners, np.arange(NCARDS)] = 1
    return out


# ---- link codes, shared by /, /play and /table

CALL_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
CARD_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"   # 52, one per card


def encode_deal(owners):
    """52 owners x 2 bits = 13 bytes = 18 url-safe chars."""
    n = 0
    for o in owners:
        n = (n << 2) | int(o)
    return base64.urlsafe_b64encode(n.to_bytes(13, "big")).decode().rstrip("=")


def decode_deal(code):
    """Inverse of encode_deal; None unless it is a real deal (13 cards each)."""
    try:
        raw = base64.urlsafe_b64decode(code + "=" * (-len(code) % 4))
    except Exception:
        return None
    if len(raw) != 13:
        return None
    n = int.from_bytes(raw, "big")
    owners = np.array([(n >> (2 * (51 - i))) & 3 for i in range(52)], dtype=np.int64)
    return owners if all((owners == s).sum() == 13 for s in range(4)) else None


def encode_cards(cards):
    return "".join(CARD_CHARS[c] for c in cards)


def decode_cards(code):
    out = []
    for ch in str(code)[:52]:
        i = CARD_CHARS.find(ch)
        if i < 0:
            break
        out.append(i)
    return out
