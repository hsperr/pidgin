"""Cards, deals, and hand encoding. No bridge knowledge beyond dealing."""
import numpy as np

RANKS = "AKQJT98765432"   # index 0 = ace ... 12 = two
SUIT_CHARS = "SHDC"       # index 0 = spades ... 3 = clubs
NCARDS = 52

def card_index(suit, rank):
    return suit * 13 + rank

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

# ---- diagnostic-only features (never fed to the agents) ----
HCP_BY_RANK = np.array([4, 3, 2, 1] + [0] * 9)

def hcp(hand52):
    return int((hand52.reshape(4, 13) * HCP_BY_RANK).sum())

def suit_lengths(hand52):
    return hand52.reshape(4, 13).sum(axis=1)
