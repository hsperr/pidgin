# Simplicity

Pidgin bids what it holds. BRL is still stronger: it beats Pidgin V1 by about
0.4 IMPs per board ([results](results.md)). But Pidgin's bidding is much easier to read.
A partner can take almost every call at face value.

- A one-level suit opening shows four or more cards in that suit, 99.9% of the time.
- 1NT shows a strong hand, about 14–23 HCP.
- Weak hands pass. Pidgin opens from about 9 HCP and opens fewer 0–7 HCP hands than SAYC.
- No convention is built in: no Stayman, no transfers, no Blackwood, no strong
  artificial 2♣. The net learns only from the score. Pidgin V1 also pays a small
  training cost for each code word (see [README](../README.md#simplicity)).

A **code word** is a call partner cannot read at face value: a suit bid without four
cards (three when raising partner), a double of a contract at level 3 or below, a
redouble, or an artificial 2♣ opening.

## Code words in whole auctions

8,000 boards. Only the named side's calls count, per 100 auctions of that partnership.
Numbers from the paper ([results](results.md#simplicity)). SAYC is EPBot, a rule-based
bridge engine, playing its SAYC card.

```text
              code words   without X/XX   suit bids     low X   XX
              per 100      per 100        with length   per 100 per 100
Pidgin V1        5.3          2.6           98.7%        1.8    0.94
SAYC (EPBot)    37.1         21.1           86.7%       16.0    0.06
BRL             84.5         50.4           72.7%       33.7    0.33
```

SAYC uses about seven times as many code words as Pidgin V1. BRL uses about sixteen
times as many.

## Openings

The same 100,000 held-out hands for every bidder. Second seat (one pass before),
not vulnerable. Each net plays its most likely call. The SAYC calls come from EPBot.

- **opens:** share of all hands that open.
- **half open from:** the lowest HCP at which at least half the hands open.
- **0-7 HCP opened:** share of 0–7 HCP hands that open (preempts included).
- **calls used:** opening calls made on at least 0.1% of hands.
- **artificial openings:** share of openings that are code words.

```text
             opens  half open  0-7 HCP  calls  artificial   1NT HCP
                         from   opened   used    openings     5-95%
Pidgin V1    60.6%      9 HCP     5.0%      9        0.0%     14-22
Pidgin V2    60.0%      9 HCP     6.8%     12        0.0%     16-23
BRL          77.0%      0 HCP    91.0%     16       57.9%     13-18
SAYC         49.2%     11 HCP    11.2%     16        5.7%     15-17
```

- Pidgin V1 and V2 open about 60% of hands, from about 9 HCP. About 1 opening in
  6,000 is artificial.
- SAYC opens less, from 11 HCP. Its light openings are weak twos and preempts.
  Its artificial openings are short 1♣/1♦ and the strong 2♣.
- BRL opens 77% of hands, including every hand with 0–3 HCP. Its 1♣ is a catch-all
  for weak hands: 2–11 HCP, 2.7 clubs on average. More than half of its openings are
  code words.

### Share of hands opened, by HCP

```text
 HCP  Pidgin V1  Pidgin V2        BRL       SAYC
   0         0%         0%       100%         0%
   1         0%         0%       100%         1%
   2         0%         1%       100%         2%
   3         0%         2%       100%         4%
   4         1%         3%        98%         9%
   5         2%         4%        94%        12%
   6         5%         8%        88%        15%
   7        12%        13%        83%        14%
   8        31%        26%        81%        15%
   9        57%        54%        78%        14%
  10        83%        82%        75%        28%
  11        96%        94%        68%        67%
  12        99%        99%        60%       100%
  13       100%       100%        58%       100%
  14       100%       100%        62%       100%
  15       100%       100%        68%       100%
  16       100%       100%        71%       100%
  17       100%       100%        79%       100%
  18       100%       100%        87%       100%
  19       100%       100%        96%       100%
  20       100%       100%        99%       100%
  21       100%       100%       100%       100%
  22       100%       100%       100%       100%
```

## What each opening means

**Every opening call:** share of hands, HCP range (5% / 50% / 95% of the hands that make
the call), average length in the named suit, share with five or more cards, share with a
balanced shape (4333, 4432, 5332), and the three most common shapes.

**Call by HCP:** for each HCP count, the share of hands that make each call. Calls made
on less than 0.5% of hands are left out. A dot means never.

### Pidgin V1

```text
call    share   HCP 5% 50% 95%   suit length  5+ cards  balanced   common shapes
1♣      11.8%        9  12  17        5.0       69%       45%   5332 4432 5431
1♦      11.5%        8  11  16        5.1       75%       42%   5332 5431 4432
1♥      16.9%        8  12  16        4.9       61%       42%   4432 5431 5332
1♠      14.3%        8  11  16        5.0       69%       41%   4432 5332 5431
1NT      5.4%       14  18  22          -         -       37%   4432 5431 5332
3♦       0.2%       12  15  18        7.0      100%        0%   7321 7420 7411
3♥       0.2%       11  14  17        6.7      100%        0%   7321 6421 6331
3♠       0.1%       10  13  16        7.1      100%        0%   7321 7420 7411
4♥       0.1%       12  15  18        6.9      100%        0%   7321 7411 6421
Pass    39.4%        2   6  10          -         -       57%   4432 5332 4333
```

Four one-level suits with four or more cards, a wide strong 1NT, a few preempts at the
three level, and pass. That is the whole system.

```text
 HCP    1♣    1♦    1♥    1♠   1NT  Pass
   0     .     .     .     .     .   100
   1     .     .     .     .     .   100
   2     .     .     0     .     .   100
   3     .     .     0     0     .   100
   4     .     .     0     0     .    99
   5     .     0     1     1     .    98
   6     0     0     2     2     .    95
   7     0     2     5     4     .    88
   8     3     8    11     9     .    69
   9     9    15    18    16     .    43
  10    15    20    26    22     0    17
  11    19    21    29    26     0     4
  12    21    20    30    27     0     1
  13    21    20    30    27     1     0
  14    23    19    29    25     3     .
  15    25    16    27    21     7     .
  16    26    14    24    17    16     .
  17    28    10    16     8    35     .
  18    22     5     5     2    65     .
  19    10     1     1     0    87     .
  20     3     .     .     .    97     .
  21     1     .     .     .    99     .
  22     .     .     .     .   100     .
  23     .     .     .     .   100     .
  24     .     .     .     .   100     .
```

### Pidgin V2

```text
call    share   HCP 5% 50% 95%   suit length  5+ cards  balanced   common shapes
1♣      13.5%        9  12  17        4.7       54%       57%   4432 5332 5431
1♦      12.9%        9  12  17        4.8       59%       52%   4432 5332 5431
1♥      12.7%        9  12  17        4.8       63%       44%   4432 5431 5332
1♠       9.0%        8  11  16        5.2       91%       31%   5332 5431 5422
1NT      4.0%       16  19  23          -         -       35%   5431 4432 5332
2♣       2.3%        5   9  12        6.3      100%        0%   6322 6421 7321
2♦       2.2%        5   9  11        6.3      100%        0%   6322 6421 7321
2♥       1.7%        5   8  11        6.2      100%        0%   6421 6322 6331
2♠       0.7%        6   8  11        6.7      100%        0%   7321 6421 7222
3♥       0.5%        9  12  16        6.8      100%        0%   7321 6421 7411
3♠       0.4%        9  14  17        6.7      100%        0%   7321 6421 7420
4♥       0.2%       12  15  18        6.9      100%        0%   7321 6421 7411
Pass    40.0%        2   7  10          -         -       58%   4432 5332 4333
```

Like Pidgin V1, plus natural weak two-bids in all four suits. They always show five
or more cards in the suit, and six or more over 90% of the time. Its 1♠ promises five cards 91% of the time.

<details>
<summary>Pidgin V2: call by HCP</summary>

```text
 HCP    1♣    1♦    1♥    1♠   1NT    2♣    2♦    2♥    2♠    3♥  Pass
   0     .     .     .     .     .     .     .     .     .     .   100
   1     .     .     .     .     .     0     .     .     .     .   100
   2     .     .     .     .     .     0     0     0     .     .    99
   3     .     .     .     .     .     1     1     0     .     .    98
   4     .     .     .     .     .     1     1     1     0     .    97
   5     .     .     .     .     .     1     1     1     0     .    96
   6     .     .     .     .     .     2     2     2     1     0    92
   7     0     0     0     0     .     4     4     3     2     0    87
   8     2     1     2     5     .     5     5     4     2     0    74
   9    10     8     8    12     .     5     5     4     2     0    46
  10    20    17    17    16     .     4     4     3     1     1    18
  11    24    23    22    17     .     3     2     2     0     1     6
  12    25    26    26    17     0     1     1     1     0     1     1
  13    26    26    27    17     0     0     0     0     0     2     0
  14    26    27    27    15     1     0     .     .     0     1     0
  15    27    27    27    13     3     .     .     .     .     1     .
  16    27    26    25     9     9     .     .     .     .     1     .
  17    27    24    20     6    21     .     .     .     .     0     .
  18    22    19    11     2    45     .     .     .     .     0     .
  19    13     5     4     0    77     .     .     .     .     .     .
  20     3     1     0     .    94     .     .     .     .     .     .
  21     1     .     .     .    99     .     .     .     .     .     .
  22     .     .     .     .   100     .     .     .     .     .     .
  23     .     .     .     .   100     .     .     .     .     .     .
  24     .     .     .     .   100     .     .     .     .     .     .
```

</details>

### SAYC (EPBot)

```text
call    share   HCP 5% 50% 95%   suit length  5+ cards  balanced   common shapes
1♣      10.2%       11  13  18        4.6       56%       51%   4432 4333 5431
1♦      11.0%       11  13  18        4.8       57%       44%   4432 5431 5422
1♥       7.3%       10  13  18        5.4      100%       17%   5431 5422 5332
1♠       7.6%       10  13  18        5.4      100%       15%   5431 5422 5332
1NT      5.1%       15  16  17          -         -       94%   4432 5332 4333
2♣       0.6%       19  22  25        3.1       13%       42%   4432 5332 5431
2♦       1.4%        4   8  10        6.0      100%        0%   6322 6331 6421
2♥       1.6%        4   8  10        6.0      100%        0%   6322 6421 6331
2♠       1.7%        4   8  10        6.0      100%        0%   6322 6421 6331
2NT      0.6%       20  20  21          -         -       91%   4432 4333 5332
3♣       0.5%        4   8  10        6.8      100%        0%   7321 6421 7222
3♦       0.4%        4   7   9        6.8      100%        0%   7321 6421 7222
3♥       0.3%        4   7   9        7.0      100%        0%   7321 7222 7330
3♠       0.3%        4   7   9        7.0      100%        0%   7321 7222 7330
4♥       0.1%        4   8  10        7.5      100%        0%   7411 7420 8221
4♠       0.1%        5   8  10        7.6      100%        0%   7411 8221 7420
Pass    50.8%        3   7  11          -         -       58%   4432 5332 5431
```

<details>
<summary>SAYC: call by HCP</summary>

```text
 HCP    1♣    1♦    1♥    1♠   1NT    2♣    2♦    2♥    2♠   2NT    3♣  Pass
   0     .     .     .     .     .     .     .     .     .     .     .   100
   1     .     .     .     .     .     .     .     .     0     .     0    99
   2     .     .     .     .     .     .     .     1     1     .     0    98
   3     .     .     .     .     .     .     .     1     1     .     0    96
   4     .     .     .     .     .     .     2     2     2     .     0    91
   5     .     .     .     .     .     .     3     3     3     .     1    88
   6     .     .     .     .     .     .     3     4     4     .     1    85
   7     .     .     .     .     .     .     3     3     4     .     1    86
   8     .     .     .     .     .     .     3     4     3     .     1    85
   9     .     .     .     .     .     .     3     3     4     .     1    86
  10     0     3     7     8     .     .     3     2     3     .     1    72
  11    16    18    16    17     .     .     .     .     .     .     .    33
  12    33    33    17    17     .     .     .     .     .     .     .     .
  13    32    32    18    18     .     .     .     .     .     .     .     .
  14    31    35    17    17     .     .     .     .     .     .     .     .
  15    10    13    13    13    52     .     .     .     .     .     .     .
  16    11    13    12    12    52     .     .     .     .     .     .     .
  17    12    13    12    12    50     0     .     .     .     .     .     .
  18    33    34    15    17     .     1     .     .     .     .     .     .
  19    34    33    14    17     .     2     .     .     .     .     .     .
  20     8    10     7     8     .     9     .     .     .    57     .     .
  21     8     7     7     6     .    15     .     .     .    58     .     .
  22     .     .     .     .     .   100     .     .     .     .     .     .
  23     .     .     .     .     .   100     .     .     .     .     .     .
  24     .     .     .     .     .   100     .     .     .     .     .     .
```

</details>

### BRL

```text
call    share   HCP 5% 50% 95%   suit length  5+ cards  balanced   common shapes
1♣      37.2%        2   6  11        2.7        9%       46%   4432 5332 5431
1♦      13.8%        7  10  13        2.3        2%       47%   5332 4432 5431
1♥       4.6%       10  14  17        4.9       80%       24%   5422 5431 5332
1♠       4.0%       12  14  17        4.8       71%       23%   5431 5422 5332
1NT      3.1%       13  15  18          -         -       67%   5332 4333 4432
2♣       4.5%       15  19  23        3.3       21%       33%   5431 4432 5332
2♦       2.6%        7  10  13        5.7      100%        0%   6322 5431 6421
2♥       1.9%        7  10  13        5.7      100%        0%   6322 6421 6331
2♠       1.4%        7  11  13        5.8      100%        0%   6322 6421 6331
2NT      0.2%       17  19  20          -         -      100%   4333 4432 5332
3♣       0.8%        7  11  14        6.7      100%        0%   7321 6421 7222
3♦       0.9%        9  12  15        6.4      100%        0%   7321 6421 6331
3♥       0.9%        8  12  15        6.5      100%        0%   7321 6421 6331
3♠       0.5%        9  12  15        6.7      100%        0%   7321 6421 7222
4♥       0.3%       11  14  17        6.8      100%        0%   7321 6421 7411
4♠       0.2%       10  14  16        7.1      100%        0%   7321 7420 6421
Pass    23.0%        6  11  16          -         -       78%   4432 4333 5332
```

<details>
<summary>BRL: call by HCP</summary>

```text
 HCP    1♣    1♦    1♥    1♠   1NT    2♣    2♦    2♥    2♠    3♣    3♦    3♥    3♠  Pass
   0   100     .     .     .     .     .     .     .     .     .     .     .     .     .
   1   100     .     .     .     .     .     .     .     .     .     .     .     .     0
   2   100     .     .     .     .     .     .     .     .     .     .     .     .     0
   3    99     .     .     .     .     .     0     0     .     0     .     .     .     0
   4    98     .     .     .     .     .     0     0     .     0     .     .     .     2
   5    93     0     .     .     .     .     1     0     0     0     0     0     0     6
   6    83     2     .     .     .     .     1     1     0     0     0     0     .    12
   7    68    10     .     .     .     .     2     2     1     0     .     0     0    17
   8    49    22     0     .     .     .     4     3     1     1     0     1     0    19
   9    34    29     1     0     .     .     5     3     2     1     0     1     0    22
  10    24    31     3     0     .     .     6     4     3     2     1     1     1    25
  11    16    27     5     1     0     .     5     4     4     2     2     1     1    32
  12    10    19     8     4     1     0     3     3     3     2     2     2     2    40
  13     6    12    10    10     4     0     2     2     2     1     3     2     1    42
  14     3     6    13    16    12     2     1     1     1     1     2     2     1    38
  15     1     4    15    19    19     4     0     0     0     0     2     1     1    32
  16     0     1    16    18    20    11     0     0     0     0     1     0     0    29
  17     .     1    15    15    19    26     .     .     .     .     0     0     0    21
  18     .     0     8     9    11    54     .     .     .     .     .     .     .    13
  19     .     .     2     3     1    81     .     .     .     .     .     .     .     4
  20     .     .     1     0     .    96     .     .     .     .     .     .     .     1
  21     .     .     .     .     .    98     .     .     .     .     .     .     .     .
  22     .     .     .     .     .   100     .     .     .     .     .     .     .     .
  23     .     .     .     .     .    99     .     .     .     .     .     .     .     .
  24     .     .     .     .     .   100     .     .     .     .     .     .     .     .
```

</details>

## Redo the tables

From the repository root, with the released weights in `server/models/`
([scripts/get_models.sh](../scripts/get_models.sh)) and the dataset in `data/`
([README](../README.md#training-data)):

```bash
python scripts/opening_tables.py \
  "Pidgin V1=server/models/D_cw_s75k.pt" \
  "Pidgin V2=server/models/pidginv2_bid_s40000.pt" \
  "BRL=brl:server/models/brl_fsp_weights.npz" \
  "SAYC=calls:sayc_seat2_nonvul.npy"
```

EPBot is not part of this repository. `calls:FILE.npy` takes one call id per hand
(0 = 1♣ … 34 = 7NT, 35 = Pass), in the order the script reads the hands. Leave the
SAYC entry out to compare only the nets. `--seat` and `--vul` pick another seat.
