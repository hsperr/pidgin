# Reading the training curves

The model guides show measurements from the runs that produced the released
models. Each plot stops at the released checkpoint. Lines connect the recorded
points, with no added smoothing. All four plots cover a fine-tuning stage that
starts with pretrained weights.

| Guide | Stage shown | Steps | What the plots measure |
|---|---|---|---|
| [Pidgin V1 bidding](pidgin_v1.md#training-history) | Table-score training | 0–75,000 | Training call-value error; calls flagged by the simplicity rule per 100 validation auctions |
| [Pidgin V2 bidding](pidgin_v2.md#training-history) | Weak-opening penalty | 0–40,000 | Validation IMPs against Pidgin V1; percentage of 0–7 high-card-point hands that open |
| [Bidding belief model](belief.md#training-history) | Learning hidden-hand summaries | 200–30,000 | Card-owner cross-entropy and summary loss |
| [Pidgin V1 card play](card_play.md#training-history) | Self-play against earlier checkpoints | 1–20,000 | Critic squared error and hidden-card cross-entropy |

## What to look for

- **Prediction losses:** lower means predictions are closer to their training
  targets. Bidding values use units of 100 bridge points; the card-play critic
  predicts tricks. Their squared errors are on different scales.
- **Cross-entropy:** lower means the model assigns more probability to the correct
  hidden-card holder or hand summary. Values use natural logarithms (nats).
- **IMPs per board:** positive means the bidder scores better than its named
  opponent. These are validation matches used during training. See
  [results](results.md) for separate comparisons of the released models.
- **Style measures:** flagged calls (code words) and weak openings per 100 auctions.

The belief losses average batches within each logging window. The other training
losses show the batch recorded at each point. The V1 call-value loss starts at
step 3,000 because the step-zero record contains validation measurements only.
The released Q-net has no saved sequence of training-loss measurements available,
so no training curve is shown for it.

## Redraw the figures

[The numeric snapshot](data/training_curves.json) contains every plotted value,
its metric name, and file hashes for checking its source. Model weights were
checked against the corresponding run before the measurements were extracted.
The snapshot contains no machine-specific paths or internal run names.

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install matplotlib
python scripts/plot_training_curves.py
```

This reads the committed snapshot and writes four SVGs under `docs/figures/`.
For local PNG previews, add `--preview-dir /tmp/pidgin-curves`. No weights,
training runs, or dashboard server are needed to redraw the plots.
