"""
STEP 4B: RETRAIN WITH LABEL SMOOTHING (IF TEMPERATURE SCALING ISN'T ENOUGH)
=============================================================================
What this does:
    Retrains your model so it's never allowed to be trained toward an
    exact 0 or exact 1 target. Instead of teaching the model "this is
    DEFINITELY sepsis" (target = 1.0), it teaches "this is very
    LIKELY sepsis" (target = 0.95). This prevents the saturation
    problem at the source, rather than patching it after the fact
    like temperature scaling does.

    Use this if:
      - Step 3 (SHAP check) didn't find an obvious leaking feature, AND
      - Step 4A (temperature scaling) didn't fix it well enough

How to run:
    python 5_retrain_with_label_smoothing.py

Before running, fill in the "LOAD YOUR OWN" section near the bottom
with your actual model, optimizer setup, and training loop details.
This file gives you the two key pieces (the loss function and the
epoch-selection logic) to drop into your existing training script —
it is a template, not a full standalone trainer, since your existing
training loop already knows how to load your specific data.

What you need:
    - Your existing training script/loop
    - Validation predictions with both AUROC and Brier score computed
      per epoch (for the smarter checkpoint selection at the bottom)

Output:
    This file defines two reusable pieces:
      1. label_smoothed_bce()   -> use this as your loss function
      2. select_best_epoch()    -> use this instead of picking the
                                    checkpoint with the single best AUROC
"""

import numpy as np
import torch


def label_smoothed_bce(logits, targets, smoothing=0.05):
    """
    Drop-in replacement for BCEWithLogitsLoss.
    Instead of pushing predictions all the way to 0 or 1, this caps
    the target at (1 - smoothing) and (0 + smoothing), so the model
    is never rewarded for being 100% confident.

    Usage in your training loop:
        loss = label_smoothed_bce(model_output_logits, batch_labels)
        loss.backward()
    """
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_smoothed)


def select_best_epoch(history):
    """
    Use this AFTER training finishes, instead of just picking the
    epoch with the highest val_auroc.

    'history' should be a dict you build during training, like:
        history = {
            "val_auroc": [0.71, 0.75, 0.77, 0.74, ...],   # one number per epoch
            "val_brier": [0.22, 0.19, 0.15, 0.24, ...],   # one number per epoch
        }

    This picks the epoch that balances good ranking (AUROC) against
    honest confidence (low Brier score), instead of rewarding an
    epoch that got a high AUROC by being falsely overconfident.
    """
    scores = [
        auroc - 0.5 * brier
        for auroc, brier in zip(history["val_auroc"], history["val_brier"])
    ]
    best_epoch = int(np.argmax(scores))
    print(f"Best epoch by AUROC-minus-calibration-penalty: {best_epoch}")
    print(f"  (val_auroc={history['val_auroc'][best_epoch]:.3f}, "
          f"val_brier={history['val_brier'][best_epoch]:.3f})")
    return best_epoch


if __name__ == "__main__":
    # ============================================================
    # LOAD YOUR OWN: this is a template, not a runnable script.
    # ============================================================
    #
    # In your existing training script, make these two changes:
    #
    # CHANGE 1 — replace your loss function:
    #     BEFORE:
    #         criterion = torch.nn.BCEWithLogitsLoss()
    #         loss = criterion(logits, targets)
    #     AFTER:
    #         loss = label_smoothed_bce(logits, targets, smoothing=0.05)
    #
    # CHANGE 2 — also add weight decay to your optimizer for extra
    # regularization:
    #     BEFORE:
    #         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    #     AFTER:
    #         optimizer = torch.optim.AdamW(
    #             model.parameters(), lr=1e-3, weight_decay=1e-4
    #         )
    #
    # CHANGE 3 — after training all epochs, instead of:
    #         best_epoch = np.argmax(history["val_auroc"])
    #     use:
    #         best_epoch = select_best_epoch(history)
    #
    # Then retrain, and re-run script 1 (diagnose_saturation.py) on
    # the new model's predictions to confirm the fraction saturated
    # and Brier score both improved.
    print("This file is a template — see the comments above for the 3 "
          "changes to make in your existing training script, then rerun "
          "training. This file itself does not need to be run directly.")
