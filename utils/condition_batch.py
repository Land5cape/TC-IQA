"""Batch-shape helpers shared by MPD and MIQD training."""

from __future__ import annotations


CONDITION_FIELDS = ("condition_id", "target", "target_raw", "robust_group_id")


def flatten_condition_batch(output, batch):
    """Flatten scalar or vector conditions into the loss/evaluation axis."""

    flat_output = dict(output)
    flat_output["mean"] = output["mean"].reshape(-1)
    flat_output["sigma"] = output["sigma"].reshape(-1)
    flat_batch = {
        name: batch[name].reshape(-1)
        for name in CONDITION_FIELDS
        if name in batch
    }
    if "group_id" in batch:
        pairs = len(batch["group_id"])
        conditions_per_pair = flat_output["mean"].numel() // max(pairs, 1)
        groups = batch["group_id"]
        flat_batch["rank_group_id"] = [
            str(group) for group in groups for _ in range(conditions_per_pair)
        ]
    return flat_output, flat_batch
