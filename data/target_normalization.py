import pandas as pd


def compute_target_stats(records, min_std=1e-6):
    """Estimate per-condition target moments using training rows only."""
    stats = {}
    for condition, group in records.groupby("condition_id"):
        values = group["target"].astype(float)
        mean = float(values.mean())
        std = max(float(values.std(ddof=0)), min_std)
        stats[str(int(condition))] = {"mean": mean, "std": std, "count": int(values.size)}
    return stats


def standardize_records(records, stats):
    result = records.copy()
    result["target_raw"] = result["target"].astype(float)
    result["target"] = [
        (float(value) - stats[str(int(condition))]["mean"]) / stats[str(int(condition))]["std"]
        for value, condition in zip(result["target_raw"], result["condition_id"])
    ]
    return result


def normalized_thresholds(thresholds, stats):
    return {
        str(condition): (float(value) - stats[str(condition)]["mean"]) / stats[str(condition)]["std"]
        for condition, value in thresholds.items()
    }
