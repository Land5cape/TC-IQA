"""Wall-clock measurements in image pairs, not condition-label counts."""

import math


def phase_timing(pairs, elapsed_seconds):
    elapsed = max(float(elapsed_seconds), 0.0)
    return {"pairs": int(pairs), "elapsed_seconds": elapsed,
            "pairs_per_second": float(pairs) / elapsed if pairs > 0 and elapsed > 0 else None}


def format_duration(seconds):
    if seconds is None or not math.isfinite(float(seconds)):
        return "n/a"
    total = max(int(round(seconds)), 0)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def estimate_full_epoch(train, train_eval, validation, full_train_pairs, full_validation_pairs):
    train_rate = train.get("pairs_per_second")
    validation_rate = validation.get("pairs_per_second")
    if not train_rate or not validation_rate:
        return None
    train_seconds = int(full_train_pairs) / train_rate
    validation_seconds = int(full_validation_pairs) / validation_rate
    train_eval_seconds = float(train_eval["elapsed_seconds"])
    return {
        "full_train_pairs": int(full_train_pairs),
        "full_validation_pairs": int(full_validation_pairs),
        "measured_train_pairs_per_second": train_rate,
        "measured_validation_pairs_per_second": validation_rate,
        "full_train_seconds": train_seconds,
        "full_validation_seconds": validation_seconds,
        "train_eval_seconds": train_eval_seconds,
        "full_epoch_seconds": train_seconds + validation_seconds + train_eval_seconds,
        "basis": "measured wall-clock rates; full train and validation; train-eval unchanged",
        "excludes": ["initial metadata scan", "checkpoint I/O", "task refinement", "final diagnostics"],
    }
