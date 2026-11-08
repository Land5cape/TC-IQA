import json
import re
from datetime import datetime
from pathlib import Path


_DATE_PREFIX = re.compile(r"^\d{8}_")
_TIMESTAMP_PREFIX = re.compile(r"^\d{8}_\d{6}_")


def dated_output_dir(path, now=None):
    """Return an output directory whose final component is date-prefixed.

    Existing date-prefixed directories are preserved so resume/evaluation
    commands do not create a second prefix.
    """
    path = Path(path)
    if _DATE_PREFIX.match(path.name):
        return path
    date = (now or datetime.now()).strftime("%Y%m%d")
    return path.with_name(f"{date}_{path.name}")


def unique_timestamped_output_dir(path, now=None):
    """Atomically reserve a non-overwriting run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base_name = path.name if _TIMESTAMP_PREFIX.match(path.name) else f"{timestamp}_{path.name}"
    attempt = 1
    while True:
        suffix = "" if attempt == 1 else f"_{attempt:02d}"
        candidate = path.with_name(f"{base_name}{suffix}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            attempt += 1


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def default_metrics_path(checkpoint, output=None):
    if output is not None:
        return Path(output)
    return Path(checkpoint).resolve().parent / "test_metrics.json"


def should_stop_early(completed_epochs, requested_epochs, bad_epochs, patience):
    return completed_epochs < requested_epochs and bad_epochs >= patience


def save_run_summary(
    path,
    *,
    requested_epochs,
    completed_epochs,
    stopped_early,
    best_epoch,
    best_macro_srcc,
):
    return save_json(
        path,
        {
            "requested_epochs": requested_epochs,
            "completed_epochs": completed_epochs,
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_macro_srcc": best_macro_srcc,
        },
    )
