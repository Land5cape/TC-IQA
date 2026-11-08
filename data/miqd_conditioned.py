"""MIQD-2.5M adapter for the unified conditioned IQA model.

The adapter deliberately keeps paths relative to the dataset root so that a
configuration can be moved from a local Windows machine to a server by changing
only ``dataset_root``.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms


MIQD_METRICS: Tuple[str, ...] = ("accuracy", "consistency", "composite")


class MIQDIncompleteError(RuntimeError):
    """Raised when a dataset looks like a full download but files are missing."""


@dataclass(frozen=True)
class MIQDSource:
    """A source image and the label tables for one downstream task."""

    task: str
    source_id: str
    group_id: str
    ref_path: Path
    image_dir: Path
    label_paths: Tuple[Path, ...]
    domain: str

    @classmethod
    def classification(cls, source_id: str) -> "MIQDSource":
        return cls(
            task="cls",
            source_id=source_id,
            group_id=f"imagenet:{source_id}",
            ref_path=Path("classification_src/src_images") / f"{source_id}.JPEG",
            image_dir=Path("classification_images/images") / source_id,
            label_paths=(
                Path("classification_labels/labels") / f"{source_id}_mmos.csv",
            ),
            domain="imagenet",
        )

    @classmethod
    def coco(cls, task: str, source_id: str) -> "MIQDSource":
        if task not in {"det", "ins"}:
            raise ValueError(f"Unsupported COCO task: {task}")
        label_dir = "detection_labels" if task == "det" else "segmentation_labels"
        return cls(
            task=task,
            source_id=source_id,
            group_id=f"coco:{source_id}",
            ref_path=Path("detection_src/src_images") / f"{source_id}.jpg",
            image_dir=Path("detection_images/images") / source_id,
            label_paths=(
                Path(label_dir) / "labels" / f"{source_id}_accuracy.csv",
                Path(label_dir) / "labels" / f"{source_id}_consistency.csv",
            ),
            domain="coco",
        )


@dataclass(frozen=True)
class MIQDCompleteness:
    mode: str
    expected_sources: int
    available_sources: int
    expected_images: int | None = None
    available_images: int | None = None
    missing_images: int | None = None
    image_check_performed: bool = False


def _source_ids(label_dir: Path, suffix: str) -> List[str]:
    if not label_dir.is_dir():
        return []
    return sorted(path.name[: -len(suffix)] for path in label_dir.glob(f"*{suffix}"))


def discover_miqd_sources(root: str | Path) -> List[MIQDSource]:
    """Discover the three official task label collections."""

    root = Path(root)
    cls_ids = _source_ids(root / "classification_labels/labels", "_mmos.csv")
    sources: List[MIQDSource] = [MIQDSource.classification(value) for value in cls_ids]

    for task, dirname in (("det", "detection_labels"), ("ins", "segmentation_labels")):
        label_dir = root / dirname / "labels"
        accuracy = set(_source_ids(label_dir, "_accuracy.csv"))
        consistency = set(_source_ids(label_dir, "_consistency.csv"))
        unpaired = accuracy.symmetric_difference(consistency)
        if unpaired:
            preview = ", ".join(sorted(unpaired)[:5])
            raise MIQDIncompleteError(
                f"Unpaired {task} label files for {len(unpaired)} sources: {preview}"
            )
        sources.extend(MIQDSource.coco(task, value) for value in sorted(accuracy))
    return sources


def _target_triplet(accuracy: float, consistency: float) -> List[float]:
    composite = 0.5 * (accuracy + consistency)
    return [1.0 - accuracy, 1.0 - consistency, 1.0 - composite]


def _robust_group(task: str, filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.rsplit("_", 2)
    distortion = parts[-2] if len(parts) >= 3 else "unknown"
    return f"{task}:{distortion}"


def read_miqd_source_rows(
    root: str | Path,
    source: MIQDSource,
    *,
    existing_only: bool = False,
) -> Iterator[Dict[str, object]]:
    """Yield one image-pair record containing all three MIQD targets."""

    root = Path(root)
    if source.task == "cls":
        frame = pd.read_csv(root / source.label_paths[0])
        required = {"dist_images", "weighted_accuracy", "weighted_consistency"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns in {source.label_paths[0]}: {sorted(missing)}")
        values = frame[["dist_images", "weighted_accuracy", "weighted_consistency"]].itertuples(
            index=False, name=None
        )
    else:
        accuracy = pd.read_csv(root / source.label_paths[0])
        consistency = pd.read_csv(root / source.label_paths[1])
        required = {"filename", "AP_weighted_avg"}
        for label_path, frame in zip(source.label_paths, (accuracy, consistency)):
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(f"Missing columns in {label_path}: {sorted(missing)}")
        merged = accuracy[["filename", "AP_weighted_avg"]].merge(
            consistency[["filename", "AP_weighted_avg"]],
            on="filename",
            how="outer",
            suffixes=("_accuracy", "_consistency"),
            validate="one_to_one",
            indicator=True,
        )
        unmatched = merged[merged["_merge"] != "both"]
        if not unmatched.empty:
            raise MIQDIncompleteError(
                f"Unmatched accuracy/consistency rows in source {source.source_id}"
            )
        values = merged[
            ["filename", "AP_weighted_avg_accuracy", "AP_weighted_avg_consistency"]
        ].itertuples(index=False, name=None)

    for filename, accuracy_value, consistency_value in values:
        filename = Path(str(filename)).name
        dist_path = source.image_dir / filename
        if existing_only and not (root / dist_path).is_file():
            continue
        accuracy_float = float(accuracy_value)
        consistency_float = float(consistency_value)
        yield {
            "dataset": "miqd",
            "task": source.task,
            "metrics": list(MIQD_METRICS),
            "target_raw": _target_triplet(accuracy_float, consistency_float),
            "score_raw": [
                accuracy_float,
                consistency_float,
                0.5 * (accuracy_float + consistency_float),
            ],
            "ref_path": source.ref_path,
            "dist_path": dist_path,
            "group_id": source.group_id,
            "pair_id": f"{source.task}:{source.source_id}:{filename}",
            "robust_group": _robust_group(source.task, filename),
        }


def split_miqd_sources(
    sources: Sequence[MIQDSource],
    *,
    seed: int = 42,
    ratios: Tuple[float, float] = (0.8, 0.2),
) -> Dict[str, List[MIQDSource]]:
    """Split ImageNet and COCO source groups independently and deterministically."""

    if len(ratios) != 2 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("MIQD split ratios must contain two values summing to 1")
    rng = np.random.default_rng(seed)
    placement: Dict[str, str] = {}
    for domain in ("imagenet", "coco"):
        groups = np.array(sorted({source.group_id for source in sources if source.domain == domain}))
        rng.shuffle(groups)
        boundary = int(len(groups) * ratios[0])
        placement.update({str(group): "train" for group in groups[:boundary]})
        placement.update({str(group): "val" for group in groups[boundary:]})
    result: Dict[str, List[MIQDSource]] = {"train": [], "val": []}
    for source in sources:
        result[placement[source.group_id]].append(source)
    return result


def interleave_miqd_sources(
    sources: Sequence[MIQDSource], *, seed: int | None = None
) -> List[MIQDSource]:
    """Round-robin task sources so capped streams retain all task domains."""

    grouped = {
        task: [source for source in sources if source.task == task]
        for task in ("cls", "det", "ins")
    }
    if seed is not None:
        rng = np.random.default_rng(seed)
        for values in grouped.values():
            rng.shuffle(values)
    queues = {task: iter(values) for task, values in grouped.items()}
    ordered: List[MIQDSource] = []
    active = list(queues)
    while active:
        remaining = []
        for task in active:
            source = next(queues[task], None)
            if source is not None:
                ordered.append(source)
                remaining.append(task)
        active = remaining
    return ordered


def partition_miqd_sources(
    sources: Sequence[MIQDSource],
    worker_id: int,
    worker_count: int,
    *,
    shuffle: bool,
    seed: int,
    balanced_shuffle: bool = False,
) -> List[MIQDSource]:
    """Build disjoint worker partitions from one shared source ordering."""

    ordered = list(sources)
    if shuffle:
        np.random.default_rng(seed).shuffle(ordered)
    else:
        ordered = interleave_miqd_sources(
            ordered, seed=seed if balanced_shuffle else None
        )
    return ordered[worker_id::worker_count]


def allocate_worker_quotas(partition_counts: Sequence[int], max_pairs: int) -> List[int]:
    """Allocate a global cap proportionally without exceeding any partition."""

    counts = np.asarray(partition_counts, dtype=np.int64)
    total = int(counts.sum())
    target = min(int(max_pairs), total)
    if total == 0:
        return [0] * len(counts)
    exact = target * counts.astype(np.float64) / total
    quotas = np.floor(exact).astype(np.int64)
    remainder = target - int(quotas.sum())
    order = sorted(range(len(counts)), key=lambda index: (-(exact[index] - quotas[index]), index))
    for index in order[:remainder]:
        quotas[index] += 1
    return quotas.astype(int).tolist()


def _physical_sources(sources: Iterable[MIQDSource]) -> Dict[Path, MIQDSource]:
    # Detection and instance segmentation use the same distorted image folder.
    return {source.image_dir: source for source in sources}


def _csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def inspect_miqd_completeness(
    root: str | Path,
    sources: Sequence[MIQDSource],
    *,
    smoke_threshold: float = 0.1,
    progress_interval: int = 250,
    progress=None,
    check_images: bool = True,
) -> MIQDCompleteness:
    """Select smoke/full mode and reject silently truncated full downloads.

    A very sparse local checkout is intentional and is allowed to contain only
    part of the labeled images. Once at least ``smoke_threshold`` of source
    folders exist, every label-referenced distorted image is required unless
    the caller explicitly disables rechecking a previously verified dataset.
    With check_images=False, only directory coverage is used to select the
    loading mode; image counts remain unknown rather than reporting success.
    """

    if not 0.0 < smoke_threshold <= 1.0:
        raise ValueError("smoke_threshold must be in (0, 1]")
    root = Path(root)
    physical = _physical_sources(sources)
    expected_sources = len(physical)
    available_sources = sum((root / image_dir).is_dir() for image_dir in physical)
    coverage = available_sources / expected_sources if expected_sources else 0.0
    if not check_images:
        return MIQDCompleteness(
            mode="smoke" if coverage < smoke_threshold else "full",
            expected_sources=expected_sources,
            available_sources=available_sources,
            image_check_performed=False,
        )
    if coverage < smoke_threshold:
        label_files = [root / source.label_paths[0] for source in sources]
        expected_images = None
        available_images = None
        missing_images = None
        if label_files and all(path.is_file() for path in label_files):
            expected_images = sum(_csv_data_rows(path) for path in label_files)
            available_images = sum(
                1
                for source in sources
                if (root / source.image_dir).is_dir()
                for _ in read_miqd_source_rows(root, source, existing_only=True)
            )
            missing_images = expected_images - available_images
        return MIQDCompleteness(
            mode="smoke",
            expected_sources=expected_sources,
            available_sources=available_sources,
            expected_images=expected_images,
            available_images=available_images,
            missing_images=missing_images,
            image_check_performed=expected_images is not None,
        )

    expected_images = 0
    available_images = 0
    missing_examples: List[str] = []
    missing_references = sorted(
        {str(source.ref_path) for source in sources if not (root / source.ref_path).is_file()}
    )
    if missing_references:
        preview = ", ".join(missing_references[:5])
        raise MIQDIncompleteError(
            f"MIQD download is incomplete: {len(missing_references)} reference images "
            f"are missing. Examples: {preview}"
        )
    # Check every task table. Shared DET/INS images are intentionally counted
    # twice here because each task has an independent label integrity contract.
    total_sources = len(sources)
    for source_index, source in enumerate(sources, start=1):
        for row in read_miqd_source_rows(root, source):
            expected_images += 1
            absolute = root / Path(row["dist_path"])
            if absolute.is_file():
                available_images += 1
            elif len(missing_examples) < 5:
                missing_examples.append(str(row["dist_path"]))
        if progress and (
            source_index == 1
            or source_index == total_sources
            or source_index % max(int(progress_interval), 1) == 0
        ):
            progress(
                "completeness",
                source_index,
                total_sources,
                available_images,
                "checking image files",
            )
    missing_images = expected_images - available_images
    if missing_images:
        preview = ", ".join(missing_examples)
        raise MIQDIncompleteError(
            f"MIQD download is incomplete: {missing_images}/{expected_images} labeled images "
            f"are missing. Examples: {preview}"
        )
    return MIQDCompleteness(
        mode="full",
        expected_sources=expected_sources,
        available_sources=available_sources,
        expected_images=expected_images,
        available_images=available_images,
        missing_images=0,
        image_check_performed=True,
    )


def _condition_ids(spec, task: str) -> List[int]:
    return [
        spec.datasets["miqd"] * 100 + spec.tasks[task] * 10 + spec.metrics[metric]
        for metric in MIQD_METRICS
    ]


def scan_miqd_target_statistics(
    root: str | Path,
    sources: Sequence[MIQDSource],
    spec,
    *,
    existing_only: bool = False,
    threshold_quantile: float = 0.75,
    reservoir_size: int = 200_000,
    seed: int = 42,
    progress_interval: int = 250,
    progress=None,
) -> Dict[str, object]:
    """Compute exact moments and reproducible sampled quantiles without a manifest.

    Exact means and variances need only constant memory. Quantiles use bounded
    reservoir samples so the complete multi-million-row dataset does not need to
    be materialized as a DataFrame.
    """

    if reservoir_size <= 0:
        raise ValueError("reservoir_size must be positive")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError("threshold_quantile must be in [0, 1]")
    rng = np.random.default_rng(seed)
    accumulators: Dict[str, Dict[str, object]] = {}
    source_counts: Dict[str, int] = {}
    pair_count = 0
    total_sources = len(sources)
    for source_index, source in enumerate(sources, start=1):
        conditions = _condition_ids(spec, source.task)
        count = 0
        for row in read_miqd_source_rows(root, source, existing_only=existing_only):
            count += 1
            pair_count += 1
            for condition, value in zip(conditions, row["target_raw"]):
                key = str(condition)
                state = accumulators.setdefault(
                    key, {"count": 0, "sum": 0.0, "sum_sq": 0.0, "reservoir": []}
                )
                value = float(value)
                state["count"] += 1
                state["sum"] += value
                state["sum_sq"] += value * value
                reservoir = state["reservoir"]
                if len(reservoir) < reservoir_size:
                    reservoir.append(value)
                else:
                    index = int(rng.integers(0, state["count"]))
                    if index < reservoir_size:
                        reservoir[index] = value
        source_counts[f"{source.task}:{source.source_id}"] = count
        if progress and (
            source_index == 1
            or source_index == total_sources
            or source_index % max(int(progress_interval), 1) == 0
        ):
            progress("scan", source_index, total_sources, pair_count, "metadata scan")

    if pair_count == 0:
        raise MIQDIncompleteError("No readable MIQD image pairs were found")
    target_stats: Dict[str, Dict[str, float | int]] = {}
    thresholds: Dict[str, float] = {}
    for condition, state in accumulators.items():
        count = int(state["count"])
        mean = float(state["sum"]) / count
        variance = max(float(state["sum_sq"]) / count - mean * mean, 0.0)
        target_stats[condition] = {
            "mean": mean,
            "std": max(float(np.sqrt(variance)), 1e-6),
            "count": count,
        }
        thresholds[condition] = float(np.quantile(state["reservoir"], threshold_quantile))
    return {
        "target_stats": target_stats,
        "thresholds": thresholds,
        "source_counts": source_counts,
        "pair_count": pair_count,
        "quantile_reservoir_size": reservoir_size,
    }


def miqd_cache_fingerprint(root: str | Path, sources: Sequence[MIQDSource]) -> str:
    """Fingerprint label metadata and source directories without reading images."""
    root = Path(root)
    entries = []
    for source in sources:
        for relative in (*source.label_paths, source.ref_path, source.image_dir):
            path = root / relative
            if path.is_file():
                stat = path.stat()
                entries.append((str(relative), stat.st_size, stat.st_mtime_ns))
            elif path.is_dir():
                stat = path.stat()
                entries.append((str(relative), "dir", stat.st_mtime_ns))
            else:
                entries.append((str(relative), "missing"))
    payload = json.dumps(sorted(entries), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_or_scan_miqd_metadata(
    root: str | Path,
    sources: Sequence[MIQDSource],
    *,
    cache_dir: str | Path,
    spec=None,
    seed: int = 42,
    threshold_quantile: float = 0.75,
    reservoir_size: int = 200_000,
    existing_only: bool = False,
    progress_interval: int = 250,
    rebuild: bool = False,
    progress=None,
) -> Dict[str, object]:
    """Load source statistics from a fingerprinted JSON cache or scan once."""
    from models.conditioned_iqa import ConditionSpec

    spec = spec or ConditionSpec.default()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = miqd_cache_fingerprint(root, sources)
    parameter_signature = hashlib.sha256(
        json.dumps(
            {"seed": seed, "threshold_quantile": threshold_quantile, "reservoir_size": reservoir_size, "existing_only": existing_only},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    cache_key = hashlib.sha256(f"{fingerprint}:{parameter_signature}".encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"miqd_{cache_key[:16]}.json"
    if not rebuild and cache_file.is_file():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            if payload.get("fingerprint") == fingerprint and payload.get("parameter_signature") == parameter_signature:
                payload["cache_hit"] = True
                return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    total = len(sources)
    if progress:
        progress("scan", 0, total, 0, "starting metadata scan")
    # Keep the scanner's bounded memory behavior and expose source-level progress.
    stats = scan_miqd_target_statistics(
        root,
        sources,
        spec,
        threshold_quantile=threshold_quantile,
        reservoir_size=reservoir_size,
        existing_only=existing_only,
        seed=seed,
        progress_interval=progress_interval,
        progress=progress,
    )
    payload = {
        "fingerprint": fingerprint,
        "parameter_signature": parameter_signature,
        **stats,
        "cache_hit": False,
    }
    temporary = cache_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(cache_file)
    return payload


class MIQDConditionedPairs(IterableDataset):
    """Stream MIQD pairs while returning three conditions per image decode."""

    def __init__(
        self,
        root: str | Path,
        sources: Sequence[MIQDSource],
        spec,
        target_stats: Dict[str, Dict[str, float]],
        *,
        image_size: int = 224,
        existing_only: bool = False,
        shuffle: bool = True,
        balanced_shuffle: bool = False,
        seed: int = 42,
        max_pairs: int | None = None,
        source_counts: Dict[str, int] | None = None,
        source_balanced: bool = False,
        source_mix_size: int = 8,
        rows_per_source_turn: int = 2,
        reference_mode: str = "fr",
    ):
        super().__init__()
        self.root = Path(root)
        self.reference_mode = reference_mode
        self.sources = list(sources)
        self.spec = spec
        self.target_stats = target_stats
        self.existing_only = bool(existing_only)
        self.shuffle = bool(shuffle)
        self.balanced_shuffle = bool(balanced_shuffle)
        self.seed = int(seed)
        self.max_pairs = int(max_pairs) if max_pairs is not None else None
        self.source_counts = dict(source_counts) if source_counts is not None else None
        self._iteration = 0
        self.source_balanced = bool(source_balanced)
        self.source_mix_size = int(source_mix_size)
        self.rows_per_source_turn = int(rows_per_source_turn)
        if self.source_mix_size < 1 or self.rows_per_source_turn < 1:
            raise ValueError("Source mixing sizes must be positive")
        if self.max_pairs is not None and self.max_pairs < 0:
            raise ValueError("max_pairs must be nonnegative or null")
        self._fixed_source_quotas = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if self.source_counts is None:
            self.source_counts = {
                f"{source.task}:{source.source_id}": self._source_count(source)
                for source in self.sources
            }
        total = sum(self._source_count(source) for source in self.sources)
        self.full_length = total
        self._length = min(total, self.max_pairs) if self.max_pairs is not None else total

    def _source_count(self, source: MIQDSource) -> int:
        key = f"{source.task}:{source.source_id}"
        if self.source_counts is not None and key in self.source_counts:
            return int(self.source_counts[key])
        return sum(
            1
            for _ in read_miqd_source_rows(
                self.root, source, existing_only=self.existing_only
            )
        )

    def __len__(self) -> int:
        return self._length

    def source_quotas(self, seed=None):
        """Allocate the cap across sources before workers or task views exist."""
        if self._fixed_source_quotas is not None:
            return dict(self._fixed_source_quotas)
        ordered = sorted(self.sources, key=lambda source: (source.task, source.source_id))
        np.random.default_rng(self.seed if seed is None else seed).shuffle(ordered)
        counts = [self._source_count(source) for source in ordered]
        quotas = allocate_worker_quotas(counts, self._length)
        return {f"{source.task}:{source.source_id}": quota
                for source, quota in zip(ordered, quotas)}

    def for_task(self, task_name):
        """Reuse exactly the parent's evaluation rows, never resample a task."""
        if self.shuffle or not self.source_balanced:
            raise ValueError("Task views require fixed source-balanced evaluation")
        view = copy.copy(self)
        view.sources = [source for source in self.sources if source.task == task_name]
        keys = {f"{source.task}:{source.source_id}" for source in view.sources}
        view.source_counts = {key: value for key, value in self.source_counts.items() if key in keys}
        view._fixed_source_quotas = {key: value for key, value in self.source_quotas().items() if key in keys}
        view.full_length = sum(view.source_counts.values())
        view._length = sum(view._fixed_source_quotas.values())
        view.max_pairs = view._length
        return view

    def sampling_summary(self):
        quotas = self.source_quotas()
        tasks = {}
        for source in self.sources:
            value = quotas[f"{source.task}:{source.source_id}"]
            item = tasks.setdefault(source.task, {"pairs": 0, "sources": 0})
            item["pairs"] += value
            item["sources"] += int(value > 0)
        payload = {"policy": "source_proportional_quota_v1", "seed": self.seed,
                   "pairs": len(self), "tasks": tasks,
                   "selected_pairs_per_source": dict(sorted(quotas.items()))}
        payload["plan_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload

    def _encode_pair(self, source, row, ref):
        with Image.open(self.root / row["dist_path"]) as image:
            dist = self.transform(image.convert("RGB"))
        conditions = _condition_ids(self.spec, source.task)
        raw = torch.tensor(row["target_raw"], dtype=torch.float32)
        means = torch.tensor([self.target_stats[str(c)]["mean"] for c in conditions])
        stds = torch.tensor([self.target_stats[str(c)]["std"] for c in conditions])
        return {
            **({"ref": ref} if self.reference_mode == "fr" else {}), "dist": dist,
            "dataset_id": torch.tensor([self.spec.datasets["miqd"]] * len(MIQD_METRICS)),
            "task_id": torch.tensor([self.spec.tasks[source.task]] * len(MIQD_METRICS)),
            "metric_id": torch.tensor([self.spec.metrics[name] for name in MIQD_METRICS]),
            "condition_id": torch.tensor(conditions),
            "robust_group_id": torch.full((len(MIQD_METRICS),), self.spec.tasks[source.task], dtype=torch.long),
            "target": (raw - means) / stds, "target_raw": raw,
            "pair_id": row["pair_id"], "group_id": source.group_id,
            "dist_path": row["dist_path"].as_posix(),
        }

    def _iter_balanced(self, sources, shared_seed):
        quotas = self.source_quotas(shared_seed)
        selected = [source for source in sources if quotas[f"{source.task}:{source.source_id}"] > 0]
        for start in range(0, len(selected), self.source_mix_size):
            active = []
            for source in selected[start:start + self.source_mix_size]:
                rows = list(read_miqd_source_rows(self.root, source, existing_only=self.existing_only))
                count = quotas[f"{source.task}:{source.source_id}"]
                if len(rows) < count:
                    raise MIQDIncompleteError(f"Source changed after indexing: {source.task}:{source.source_id}")
                # Stable per-source seed makes row membership independent of workers
                # and filtering the full validation stream to an individual task.
                digest = hashlib.sha256(f"{shared_seed}:{source.task}:{source.source_id}".encode()).digest()
                np.random.default_rng(int.from_bytes(digest[:8], "little")).shuffle(rows)
                ref = None
                if self.reference_mode == "fr":
                    with Image.open(self.root / source.ref_path) as image:
                        ref = self.transform(image.convert("RGB"))
                active.append((source, ref, iter(rows[:count])))
            while active:
                remaining = []
                for source, ref, rows in active:
                    for _ in range(self.rows_per_source_turn):
                        row = next(rows, None)
                        if row is None:
                            break
                        yield self._encode_pair(source, row, ref)
                    else:
                        remaining.append((source, ref, rows))
                active = remaining

    def _worker_quota(
        self, worker_id: int, worker_count: int, shared_seed: int
    ) -> int | None:
        if self.max_pairs is None:
            return None
        partition_counts = []
        for index in range(worker_count):
            partition = partition_miqd_sources(
                self.sources,
                index,
                worker_count,
                shuffle=self.shuffle,
                seed=shared_seed,
                balanced_shuffle=self.balanced_shuffle,
            )
            partition_counts.append(sum(self._source_count(source) for source in partition))
        return allocate_worker_quotas(partition_counts, self.max_pairs)[worker_id]

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        if worker:
            # PyTorch assigns base_seed + worker_id. Removing worker_id gives
            # every process the same ordering before disjoint slicing.
            shared_seed = (
                self.seed + int(worker.seed) - worker_id
                if self.shuffle
                else self.seed
            )
        else:
            shared_seed = self.seed + self._iteration if self.shuffle else self.seed
            if self.shuffle:
                self._iteration += 1
        rng = np.random.default_rng(shared_seed + worker_id + 1)
        sources = partition_miqd_sources(
            self.sources,
            worker_id,
            worker_count,
            shuffle=self.shuffle,
            seed=shared_seed,
            balanced_shuffle=self.balanced_shuffle,
        )
        if self.source_balanced:
            yield from self._iter_balanced(sources, shared_seed)
            return
        quota = self._worker_quota(worker_id, worker_count, shared_seed)
        yielded = 0

        for source in sources:
            ref_path = self.root / source.ref_path
            if self.reference_mode == "fr" and not ref_path.is_file():
                if self.existing_only:
                    continue
                raise FileNotFoundError(f"Missing MIQD reference image: {ref_path}")
            rows = list(
                read_miqd_source_rows(
                    self.root, source, existing_only=self.existing_only
                )
            )
            if self.shuffle:
                rng.shuffle(rows)
            ref = None
            if self.reference_mode == "fr":
                with Image.open(ref_path) as image:
                    ref = self.transform(image.convert("RGB"))
            condition_ids = _condition_ids(self.spec, source.task)
            dataset_ids = [self.spec.datasets["miqd"]] * len(MIQD_METRICS)
            task_ids = [self.spec.tasks[source.task]] * len(MIQD_METRICS)
            metric_ids = [self.spec.metrics[name] for name in MIQD_METRICS]
            for row in rows:
                if quota is not None and yielded >= quota:
                    return
                with Image.open(self.root / row["dist_path"]) as image:
                    dist = self.transform(image.convert("RGB"))
                raw = torch.tensor(row["target_raw"], dtype=torch.float32)
                means = torch.tensor(
                    [self.target_stats[str(value)]["mean"] for value in condition_ids],
                    dtype=torch.float32,
                )
                stds = torch.tensor(
                    [self.target_stats[str(value)]["std"] for value in condition_ids],
                    dtype=torch.float32,
                )
                yielded += 1
                yield {
                    **({"ref": ref} if self.reference_mode == "fr" else {}),
                    "dist": dist,
                    "dataset_id": torch.tensor(dataset_ids, dtype=torch.long),
                    "task_id": torch.tensor(task_ids, dtype=torch.long),
                    "metric_id": torch.tensor(metric_ids, dtype=torch.long),
                    "condition_id": torch.tensor(condition_ids, dtype=torch.long),
                    "robust_group_id": torch.full(
                        (len(MIQD_METRICS),), self.spec.tasks[source.task], dtype=torch.long
                    ),
                    "target": (raw - means) / stds,
                    "target_raw": raw,
                    "pair_id": row["pair_id"],
                    "group_id": source.group_id,
                    "dist_path": row["dist_path"].as_posix(),
                }
