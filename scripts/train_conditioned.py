import argparse
import copy
import csv
import gzip
import hashlib
import os
import platform
import shutil
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from data.conditioned_dataset import ConditionedImagePairs
from data.conditioned_records import prepare_mpd_manifest, split_by_group, split_by_pair, split_by_row
from data.miqd_conditioned import (
    MIQDConditionedPairs,
    discover_miqd_sources,
    inspect_miqd_completeness,
    load_or_scan_miqd_metadata,
    scan_miqd_target_statistics,
    split_miqd_sources,
)
from data.rank_sampler import GlobalPairBatchSampler
from data.target_normalization import compute_target_stats, normalized_thresholds, standardize_records
from evaluation.conformal import fit_conditional_conformal
from evaluation.metrics import evaluate_predictions, macro_average
from losses.conditioned_objectives import conditioned_loss, loss_weights_for_epoch, scheduled_weight
from losses.group_robust import kl_group_weights, mix_group_weights, scheduled_mix_strength
from models.conditioned_iqa import ConditionSpec, ConditionedFRIQA, failure_probability
from scripts.pack_result import pack_result
from utils.console_report import format_epoch_report
from utils.condition_batch import flatten_condition_batch
from utils.experiment_io import save_json, unique_timestamped_output_dir
from utils.seed import seed_everything
from utils.runtime_timing import estimate_full_epoch, format_duration, phase_timing
from utils.task_refinement import (
    load_task_branch_state_dict,
    set_task_refinement_phase,
    set_task_refinement_train_mode,
    task_branch_state_dict,
)
from utils.training_strategy import build_optimizer, selection_score, set_backbone_phase, update_bad_epochs


def resolve_project_path(value, project_root=PROJECT_ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def resolve_config_path(value):
    return resolve_project_path(value)


def resolve_training_device(requested_device="auto"):
    """Validate the visible CUDA index before loading data or creating outputs."""
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested {device}, but CUDA is not available. "
                "Check the PyTorch CUDA installation and GPU visibility, "
                "or explicitly set device: cpu for a local smoke test."
            )
        count = torch.cuda.device_count()
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= count:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
            raise RuntimeError(
                f"Requested cuda:{index}, but only {count} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={visible}. Indices refer to visible GPUs; "
                "if only the second physical GPU is exposed, use device: cuda:0. "
                "Otherwise expose both GPUs and use device: cuda:1."
            )
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def resolve_dataset_root(config, project_root=PROJECT_ROOT):
    environment_name = "MIQD_DATASET_ROOT" if config.get("dataset_type") == "miqd" else "MPD_DATASET_ROOT"
    value = os.environ.get(environment_name, config["dataset_root"])
    return resolve_project_path(value, project_root=project_root)


def prepare_records(path, spec, seed, split_ratios=(0.8, 0.2), split_mode="row"):
    records = pd.read_csv(path)
    required = {"ref_path", "dist_path", "dataset", "task", "metric", "target", "group_id"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if "pair_id" not in records.columns:
        records["pair_id"] = records["dist_path"].astype(str)
    encoded = [spec.encode(row.dataset, row.task, row.metric) for _, row in records.iterrows()]
    records["condition_id"] = [dataset * 100 + task * 10 + metric for dataset, task, metric in encoded]
    if "robust_group" not in records.columns:
        records["robust_group"] = records.apply(lambda row: f"{row['task']}:unknown:unknown", axis=1)
    robust_names = {name: index for index, name in enumerate(sorted(records["robust_group"].astype(str).unique()))}
    records["robust_group"] = records["robust_group"].astype(str)
    records["robust_group_id"] = records["robust_group"].map(robust_names).astype(int)
    splitter = split_by_pair if split_mode == "pair" else (split_by_row if split_mode == "row" else split_by_group)
    splits = splitter(records, seed=seed, ratios=tuple(split_ratios))
    for split in splits.values():
        split.attrs["robust_group_names"] = {str(index): name for name, index in robust_names.items()}
    return splits


def threshold_tensor(thresholds, device):
    largest = max(int(key) for key in thresholds)
    result = torch.full((largest + 1,), 0.5, dtype=torch.float32, device=device)
    for key, value in thresholds.items():
        result[int(key)] = float(value)
    return result


def collect_predictions(model, loader, thresholds, target_stats, device, max_batches=None,
                        timing=None, prediction_path=None, progress_label=None):
    model.eval()
    buckets = {}
    started = time.perf_counter()
    pairs_seen = 0
    with ExitStack() as stack, torch.no_grad():
        writer = None
        if prediction_path is not None:
            path = Path(prediction_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(gzip.open(path, "wt", encoding="utf-8", newline=""))
            writer = csv.DictWriter(handle, fieldnames=[
                "pair_id", "group_id", "dist_path", "condition_id", "target",
                "prediction", "sigma", "failure_probability", "threshold", "failure",
            ])
            writer.writeheader()
        for step,batch in enumerate(loader,start=1):
            if max_batches is not None and step>max_batches:
                break
            condition_id = batch["condition_id"].to(device)
            output = model(
                batch["ref"].to(device, non_blocking=True) if "ref" in batch else None,
                batch["dist"].to(device, non_blocking=True),
                batch["dataset_id"].to(device),
                batch["task_id"].to(device),
                batch["metric_id"].to(device),
            )
            output, flat_batch = flatten_condition_batch(output, batch)
            condition_id = flat_batch["condition_id"].to(device)
            conditions = condition_id.detach().cpu().tolist()
            means = torch.tensor([target_stats[str(c)]["mean"] for c in conditions], device=device)
            stds = torch.tensor([target_stats[str(c)]["std"] for c in conditions], device=device)
            mean_raw = output["mean"] * stds + means
            sigma_raw = output["sigma"] * stds
            threshold = thresholds[condition_id]
            risk = failure_probability(mean_raw, sigma_raw, threshold)
            # Transfer a whole batch once instead of synchronizing CUDA for every scalar.
            predictions = mean_raw.detach().cpu().tolist()
            sigmas = sigma_raw.detach().cpu().tolist()
            risks = risk.detach().cpu().tolist()
            targets = flat_batch["target_raw"].detach().cpu().tolist()
            cutoff = threshold.detach().cpu().tolist()
            batch_pairs = int(batch["dist"].shape[0])
            per_pair = len(conditions) // batch_pairs
            for index, condition in enumerate(conditions):
                bucket = buckets.setdefault(int(condition), [[], [], [], []])
                bucket[0].append(targets[index])
                bucket[1].append(predictions[index])
                bucket[2].append(risks[index])
                bucket[3].append(sigmas[index])
                if writer is not None:
                    pair_index = index // per_pair
                    writer.writerow({
                        "pair_id": batch.get("pair_id", [""] * batch_pairs)[pair_index],
                        "group_id": batch.get("group_id", [""] * batch_pairs)[pair_index],
                        "dist_path": batch.get("dist_path", [""] * batch_pairs)[pair_index],
                        "condition_id": condition, "target": targets[index],
                        "prediction": predictions[index], "sigma": sigmas[index],
                        "failure_probability": risks[index], "threshold": cutoff[index],
                        "failure": int(targets[index] >= cutoff[index]),
                    })
            pairs_seen += batch_pairs
            if progress_label and (step == 1 or step % 250 == 0 or step == len(loader)):
                elapsed = time.perf_counter() - started
                rate = pairs_seen / max(elapsed, 1e-9)
                print(f"{progress_label}: {pairs_seen}/{len(loader.dataset)} pairs"
                      f" | {rate:.1f} pairs/s | elapsed {format_duration(elapsed)}"
                      f" | ETA {format_duration(max(len(loader.dataset) - pairs_seen, 0) / rate)}",
                      flush=True)
    if timing is not None:
        timing.update(phase_timing(pairs_seen, time.perf_counter() - started))
    return buckets


def metrics_from_predictions(buckets, thresholds, logistic_fits=None):
    result = {}
    for condition, (target, prediction, risk, _sigma) in buckets.items():
        condition_threshold = float(thresholds[condition])
        failure = (np.asarray(target) >= condition_threshold).astype(int)
        details = {} if logistic_fits is not None else None
        result[str(condition)] = evaluate_predictions(target, prediction, failure, risk, logistic_details=details)
        if logistic_fits is not None:logistic_fits[str(condition)] = details
    if result:
        result["macro"] = macro_average(result)
    return result


def evaluate(model, loader, thresholds, target_stats, device, max_batches=None,
             timing=None, prediction_path=None, progress_label=None):
    started = time.perf_counter()
    buckets = collect_predictions(model, loader, thresholds, target_stats, device, max_batches,
                                  timing=timing, prediction_path=prediction_path,
                                  progress_label=progress_label)
    result = metrics_from_predictions(buckets, thresholds)
    if timing is not None:
        timing.update(phase_timing(timing["pairs"], time.perf_counter() - started))
    return result


def train_one_epoch(model, loader, rank_loader, optimizer, scaler, thresholds, device, epoch, config, group_weights=None):
    started = time.perf_counter()
    pairs_seen = 0
    model.train()
    refinement_task_id = config.get("refinement_task_id")
    if refinement_task_id is not None:
        set_task_refinement_train_mode(model, refinement_task_id)
    nll_weight, decision_weight = loss_weights_for_epoch(
        epoch,
        config.get("loss_warmup_epochs", 3),
        config.get("loss_ramp_epochs", 3),
        config.get("nll_weight", 0.01),
        config.get("decision_weight", 0.1),
    )
    rank_weight = scheduled_weight(
        epoch,
        config.get("rank_warmup_epochs", 0),
        config.get("rank_ramp_epochs", 1),
        config.get("rank_weight", 0.0),
    )
    totals = {name: 0.0 for name in ("total", "huber", "nll", "rank", "decision", "sigma_mean", "sigma_min", "sigma_max")}
    rank_pair_total = 0
    gate_totals = {}
    group_totals, group_counts = {}, {}
    seen = 0
    max_batches = config.get("max_batches")
    rank_iterator = iter(rank_loader) if rank_loader is not None else None
    def group_factor(group):
        key = str(int(group))
        if not group_weights:
            return 1.0
        priors = config.get("group_priors", {})
        return float(group_weights.get(key, 0.0)) / max(float(priors.get(key, 1.0)), 1e-8)
    for step, batch in enumerate(loader, start=1):
        if max_batches is not None and step > max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(
                batch["ref"].to(device, non_blocking=True) if "ref" in batch else None,
                batch["dist"].to(device, non_blocking=True),
                batch["dataset_id"].to(device),
                batch["task_id"].to(device),
                batch["metric_id"].to(device),
            )
            output, flat_batch = flatten_condition_batch(output, batch)
            if "content_gate" in output:
                gates = output["content_gate"].detach().float().reshape(-1, output["content_gate"].shape[-1])
                gate_tasks = batch["task_id"].to(gates.device).reshape(-1)
                for task in torch.unique(gate_tasks).tolist():
                    values = gates[gate_tasks == task]
                    stats = gate_totals.setdefault(str(task), {"sum": 0.0, "low": 0, "high": 0, "count": 0})
                    stats["sum"] += values.sum().item()
                    stats["low"] += (values < .05).sum().item()
                    stats["high"] += (values > .95).sum().item()
                    stats["count"] += values.numel()
            factors = torch.tensor([group_factor(group) for group in flat_batch["robust_group_id"]], device=device)
            base_losses = conditioned_loss(
                output,
                flat_batch["target"].to(device),
                flat_batch["condition_id"].to(device),
                thresholds,
                huber_weight=config.get("huber_weight", 1.0),
                nll_weight=nll_weight,
                rank_weight=rank_weight if rank_iterator is None else 0.0,
                decision_weight=decision_weight,
                reduction="none",
                sample_weights=factors,
                rank_group_id=(flat_batch.get("rank_group_id")
                    if rank_iterator is None
                    and config.get("rank_scope", "within_condition") == "within_source"
                    else None),
            )
            sample_group_ids = flat_batch["robust_group_id"].to(device)
            raw_sample_total = base_losses["huber_values"] * config.get("huber_weight", 1.0) + nll_weight * base_losses["nll_values"] + decision_weight * base_losses["decision_values"]
            sample_total = raw_sample_total * factors
            for group in torch.unique(sample_group_ids).detach().cpu().tolist():
                mask = sample_group_ids == int(group)
                group_totals[str(int(group))] = group_totals.get(str(int(group)), 0.0) + float(raw_sample_total[mask].detach().sum())
                group_counts[str(int(group))] = group_counts.get(str(int(group)), 0) + int(mask.sum())
            if rank_iterator is not None and config.get("rank_weight", 0.0) > 0:
                try:
                    rank_batch = next(rank_iterator)
                except StopIteration:
                    rank_iterator = iter(rank_loader)
                    rank_batch = next(rank_iterator)
                rank_output = model(
                    rank_batch["ref"].to(device, non_blocking=True) if "ref" in rank_batch else None,
                    rank_batch["dist"].to(device, non_blocking=True),
                    rank_batch["dataset_id"].to(device),
                    rank_batch["task_id"].to(device),
                    rank_batch["metric_id"].to(device),
                )
                rank_output, flat_rank_batch = flatten_condition_batch(rank_output, rank_batch)
                rank_losses = conditioned_loss(
                    rank_output,
                    flat_rank_batch["target"].to(device),
                    flat_rank_batch["condition_id"].to(device),
                    thresholds,
                    huber_weight=0.0,
                    nll_weight=0.0,
                    rank_weight=rank_weight,
                    decision_weight=0.0,
                )
                rank_pair_total += rank_losses["rank_pair_count"]
                losses = dict(base_losses)
                losses["total"] = base_losses["total"] + rank_weight * rank_losses["rank"]
                losses["rank"] = rank_losses["rank"]
            else:
                losses = base_losses
                rank_pair_total += base_losses["rank_pair_count"]
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        seen += 1
        pairs_seen += int(batch["dist"].shape[0])
        for name in totals:
            totals[name] += float(losses[name].detach())
        if step == 1 or step % config.get("log_interval", 50) == 0 or step == len(loader):
            elapsed = time.perf_counter() - started
            rate = pairs_seen / max(elapsed, 1e-9)
            eta = max(len(loader.dataset) - pairs_seen, 0) / rate
            full_pairs = getattr(loader.dataset, "full_length", len(loader.dataset))
            print(
                f"epoch {epoch:03d}/{config['epochs']:03d}  batch {step:04d}/{len(loader):04d}"
                f"  total={totals['total']/seen:.5f}  huber={totals['huber']/seen:.5f}"
                f"  rank={totals['rank']/seen:.5f}  rank_pairs={rank_pair_total}"
                f"  | {rate:.1f} pairs/s  elapsed {format_duration(elapsed)}"
                f"  ETA {format_duration(eta)}  full-train~{format_duration(full_pairs / rate)}",
                flush=True,
            )
    result = {name: value / max(seen, 1) for name, value in totals.items()}
    result["rank_pair_count"] = rank_pair_total
    if gate_totals:
        result["content_gate_by_task"] = {
            key: {"mean": s["sum"] / s["count"], "near_zero_fraction": s["low"] / s["count"],
                  "near_one_fraction": s["high"] / s["count"], "channel_values": s["count"]}
            for key, s in gate_totals.items()}
    result["group_risk"] = {group: group_totals[group] / max(group_counts[group], 1) for group in group_totals}
    result["group_counts"] = group_counts
    result["scheduled_weights"] = {"nll": nll_weight, "rank": rank_weight, "decision": decision_weight}
    result["timing"] = phase_timing(pairs_seen, time.perf_counter() - started)
    return result


def make_condition_names(train_records, spec):
    names = {}
    for _, row in train_records.drop_duplicates(["dataset", "task", "metric"]).iterrows():
        dataset, task, metric = spec.encode(row.dataset, row.task, row.metric)
        names[str(dataset * 100 + task * 10 + metric)] = f"{row.dataset.upper()} / {row.task.upper()} / {row.metric}"
    return names


def prepare_miqd_runtime(dataset_root, spec, config, device, cache_dir=None):
    print("[MIQD 1/4] Discovering official label files...", flush=True)
    sources = discover_miqd_sources(dataset_root)
    if not sources:
        raise RuntimeError(f"No MIQD label files found under {dataset_root}")
    def completeness_report(stage, current, total, pairs, message):
        if current == 1 or current == total or current % max(int(config.get("miqd_progress_interval", 250)), 1) == 0:
            percent = 100.0 * current / max(total, 1)
            print(f"[MIQD 2/4] {message}: {current}/{total} sources ({percent:.1f}%), images={pairs}", flush=True)

    completeness = inspect_miqd_completeness(
        dataset_root,
        sources,
        smoke_threshold=float(config.get("miqd_smoke_threshold", 0.1)),
        progress_interval=int(config.get("miqd_progress_interval", 250)),
        progress=completeness_report,
        check_images=bool(config.get("miqd_check_completeness", True)),
    )
    if not config.get("miqd_check_completeness", True):
        print(
            "[MIQD 2/4] Image completeness recheck skipped "
            "(miqd_check_completeness=false); checking source folders only.",
            flush=True,
        )
    print(
        f"[MIQD 2/4] Data mode={completeness.mode}; "
        f"source folders {completeness.available_sources}/{completeness.expected_sources}",
        flush=True,
    )
    existing_only = completeness.mode == "smoke"
    if existing_only:
        sources = [
            source
            for source in sources
            if (dataset_root / source.image_dir).is_dir()
            and (dataset_root / source.ref_path).is_file()
        ]
    splits = split_miqd_sources(
        sources,
        seed=config["seed"],
        ratios=tuple(config.get("split_ratios", [0.8, 0.2])),
    )
    if not splits["train"] or not splits["val"]:
        raise RuntimeError(
            "MIQD source split is empty. Smoke mode needs at least two available "
            "sources in one or more source domains."
        )
    def report(stage, current, total, pairs, message):
        if current == 0 or current == total or current % max(int(config.get("miqd_progress_interval", 250)), 1) == 0:
            percent = 100.0 * current / max(total, 1)
            print(f"[MIQD 3/4] {message}: {current}/{total} sources ({percent:.1f}%), pairs={pairs}", flush=True)

    statistics = load_or_scan_miqd_metadata(
        dataset_root,
        splits["train"],
        cache_dir=cache_dir or (PROJECT_ROOT / "data" / "cache"),
        spec=spec,
        threshold_quantile=float(config.get("threshold_quantile", 0.75)),
        reservoir_size=int(config.get("miqd_quantile_reservoir_size", 200_000)),
        existing_only=existing_only,
        seed=config["seed"],
        progress_interval=int(config.get("miqd_progress_interval", 250)),
        rebuild=bool(config.get("miqd_rebuild_cache", False)),
        progress=report,
    )
    target_stats = statistics["target_stats"]
    raw_thresholds = statistics["thresholds"]
    train_counts = statistics["source_counts"]

    common = {
        "root": dataset_root,
        "spec": spec,
        "target_stats": target_stats,
        "image_size": config["image_size"],
        "reference_mode": config.get("reference_mode", "fr"),
        "existing_only": existing_only,
        "seed": config["seed"],
        "source_mix_size": config.get("miqd_source_mix_size", 8),
        "rows_per_source_turn": config.get("miqd_rows_per_source_turn", 2),
    }
    datasets = {
        "train": MIQDConditionedPairs(
            sources=splits["train"],
            shuffle=True,
            max_pairs=config.get("miqd_train_pairs_per_epoch"),
            source_counts=train_counts,
            source_balanced=config.get("miqd_source_balanced", True),
            **common,
        ),
        "train_eval": MIQDConditionedPairs(
            sources=splits["train"],
            shuffle=False,
            balanced_shuffle=True,
            max_pairs=config.get("miqd_train_eval_pairs"),
            source_counts=train_counts,
            source_balanced=True,
            **common,
        ),
        "val": MIQDConditionedPairs(
            sources=splits["val"],
            shuffle=False,
            balanced_shuffle=True,
            max_pairs=config.get("miqd_validation_pairs"),
            source_balanced=True,
            **common,
        ),
    }
    loader_options = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        name: DataLoader(dataset, **loader_options)
        for name, dataset in datasets.items()
    }
    loaders["rank"] = None

    active_tasks = sorted({source.task for source in splits["train"]})
    condition_names = {}
    for task in active_tasks:
        for metric in ("accuracy", "consistency", "composite"):
            dataset_id, task_id, metric_id = spec.encode("miqd", task, metric)
            condition = dataset_id * 100 + task_id * 10 + metric_id
            condition_names[str(condition)] = f"MIQD / {task.upper()} / {metric}"
    robust_group_names = {
        str(spec.tasks[task]): f"MIQD / {task.upper()}" for task in active_tasks
    }
    task_counts = {}
    for task in active_tasks:
        condition = str(
            spec.datasets["miqd"] * 100
            + spec.tasks[task] * 10
            + spec.metrics["accuracy"]
        )
        task_counts[str(spec.tasks[task])] = int(target_stats[condition]["count"])
    total = sum(task_counts.values())
    group_priors = {key: value / total for key, value in task_counts.items()}

    split_counts = {}
    for split_name, dataset_name in (("train", "train"), ("val", "val")):
        dataset = datasets[dataset_name]
        task_pairs = {}
        for source in splits[split_name]:
            key = f"{source.task}:{source.source_id}"
            task_pairs[source.task] = task_pairs.get(source.task, 0) + int(
                dataset.source_counts.get(key, 0)
            )
        split_counts[split_name] = {
            "available_pairs": int(dataset.full_length),
            "sampled_pairs": int(len(dataset)),
            "tasks": {
                task: {
                    "pairs": count,
                    "conditions": {
                        metric: count
                        for metric in ("accuracy", "consistency", "composite")
                    },
                }
                for task, count in sorted(task_pairs.items())
            },
        }
    dataset_info = {
        "dataset_type": "miqd",
        "mode": completeness.mode,
        "completeness": asdict(completeness),
        "source_splits": {name: len(values) for name, values in splits.items()},
        "split_counts": split_counts,
        "evaluation_sampling": {
            "policy": "fixed source quotas and per-source row seeds; shared refinement views",
            "train_eval_pairs": len(datasets["train_eval"]),
            "validation_pairs": len(datasets["val"]),
            "validation_sources": datasets["val"].sampling_summary()["tasks"],
        },
        "statistics_train_pairs": statistics["pair_count"],
        "target_direction": "loss = 1 - official score",
    }
    print(f"[MIQD 4/4] Loaders ready; cache_hit={statistics.get('cache_hit', False)}", flush=True)
    return {
        "splits": splits,
        "loaders": loaders,
        "target_stats": target_stats,
        "raw_thresholds": raw_thresholds,
        "condition_names": condition_names,
        "robust_group_names": robust_group_names,
        "group_priors": group_priors,
        "dataset_info": dataset_info,
        "rank_sampler": None,
        "existing_only": existing_only,
    }


def task_refinement_loaders(splits, task_name, spec, config, device, target_stats=None, full_loaders=None):
    if not isinstance(splits["train"], pd.DataFrame):
        if full_loaders is None:
            raise ValueError("MIQD refinement needs main evaluation loaders to preserve sample membership")
        task_splits = {
            name: [source for source in sources if source.task == task_name]
            for name, sources in splits.items()
        }
        common = {
            "root": Path(config["dataset_root"]),
            "spec": spec,
            "target_stats": target_stats,
            "image_size": config["image_size"],
            "reference_mode": config.get("reference_mode", "fr"),
            "existing_only": config.get("miqd_existing_only", False),
            "seed": config["seed"],
            "source_balanced": config.get("miqd_source_balanced", True),
            "source_mix_size": config.get("miqd_source_mix_size", 8),
            "rows_per_source_turn": config.get("miqd_rows_per_source_turn", 2),
        }
        datasets = {
            "train": MIQDConditionedPairs(
                sources=task_splits["train"],
                shuffle=True,
                max_pairs=config.get("miqd_refinement_pairs_per_epoch"),
                source_counts=full_loaders["train"].dataset.source_counts,
                **common,
            ),
            "train_eval": full_loaders["train_eval"].dataset.for_task(task_name),
            "val": full_loaders["val"].dataset.for_task(task_name),
        }
        options = {
            "batch_size": config["batch_size"],
            "num_workers": config.get("num_workers", 0),
            "pin_memory": device.type == "cuda",
        }
        return {
            name: DataLoader(dataset, **options) for name, dataset in datasets.items()
        } | {"rank": None, "rank_sampler": None}

    task_splits = {
        name: records[records["task"] == task_name].reset_index(drop=True)
        for name, records in splits.items()
    }
    options = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
    }
    datasets = {
        name: ConditionedImagePairs(records, spec, config["image_size"], config.get("reference_mode", "fr"))
        for name, records in task_splits.items()
    }
    rank_sampler = GlobalPairBatchSampler(
        task_splits["train"],
        config["batch_size"],
        config.get("task_refinement_rank_pairs_per_epoch", 1200),
        config.get("rank_minimum_gap", 0.1),
        config["seed"],
        config.get("rank_pair_pool_size", 20000),
    )
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **options),
        "rank": DataLoader(
            datasets["train"],
            batch_sampler=rank_sampler,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
        ),
        "train_eval": DataLoader(datasets["train"], shuffle=False, **options),
        "val": DataLoader(datasets["val"], shuffle=False, **options),
        "rank_sampler": rank_sampler,
    }


def run_task_refinement(
    model,
    joint_checkpoint,
    splits,
    spec,
    config,
    thresholds,
    eval_thresholds,
    target_stats,
    device,
    output,
    full_loaders,
):
    joint_state = copy.deepcopy(joint_checkpoint["model"])
    histories = {}
    summaries = {}
    best_branches = {}
    refinement_epochs = int(config.get("task_refinement_epochs", 6))
    patience = int(config.get("task_refinement_patience", 3))
    if isinstance(splits["train"], pd.DataFrame):
        available_tasks = set(splits["train"]["task"])
        candidates = ("det", "seg", "ret")
    else:
        available_tasks = {source.task for source in splits["train"]}
        candidates = ("cls", "det", "ins")
    task_names = [name for name in candidates if name in available_tasks]

    for task_name in task_names:
        task_id = spec.tasks[task_name]
        if config.get("dataset_type") == "miqd":
            condition_id = [
                spec.datasets["miqd"] * 100 + task_id * 10 + spec.metrics[metric]
                for metric in ("accuracy", "consistency", "composite")
            ]
        else:
            condition_id = task_id * 10 + spec.metrics["utility"]
        model.load_state_dict(joint_state)
        parameters = set_task_refinement_phase(model, task_id)
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config.get("task_refinement_learning_rate", config.get("head_learning_rate", 3e-5))),
            weight_decay=float(config.get("task_refinement_weight_decay", config.get("weight_decay", 1e-3))),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(refinement_epochs, 1),
            eta_min=float(config.get("minimum_learning_rate", 1e-7)),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("amp", True)) and device.type == "cuda")
        loaders = task_refinement_loaders(
            splits, task_name, spec, config, device, target_stats, full_loaders=full_loaders
        )
        refinement_config = dict(config)
        refinement_config.update(
            {
                "refinement_task_id": task_id,
                "loss_warmup_epochs": 0,
                "loss_ramp_epochs": 1,
                "rank_warmup_epochs": 0,
                "rank_ramp_epochs": 1,
            }
        )
        baseline_metrics = evaluate(model, loaders["val"], eval_thresholds, target_stats, device)
        baseline_score = selection_score(baseline_metrics)
        best_score = baseline_score
        best_epoch = 0
        best_branch = task_branch_state_dict(model, task_id)
        bad_epochs = 0
        task_history = []
        print(f"\n{'=' * 72}\nTask refinement: {task_name.upper()} | baseline={baseline_score:.6f}\n{'=' * 72}", flush=True)

        for epoch in range(1, refinement_epochs + 1):
            epoch_started = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                loaders["train"],
                loaders["rank"],
                optimizer,
                scaler,
                thresholds,
                device,
                epoch,
                refinement_config,
            )
            train_eval_timing, val_timing = {}, {}
            train_eval_metrics = evaluate(model, loaders["train_eval"], eval_thresholds, target_stats, device,
                                          timing=train_eval_timing, progress_label=f"Refine {task_name} train-eval")
            validation_metrics = evaluate(model, loaders["val"], eval_thresholds, target_stats, device,
                                          timing=val_timing, progress_label=f"Refine {task_name} validation")
            score = selection_score(validation_metrics)
            is_new_best = bool(np.isfinite(score) and score > best_score)
            if is_new_best:
                best_score = score
                best_epoch = epoch
                best_branch = task_branch_state_dict(model, task_id)
                bad_epochs = 0
            else:
                bad_epochs += 1
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["total"],
                "train_components": train_metrics,
                "train_eval": train_eval_metrics,
                "validation": validation_metrics,
                "selection_score": score,
                "is_new_best": is_new_best,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "timing": {"train": train_metrics["timing"], "train_eval": train_eval_timing,
                           "validation": val_timing, "epoch_seconds": time.perf_counter() - epoch_started},
                "rank_sampling": (
                    loaders["rank_sampler"].last_stats
                    if loaders["rank_sampler"] is not None
                    else {"mode": "within_batch"}
                ),
            }
            task_history.append(row)
            histories[task_name] = task_history
            save_json(output / "refinement_history.json", histories)
            print(
                f"Refine {task_name.upper()} {epoch:02d}/{refinement_epochs:02d} | "
                f"score={score:.6f} | best={best_score:.6f}@{best_epoch} | "
                f"bad={bad_epochs}/{patience}",
                flush=True,
            )
            scheduler.step()
            if bad_epochs >= patience:
                break

        best_branches[task_name] = best_branch
        summaries[task_name] = {
            "condition_id": condition_id,
            "baseline_score": baseline_score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "completed_epochs": len(task_history),
            "improvement": best_score - baseline_score,
            "evaluation_pairs": len(loaders["val"].dataset),
            "evaluation_source": "main_validation_task_view" if config.get("dataset_type") == "miqd" else "task_validation",
        }
        save_json(output / "refinement_summary.json", summaries)

    model.load_state_dict(joint_state)
    for task_name, branch in best_branches.items():
        load_task_branch_state_dict(model, spec.tasks[task_name], branch)
    final_train_metrics = evaluate(model, full_loaders["train_eval"], eval_thresholds, target_stats, device)
    final_validation_metrics = evaluate(model, full_loaders["val"], eval_thresholds, target_stats, device)
    save_json(output / "training_metrics.json", final_train_metrics)
    save_json(output / "validation_metrics.json", final_validation_metrics)
    save_json(output / "best_metrics.json", final_validation_metrics)
    save_json(output / "refinement_history.json", histories)
    save_json(output / "refinement_summary.json", summaries)
    final_checkpoint = {
        "model": model.state_dict(),
        "config": config,
        "thresholds": joint_checkpoint["thresholds"],
        "target_stats": target_stats,
        "joint_best_epoch": joint_checkpoint["epoch"],
        "joint_selection_score": joint_checkpoint["selection_score"],
        "refinement_summary": summaries,
    }
    torch.save(final_checkpoint, output / "best.pt")
    return summaries, final_validation_metrics


def save_summary(output, config, completed_epochs, stop_reason, best_epoch, best_score, elapsed_seconds=None):
    save_json(
        output / "run_summary.json",
        {
            "requested_epochs": config["epochs"],
            "completed_epochs": completed_epochs,
            "stop_reason": stop_reason,
            "best_epoch": best_epoch,
            "best_selection_score": best_score if np.isfinite(best_score) else None,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def main():
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config.setdefault("pair_features", "symmetric")
    config.setdefault("semantic_bottleneck_dim", None)
    config.setdefault("local_grid_size", None)
    config.setdefault("spatial_task_adapter_dim", None)
    config.setdefault("spatial_interaction_dim", None)
    config.setdefault("spatial_interaction_grid", 4)
    config.setdefault("backbone_weights", None)
    config.setdefault("reference_mode", "fr")
    config.setdefault("scale_fusion", "attention_sum")
    config.setdefault("cross_layer_dim", None)
    config.setdefault("cross_layer_grid", 28)
    config.setdefault("cross_layer_heads", 1)
    config.setdefault("cross_layer_fusion", "add")
    config.setdefault("rank_scope", "within_condition")
    if config["rank_scope"] not in ("within_condition", "within_source"):
        raise ValueError("rank_scope must be within_condition or within_source")
    config["evaluation_plcc"] = "five_parameter_curve_fit_monotonic_clip01_per_condition"
    config["selection_plcc"] = "raw"
    if config["scale_fusion"] not in ("attention_sum", "attention_concat"):
        raise ValueError("scale_fusion must be attention_sum or attention_concat")
    if config["reference_mode"] not in ("fr", "nr"):
        raise ValueError("reference_mode must be fr or nr")
    if config.get("backbone") == "dinov2_vits14" and config["image_size"] % 14:
        raise ValueError("DINOv2 image_size must be a multiple of 14, e.g. 224")
    requested_device = config.get("device", "auto")
    device = resolve_training_device(requested_device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    print(
        f"[Device] requested={requested_device} | selected={device} "
        f"| GPU={gpu_name or 'none'} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    seed_everything(config["seed"])
    spec = ConditionSpec.default()
    dataset_root = resolve_dataset_root(config)
    dataset_type = config.get("dataset_type", "mpd").lower()
    output = unique_timestamped_output_dir(resolve_project_path(
        config["output_dir"].replace("{backbone}", config.get("backbone", "resnet50")).replace("{reference_mode}", config["reference_mode"])
    ))

    if dataset_type == "mpd":
        manifest = resolve_project_path(config["manifest"])
        manifest = prepare_mpd_manifest(dataset_root, manifest)
        splits = prepare_records(
            manifest,
            spec,
            config["seed"],
            config.get("split_ratios", [0.8, 0.2]),
            config.get("split_mode", "pair"),
        )
        train_records = splits["train"]
        robust_group_names = {
            str(int(group_id)): str(name)
            for group_id, name in train_records[
                ["robust_group_id", "robust_group"]
            ].drop_duplicates().itertuples(index=False)
        }
        raw_thresholds = {
            str(int(condition)): float(
                group.target.quantile(config.get("threshold_quantile", 0.75))
            )
            for condition, group in train_records.groupby("condition_id")
        }
        target_stats = compute_target_stats(train_records)
        splits = {
            name: standardize_records(records, target_stats)
            for name, records in splits.items()
        }
        loader_options = {
            "batch_size": config["batch_size"],
            "num_workers": config.get("num_workers", 0),
            "pin_memory": device.type == "cuda",
        }
        datasets = {
            name: ConditionedImagePairs(records, spec, config["image_size"], config["reference_mode"])
            for name, records in splits.items()
        }
        rank_sampler = GlobalPairBatchSampler(
            splits["train"],
            config["batch_size"],
            config.get("rank_pairs_per_epoch", 3600),
            config.get("rank_minimum_gap", 0.1),
            config["seed"],
            config.get("rank_pair_pool_size", 20000),
        )
        loaders = {
            "train": DataLoader(datasets["train"], shuffle=True, **loader_options),
            "rank": DataLoader(
                datasets["train"],
                batch_sampler=rank_sampler,
                num_workers=loader_options["num_workers"],
                pin_memory=loader_options["pin_memory"],
            ),
            "train_eval": DataLoader(
                datasets["train"], shuffle=False, **loader_options
            ),
            **{
                name: DataLoader(dataset, shuffle=False, **loader_options)
                for name, dataset in datasets.items()
                if name != "train"
            },
        }
        condition_names = make_condition_names(train_records, spec)
        group_counts = train_records["robust_group_id"].value_counts().sort_index()
        group_priors = {
            str(int(group)): float(count / len(train_records))
            for group, count in group_counts.items()
        }
        dataset_info = {
            "dataset_type": "mpd",
            "manifest": str(manifest.resolve()),
            "splits": {key: len(value) for key, value in splits.items()},
        }
        config["manifest"] = str(manifest)
        miqd_existing_only = False
    elif dataset_type == "miqd":
        runtime = prepare_miqd_runtime(
            dataset_root,
            spec,
            config,
            device,
            cache_dir=resolve_project_path(config.get("miqd_cache_dir", "data/cache")),
        )
        splits = runtime["splits"]
        loaders = runtime["loaders"]
        target_stats = runtime["target_stats"]
        raw_thresholds = runtime["raw_thresholds"]
        condition_names = runtime["condition_names"]
        robust_group_names = runtime["robust_group_names"]
        group_priors = runtime["group_priors"]
        dataset_info = runtime["dataset_info"]
        rank_sampler = runtime["rank_sampler"]
        miqd_existing_only = runtime["existing_only"]
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")

    eval_thresholds = threshold_tensor(raw_thresholds, device)
    normalized = normalized_thresholds(raw_thresholds, target_stats)
    thresholds = threshold_tensor(normalized, device)
    config["dataset_root"] = str(dataset_root)
    config["output_dir"] = str(output)
    config["group_priors"] = group_priors
    config["miqd_existing_only"] = miqd_existing_only
    save_json(output / "config_resolved.json", config)
    save_json(output / "dataset_info.json", dataset_info)
    if dataset_type == "miqd":
        save_json(output / "evaluation_sampling.json", {
            name: loaders[name].dataset.sampling_summary() for name in ("train_eval", "val")
        })
    save_json(
        output / "system_info.json",
        {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "requested_device": requested_device,
            "device": str(device),
            "cuda": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "cuda_device_index": device.index if device.type == "cuda" else None,
            "cuda_visible_device_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    save_json(output / "robust_group_names.json", robust_group_names)
    save_json(output / "target_stats.json", target_stats)
    save_json(output / "thresholds.json", raw_thresholds)
    save_json(output / "condition_names.json", condition_names)
    group_weights = dict(group_priors)
    robust_weights = dict(group_priors)
    save_json(output / "group_priors.json", group_priors)

    model = ConditionedFRIQA(
        pretrained=config.get("pretrained", True),
        freeze_backbone=True,
        spec=spec,
        backbone=config.get("backbone", "resnet50"),
        scale_dim=config.get("scale_dim", 128),
        hidden_dim=config.get("hidden_dim", 512),
        adapter_dim=config.get("adapter_dim", 64),
        task_adapter_dim=config.get("task_adapter_dim", 16),
        sigma_min=config.get("sigma_min", 0.03),
        pair_features=config["pair_features"],
        semantic_bottleneck_dim=config["semantic_bottleneck_dim"],
        local_grid_size=config["local_grid_size"],
        spatial_task_adapter_dim=config["spatial_task_adapter_dim"],
        spatial_interaction_dim=config["spatial_interaction_dim"],
        spatial_interaction_grid=config["spatial_interaction_grid"],
        backbone_weights=config["backbone_weights"],
        reference_mode=config["reference_mode"],
        scale_fusion=config["scale_fusion"],
        cross_layer_dim=config["cross_layer_dim"],
        cross_layer_grid=config["cross_layer_grid"],
        cross_layer_heads=config["cross_layer_heads"],
        cross_layer_fusion=config["cross_layer_fusion"],
    ).to(device)
    save_json(output / "model_info.json", {
        "class": type(model).__name__, "backbone": config.get("backbone", "resnet50"),
        "reference_mode": model.reference_mode, "scale_fusion": model.scale_fusion,
        "fusion_weight_shape": list(model.fusion[0].weight.shape),
        "cross_layer_dim": model.cross_layer_dim, "cross_layer_grid": model.cross_layer_grid,
        "cross_layer_heads": model.cross_layer_heads,
        "cross_layer_fusion": model.cross_layer_fusion,
        "cross_layer_enabled": hasattr(model, "cross_layer"),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("models/conditioned_iqa.py", "models/cross_layer_fusion.py", "scripts/train_conditioned.py")
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "cross_layer_parameters": sum(p.numel() for p in model.cross_layer.parameters()) if hasattr(model, "cross_layer") else 0,
    })
    optimizer = build_optimizer(
        model,
        config.get("head_learning_rate", 1e-4),
        config.get("backbone_learning_rate", 5e-6),
        config.get("weight_decay", 5e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config.get("minimum_learning_rate", 1e-7)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("amp", True)) and device.type == "cuda")
    history = []
    best_score = -float("inf")
    best_epoch = None
    best_metrics = None
    bad_epochs = 0
    stop_reason = "completed"
    startup_seconds = time.perf_counter() - run_started
    runtime_estimate = None

    print(
        f"\nDataset: {dataset_type.upper()} | Backbone: {config.get('backbone','resnet50')} "
        f"| Reference mode: {model.reference_mode.upper()} "
        f"| Scale fusion: {model.scale_fusion} "
        f"| Cross-layer: {f'{model.cross_layer_dim}d/{model.cross_layer_grid}x{model.cross_layer_grid}' if model.cross_layer_dim is not None else 'off'} "
        f"| Region heads: {model.cross_layer_heads} | Rank scope: {config['rank_scope']} "
        f"| Local fusion: {model.cross_layer_fusion} "
        f"| Pair features: {model.pair_features} "
        f"| Semantic bottleneck: {(model.semantic_bottleneck_dim if model.semantic_bottleneck_dim is not None else 'off') if model.reference_mode == 'fr' else 'n/a (single-image projection)'} "
        f"| Local grid: {model.local_grid_size if model.local_grid_size is not None else 'off'} "
        f"| Spatial task adapter: {model.spatial_task_adapter_dim if model.spatial_task_adapter_dim is not None else 'off'} "
        f"| Pair spatial interaction: "
        f"{f'{model.spatial_interaction_dim}d/{model.spatial_interaction_grid}x{model.spatial_interaction_grid}' if model.spatial_interaction_dim is not None else 'off'} "
        f"| Device: {device} | Train/Val pairs: "
        f"{len(loaders['train'].dataset)}/{len(loaders['val'].dataset)}",
        flush=True,
    )
    try:
        for epoch in range(1, config["epochs"] + 1):
            epoch_started = time.perf_counter()
            set_backbone_phase(model, epoch, config.get("unfreeze_epoch", 6))
            dro_strength = scheduled_mix_strength(
                epoch,
                config.get("dro_warmup_epochs", 5),
                config.get("dro_ramp_epochs", 5),
                config.get("dro_mix_weight", 0.2),
            )
            group_weights = mix_group_weights(group_priors, robust_weights, dro_strength)
            train_metrics = train_one_epoch(model, loaders["train"], loaders["rank"], optimizer, scaler, thresholds, device, epoch, config, group_weights)
            robust_weights = kl_group_weights(
                train_metrics.get("group_risk", {}),
                group_priors,
                eta=config.get("dro_eta", 2.0),
                minimum_weight=config.get("dro_minimum_group_weight", 0.001),
            )
            train_eval_timing, val_timing = {}, {}
            train_eval_metrics = evaluate(model, loaders["train_eval"], eval_thresholds, target_stats, device,
                                          config.get("train_eval_max_batches"), timing=train_eval_timing,
                                          progress_label="Train evaluation")
            validation_metrics = evaluate(model, loaders["val"], eval_thresholds, target_stats, device,
                                          config.get("max_eval_batches"), timing=val_timing,
                                          progress_label="Validation")
            phase_times = {"train": train_metrics["timing"], "train_eval": train_eval_timing,
                           "validation": val_timing}
            runtime_estimate = estimate_full_epoch(
                train_metrics["timing"], train_eval_timing, val_timing,
                getattr(loaders["train"].dataset, "full_length", len(loaders["train"].dataset)),
                getattr(loaders["val"].dataset, "full_length", len(loaders["val"].dataset)),
            )
            if runtime_estimate:
                runtime_estimate.update({"based_on_epoch": epoch, "startup_seconds": startup_seconds,
                                         "sampled_phase_timing": phase_times})
                save_json(output / "runtime_estimate.json", runtime_estimate)
            score = selection_score(validation_metrics, config.get("selection_worst_task_weight", 0.0))
            selection_active = epoch >= config.get("selection_start_epoch", 1)
            is_new_best = bool(selection_active and np.isfinite(score) and score > best_score)
            checkpoint = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "config": config, "thresholds": raw_thresholds, "target_stats": target_stats, "epoch": epoch, "selection_score": score,
            }
            torch.save(checkpoint, output / "last.pt")
            if is_new_best:
                best_score = score
                best_epoch = epoch
                best_metrics = validation_metrics
                bad_epochs = 0
                torch.save(checkpoint, output / "best.pt")
                save_json(output / "best_metrics.json", best_metrics)
            bad_epochs = update_bad_epochs(
                is_new_best,
                epoch,
                config.get("early_stopping_start_epoch", 1),
                bad_epochs,
            )
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            row = {
                "epoch": epoch, "train_loss": train_metrics["total"], "train_components": train_metrics,
                "train_eval": train_eval_metrics,
                "timing": {**phase_times, "epoch_seconds": time.perf_counter() - epoch_started},
                "rank_sampling": rank_sampler.last_stats if rank_sampler is not None else {"mode": "within_batch", "scope": config['rank_scope'], "effective_pairs": train_metrics.get('rank_pair_count', 0)},
                "group_weights": group_weights, "dro_strength": dro_strength, "learning_rates": learning_rates, "selection_score": score, **validation_metrics,
            }
            history.append(row)
            save_json(output / "history.json", history)
            save_json(output / "training_metrics.json", train_eval_metrics)
            save_json(output / "validation_metrics.json", validation_metrics)
            save_summary(output, config, epoch, "running", best_epoch, best_score, time.perf_counter() - run_started)
            print(
                format_epoch_report(
                    epoch, config["epochs"], train_metrics["total"], validation_metrics, condition_names,
                    is_new_best, best_epoch, best_score, bad_epochs, config.get("patience", 10),
                    loss_components=train_metrics, selection_score=score, learning_rates=learning_rates,
                    train_metrics=train_eval_metrics,
                    timing=phase_times, runtime_estimate=runtime_estimate,
                ),
                flush=True,
            )
            scheduler.step()
            if epoch >= config.get("early_stopping_start_epoch", 1) and bad_epochs >= config.get("patience", 10):
                stop_reason = "early_stopping"
                break
    except KeyboardInterrupt:
        stop_reason = "manual_interrupt"
        print("\nTraining interrupted. Saving completed epochs and result bundle...", flush=True)
    finally:
        post_started = time.perf_counter()
        diagnostics = {"best_epoch": best_epoch, "selection_score": best_score if np.isfinite(best_score) else None}
        checkpoint_path = output / "best.pt"
        if checkpoint_path.exists() and stop_reason != "manual_interrupt" and config.get("task_refinement_enabled", False):
            shutil.copy2(checkpoint_path, output / "joint_best.pt")
            joint_checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(joint_checkpoint["model"])
            save_json(output / "joint_best_metrics.json", best_metrics)
            refinement_summary, refinement_metrics = run_task_refinement(
                model,
                joint_checkpoint,
                splits,
                spec,
                config,
                thresholds,
                eval_thresholds,
                target_stats,
                device,
                output,
                loaders,
            )
            diagnostics["refinement"] = refinement_summary
            diagnostics["refinement_metrics"] = refinement_metrics
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model"])
            diagnostics["train"] = evaluate(
                model, loaders["train_eval"], eval_thresholds, target_stats, device,
                config.get("diagnostic_max_batches"),
                progress_label="Final train evaluation",
            )
        else:
            diagnostics["status"] = "no_best_checkpoint"
        calibration_buckets = collect_predictions(
            model, loaders["val"], eval_thresholds, target_stats, device,
            config.get("max_eval_batches"),
            prediction_path=(output / "validation_predictions.csv.gz")
            if config.get("save_validation_predictions", dataset_type == "miqd") else None,
            progress_label="Final validation export",
        )
        logistic_fits = {}
        diagnostics["val"] = metrics_from_predictions(calibration_buckets, eval_thresholds, logistic_fits)
        save_json(output / "plcc_mapping.json", {
            "formula": "clip(y_mean + y_std * (b1*(sigmoid(b2*(z-b3))-.5)+b4*z+b5), 0, 1); z=(prediction-x_mean)/x_std",
            "fit_scope": "each condition of the current validation set, using its labels",
            "selection": "unchanged: average of SRCC and raw PLCC",
            "optimizer": "scipy.optimize.curve_fit(method=trf), deterministic multistart; beta1>=0, beta2>0, beta4>=0",
            "note": "Constraints enforce a nondecreasing map because prediction/target are both task loss. This is evaluation-set post-hoc fitting, not independent calibration.",
            "conditions": logistic_fits,
        })
        diagnostics["conformal_coverage_scope"] = "in-sample calibration coverage on the model-selection validation set"
        if checkpoint_path.exists():
            save_json(output / "validation_metrics.json", diagnostics["val"])
            save_json(output / "best_metrics.json", diagnostics["val"])
        calibration_targets = []
        calibration_means = []
        calibration_sigmas = []
        calibration_conditions = []
        for condition, (targets, means, _risks, sigmas) in calibration_buckets.items():
            calibration_targets.extend(targets)
            calibration_means.extend(means)
            calibration_sigmas.extend(sigmas)
            calibration_conditions.extend([int(condition)] * len(targets))
        save_json(
            output / "conformal_calibration.json",
            fit_conditional_conformal(
                calibration_targets,
                calibration_means,
                calibration_sigmas,
                calibration_conditions,
                alpha=config.get("conformal_alpha", 0.1),
            ),
        )
        save_json(output / "diagnostics.json", diagnostics)
        elapsed = time.perf_counter() - run_started
        if runtime_estimate:
            runtime_estimate.update({"run_elapsed_seconds": elapsed,
                                     "post_training_seconds": time.perf_counter() - post_started})
            save_json(output / "runtime_estimate.json", runtime_estimate)
        save_summary(output, config, len(history), stop_reason, best_epoch, best_score, elapsed)
        bundle = pack_result(output)
        print(f"Result bundle: {bundle.resolve()}", flush=True)


if __name__ == "__main__":
    main()
