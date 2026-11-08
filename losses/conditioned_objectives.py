import math

import torch
import torch.nn.functional as F

from models.conditioned_iqa import failure_probability


def _same_condition_rank(mean, target, condition_id, minimum_gap=0.01,
                         rank_group_id=None, return_count=False):
    losses = []
    pair_count = 0
    if rank_group_id is not None and len(rank_group_id) != mean.numel():
        raise ValueError("rank_group_id must align with flattened condition predictions")
    for condition in torch.unique(condition_id):
        indices = torch.nonzero(condition_id == condition, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        if rank_group_id is None:
            condition_groups = [indices]
        else:
            buckets = {}
            for index in indices.detach().cpu().tolist():
                buckets.setdefault(rank_group_id[index], []).append(index)
            condition_groups = [torch.tensor(value, dtype=torch.long, device=mean.device)
                                for value in buckets.values() if len(value) >= 2]
        for group_indices in condition_groups:
            left, right = torch.triu_indices(group_indices.numel(), group_indices.numel(), offset=1, device=mean.device)
            target_delta = target[group_indices[left]] - target[group_indices[right]]
            keep = target_delta.abs() >= minimum_gap
            if keep.any():
                prediction_delta = mean[group_indices[left]][keep] - mean[group_indices[right]][keep]
                losses.append(F.softplus(-target_delta[keep].sign() * prediction_delta).mean())
                pair_count += int(keep.sum().detach().cpu())
    rank = torch.stack(losses).mean() if losses else mean.new_zeros(())
    return (rank, pair_count) if return_count else rank


def loss_weights_for_epoch(epoch, warmup_epochs, ramp_epochs, nll_weight, decision_weight):
    if epoch <= warmup_epochs:
        return 0.0, 0.0
    progress = min(1.0, (epoch - warmup_epochs) / max(ramp_epochs, 1))
    return nll_weight * progress, decision_weight * progress


def scheduled_weight(epoch, warmup_epochs, ramp_epochs, weight):
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / max(ramp_epochs, 1))
    return float(weight) * progress


def conditioned_loss(
    outputs,
    target,
    condition_id,
    thresholds,
    huber_weight=1.0,
    nll_weight=0.01,
    rank_weight=0.3,
    decision_weight=0.1,
    reduction="mean",
    sample_weights=None,
    rank_group_id=None,
):
    mean = outputs["mean"]
    sigma = outputs["sigma"].clamp_min(1e-6)
    threshold = thresholds.to(mean.device)[condition_id]
    huber_values = F.smooth_l1_loss(mean, target, reduction="none")
    nll_values = torch.log(sigma) + 0.5 * ((mean - target) / sigma).pow(2) + 0.5 * math.log(2.0 * math.pi)
    rank, rank_pair_count = _same_condition_rank(
        mean, target, condition_id, rank_group_id=rank_group_id, return_count=True)
    failure = (target >= threshold).float()
    risk = failure_probability(mean.float(), sigma.float(), threshold.float()).clamp(1e-6, 1.0 - 1e-6)
    risk_logit = torch.logit(risk)
    decision_values = F.binary_cross_entropy_with_logits(risk_logit, failure.float(), reduction="none")
    if reduction == "none":
        weights = torch.ones_like(target) if sample_weights is None else sample_weights.to(target.device)
        normalizer = weights.sum().clamp_min(1e-6)
        huber = (huber_values * weights).sum() / normalizer
        nll = (nll_values * weights).sum() / normalizer
        decision = (decision_values * weights).sum() / normalizer
    else:
        huber = huber_values.mean()
        nll = nll_values.mean()
        decision = decision_values.mean()
    total = huber_weight * huber + nll_weight * nll + rank_weight * rank + decision_weight * decision
    return {
        "total": total,
        "huber": huber,
        "nll": nll,
        "rank": rank,
        "rank_pair_count": rank_pair_count,
        "decision": decision,
        "sigma_mean": sigma.mean(),
        "sigma_min": sigma.min(),
        "sigma_max": sigma.max(),
        "huber_values": huber_values,
        "nll_values": nll_values,
        "decision_values": decision_values,
    }
