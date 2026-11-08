import math


def kl_group_weights(risks, priors, eta=1.0, minimum_weight=1e-3):
    """Return normalized exponential-tilted group weights."""
    groups = sorted(set(risks) & set(priors))
    if not groups:
        return {}
    scores = {group: math.log(max(float(priors[group]), 1e-12)) + float(eta) * float(risks[group]) for group in groups}
    maximum = max(scores.values())
    raw = {group: math.exp(scores[group] - maximum) for group in groups}
    total = sum(raw.values())
    weights = {group: value / total for group, value in raw.items()}
    if minimum_weight > 0:
        weights = {group: max(value, minimum_weight) for group, value in weights.items()}
        total = sum(weights.values())
        weights = {group: value / total for group, value in weights.items()}
    return weights


def mix_group_weights(uniform, robust, strength):
    strength = min(max(float(strength), 0.0), 1.0)
    groups = sorted(set(uniform) | set(robust))
    values = {group: (1.0 - strength) * float(uniform.get(group, 0.0)) + strength * float(robust.get(group, 0.0)) for group in groups}
    total = sum(values.values())
    return {group: value / total for group, value in values.items()} if total > 0 else values


def scheduled_mix_strength(epoch, warmup_epochs, ramp_epochs, maximum):
    if epoch <= warmup_epochs:
        return 0.0
    progress = min(1.0, (epoch - warmup_epochs) / max(ramp_epochs, 1))
    return float(maximum) * progress
