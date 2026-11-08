import numpy as np


def conformal_upper_bound(residuals, alpha=0.1):
    """Finite-sample split-conformal upper residual summary."""
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "alpha": float(alpha), "quantile": None, "coverage": None}
    level = min(max(1.0 - float(alpha), 0.0), 1.0)
    rank = min(max(int(np.ceil((values.size + 1) * level)) - 1, 0), values.size - 1)
    quantile = float(np.sort(values)[rank])
    return {"count": int(values.size), "alpha": float(alpha), "quantile": quantile, "coverage": float(np.mean(values <= quantile))}


def fit_conditional_conformal(targets, means, sigmas, conditions, alpha=0.1, epsilon=1e-6):
    result = {}
    conditions = np.asarray(conditions)
    for condition in sorted(set(map(int, conditions))):
        mask = conditions == condition
        residuals = (np.asarray(targets)[mask] - np.asarray(means)[mask]) / (np.asarray(sigmas)[mask] + epsilon)
        result[str(condition)] = conformal_upper_bound(residuals, alpha=alpha)
    return result
