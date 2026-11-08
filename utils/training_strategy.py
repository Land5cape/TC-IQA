import numpy as np
import torch


def selection_score(metrics, worst_task_weight=0.0):
    macro=metrics.get("macro",{})
    values=[macro.get("srcc",float("nan")),macro.get("plcc_raw",macro.get("plcc",float("nan")))]
    macro_score = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    if not np.isfinite(macro_score) or worst_task_weight <= 0:
        return macro_score
    task_scores = []
    for key, value in metrics.items():
        if key == "macro":
            continue
        score = np.nanmean([value.get("srcc", np.nan), value.get("plcc_raw", value.get("plcc", np.nan))])
        if np.isfinite(score):
            task_scores.append(float(score))
    if not task_scores:
        return macro_score
    weight = min(max(float(worst_task_weight), 0.0), 1.0)
    return float((1.0 - weight) * macro_score + weight * min(task_scores))


def build_optimizer(model,head_learning_rate,backbone_learning_rate,weight_decay):
    backbone_parameters=list(model.encoder.last_stage.parameters())
    backbone_ids={id(parameter) for parameter in backbone_parameters}
    head_parameters=[parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    return torch.optim.AdamW(
        [
            {"params":head_parameters,"lr":head_learning_rate,"name":"head"},
            {"params":backbone_parameters,"lr":backbone_learning_rate,"name":"backbone"},
        ],
        weight_decay=weight_decay,
    )


def set_backbone_phase(model,epoch,unfreeze_epoch):
    model.set_last_stage_trainable(epoch>=unfreeze_epoch)


def update_bad_epochs(is_new_best, epoch, start_epoch, bad_epochs):
    if is_new_best or epoch < start_epoch:
        return 0
    return bad_epochs + 1
