def task_branch_modules(model, task_id):
    task_id = int(task_id)
    return {
        "task_adapter_down": model.task_adapter_down[task_id],
        "task_adapter_up": model.task_adapter_up[task_id],
        "mean_heads": model.mean_heads[task_id],
        "sigma_heads": model.sigma_heads[task_id],
    }


def set_task_refinement_phase(model, task_id):
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules = task_branch_modules(model, task_id)
    trainable = []
    for module in modules.values():
        for parameter in module.parameters():
            parameter.requires_grad = True
            trainable.append(parameter)
    return trainable


def set_task_refinement_train_mode(model, task_id):
    model.eval()
    for module in task_branch_modules(model, task_id).values():
        module.train()


def task_branch_state_dict(model, task_id):
    state = {}
    for name, module in task_branch_modules(model, task_id).items():
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value.detach().cpu().clone()
    return state


def load_task_branch_state_dict(model, task_id, state):
    for name, module in task_branch_modules(model, task_id).items():
        prefix = f"{name}."
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)})
