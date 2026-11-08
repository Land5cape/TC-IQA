from utils.runtime_timing import format_duration


def _value(metrics,name):
    value=metrics.get(name,float('nan'))
    return f"{value:.4f}" if value==value else "  n/a "

def format_epoch_report(epoch,total_epochs,train_loss,metrics,condition_names,is_new_best,best_epoch,best_score,bad_epochs,patience,loss_components=None,selection_score=None,learning_rates=None,train_metrics=None,timing=None,runtime_estimate=None):
    macro=metrics.get('macro',{}); width=78
    status="NEW BEST" if is_new_best else f"best: epoch {best_epoch} / selection {best_score:.4f}"
    selection_text=f"   |   selection {selection_score:.4f}" if selection_score is not None else ""
    lines=["",'='*width,f" EPOCH {epoch:03d} / {total_epochs:03d}   |   train loss {train_loss:.6f}   |   {status}",'='*width,
           f" VAL EVAL  MACRO   SRCC  {_value(macro,'srcc')}   PLCC  {_value(macro,'plcc')}   MAE  {_value(macro,'mae')}   AUROC  {_value(macro,'auroc')}",
           *([f" TRAIN EVAL MACRO   SRCC  {_value(train_metrics.get('macro',{}),'srcc')}   PLCC  {_value(train_metrics.get('macro',{}),'plcc')}   MAE  {_value(train_metrics.get('macro',{}),'mae')}   AUROC  {_value(train_metrics.get('macro',{}),'auroc')}"] if train_metrics else []),'-'*width,
           f" {'Condition':30s}  {'SRCC':>7s}  {'PLCC':>7s}  {'MAE':>7s}  {'AUROC':>7s}"]
    for key,value in metrics.items():
        if key=='macro': continue
        lines.append(f" {condition_names.get(str(key),str(key)):30.30s}  {_value(value,'srcc'):>7s}  {_value(value,'plcc'):>7s}  {_value(value,'mae'):>7s}  {_value(value,'auroc'):>7s}")
    if loss_components:
        if 'plcc_raw' in macro:
            lines.append(f" PLCC: logistic-5 + clip[0,1] | raw {_value(macro,'plcc_raw')} | fit success {_value(macro,'plcc_fit_success')} | selection uses raw PLCC")
        lines.extend(['-'*width,f" Losses   Huber {_value(loss_components,'huber')}   Rank {_value(loss_components,'rank')}   NLL {_value(loss_components,'nll')}   Decision {_value(loss_components,'decision')}",f" Rank pairs (epoch) {int(loss_components.get('rank_pair_count',0))}",f" Sigma    mean {_value(loss_components,'sigma_mean')}   min {_value(loss_components,'sigma_min')}   max {_value(loss_components,'sigma_max')}"])
        task_labels = {'0': 'DET', '1': 'SEG', '2': 'RET', '3': 'CLS', '4': 'INS'}
        for task, stats in loss_components.get('content_gate_by_task', {}).items():
            lines.append(f" Gate TRAIN {task_labels.get(task, task)} mean={stats['mean']:.3f}"
                         f"  <.05={stats['near_zero_fraction']:.1%}  >.95={stats['near_one_fraction']:.1%}")
    if learning_rates:
        lines.append(f" LR       head {learning_rates[0]:.2e}   backbone {learning_rates[1]:.2e}")
    if timing:
        train = timing["train"]
        rate = train.get("pairs_per_second")
        rate_text = f"{rate:.1f}" if rate is not None else "n/a"
        lines.append(f" Time     train {format_duration(train['elapsed_seconds'])} ({rate_text} pairs/s)"
                     f"   train-eval {format_duration(timing['train_eval']['elapsed_seconds'])}"
                     f"   val {format_duration(timing['validation']['elapsed_seconds'])}")
    if runtime_estimate:
        lines.append(f" Full train+val estimate: {format_duration(runtime_estimate['full_epoch_seconds'])} / epoch"
                     " (includes current train-eval; excludes refinement)")
    lines.extend(['-'*width,f" Early stopping: {bad_epochs}/{patience} epochs without selection improvement{selection_text}",'='*width])
    return '\n'.join(lines)
