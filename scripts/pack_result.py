import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

def pack_result(output_dir):
    output_dir=Path(output_dir)
    bundle=output_dir/"result_bundle.zip"
    include=("config_resolved.json","system_info.json","split_info.json","history.json","training_metrics.json","validation_metrics.json","best_metrics.json","joint_best_metrics.json","refinement_history.json","refinement_summary.json","diagnostics.json","run_summary.json","thresholds.json","target_stats.json","condition_names.json","robust_group_names.json","group_priors.json","conformal_calibration.json","test_metrics.json","dataset_info.json","runtime_estimate.json","evaluation_sampling.json","validation_predictions.csv.gz")
    with ZipFile(bundle,"w",ZIP_DEFLATED) as z:
        architecture = output_dir / "model_info.json"
        if architecture.exists(): z.write(architecture, architecture.name)
        mapping = output_dir / "plcc_mapping.json"
        if mapping.exists(): z.write(mapping, mapping.name)
        for name in include:
            path=output_dir/name
            if path.exists(): z.write(path,name)
    return bundle

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("output_dir"); args=p.parse_args()
    print(pack_result(args.output_dir))
