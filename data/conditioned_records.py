from pathlib import Path
import numpy as np
import pandas as pd

def expand_mpd_records(records):
    rows=[]
    for _, row in records.iterrows():
        for task in ("det", "seg", "ret"):
            dist_type = row.get("dist_type", "unknown")
            strength = row.get("strength", "unknown")
            strength_label = f"S{int(strength)}" if str(strength).isdigit() else str(strength)
            item = {"ref_path":row["ref_path"],"dist_path":row.get("dist_path",row.get("dis_path")),"dataset":"mpd","task":task,"metric":"utility","target":float(row[f"{task}_drop"]),"group_id":row.get("ref_name",row.get("ref_path")),"pair_id":row.get("dis_name", row.get("dist_path", row.get("dis_path")))}
            if "dist_type" in row and "strength" in row:
                item["robust_group"] = f"{task}:{dist_type}"
            rows.append(item)
    return pd.DataFrame(rows)

def prepare_mpd_manifest(dataset_root, output):
    from .mpd_pair_index import MPDPairIndex
    root=Path(dataset_root); output=Path(output)
    records=MPDPairIndex.from_csv(root/"quality_release.csv",root).records
    output.parent.mkdir(parents=True,exist_ok=True)
    expand_mpd_records(records).to_csv(output,index=False)
    return output

def expand_miqd_records(records, root=".", task="cls", ref_col="src", dist_col="dist", group_col=None):
    root=Path(root); rows=[]
    for _, row in records.iterrows():
        group=row[group_col] if group_col else row[ref_col]
        for metric in ("accuracy","consistency","composite"):
            rows.append({"ref_path":str(root/str(row[ref_col])),"dist_path":str(root/str(row[dist_col])),"dataset":"miqd","task":task,"metric":metric,"target":round(1.0-float(row[metric]),8),"group_id":group,"pair_id":str(row[dist_col]),"robust_group":f"{task}:unknown:unknown"})
    return pd.DataFrame(rows)

def split_by_group(records, group_col="group_id", seed=42, ratios=(.7,.1,.2)):
    if len(ratios) not in (2,3) or not np.isclose(sum(ratios),1.0): raise ValueError("ratios must contain 2 or 3 values summing to 1")
    rng=np.random.default_rng(seed); groups=np.array(sorted(records[group_col].unique())); rng.shuffle(groups); a=int(len(groups)*ratios[0])
    if len(ratios)==2:
        return {"train":records[records[group_col].isin(groups[:a])].reset_index(drop=True),"val":records[records[group_col].isin(groups[a:])].reset_index(drop=True)}
    b=a+int(len(groups)*ratios[1])
    return {"train":records[records[group_col].isin(groups[:a])].reset_index(drop=True),"val":records[records[group_col].isin(groups[a:b])].reset_index(drop=True),"test":records[records[group_col].isin(groups[b:])].reset_index(drop=True)}


def split_by_row(records, seed=42, ratios=(.8, .2)):
    """Randomly split individual image-pair rows without source-group blocking."""
    if len(ratios) not in (2, 3) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain 2 or 3 values summing to 1")
    rng = np.random.default_rng(seed)
    positions = np.arange(len(records))
    rng.shuffle(positions)
    first = int(len(records) * ratios[0])
    if len(ratios) == 2:
        return {
            "train": records.iloc[positions[:first]].reset_index(drop=True),
            "val": records.iloc[positions[first:]].reset_index(drop=True),
        }
    second = first + int(len(records) * ratios[1])
    return {
        "train": records.iloc[positions[:first]].reset_index(drop=True),
        "val": records.iloc[positions[first:second]].reset_index(drop=True),
        "test": records.iloc[positions[second:]].reset_index(drop=True),
    }


def split_by_pair(records, seed=42, ratios=(.8, .2), pair_col="pair_id"):
    """Randomly split unique image pairs while keeping their task rows together."""
    if pair_col not in records.columns:
        raise ValueError(f"Missing pair column: {pair_col}")
    pairs = records[[pair_col]].drop_duplicates().reset_index(drop=True)
    pair_splits = split_by_row(pairs, seed=seed, ratios=ratios)
    return {
        name: records[records[pair_col].isin(part[pair_col])].reset_index(drop=True)
        for name, part in pair_splits.items()
    }
