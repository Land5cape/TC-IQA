from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import re

_PATTERN=re.compile(r"^(?P<ref>.+)_(?P<dist>T\d{2})_(?P<strength>S\d)\.png$")
DROP_COLUMNS=("det_drop","seg_drop","ret_drop")

@dataclass
class MPDPairIndex:
    records: pd.DataFrame

    @classmethod
    def from_csv(cls,csv_path,dataset_root):
        root=Path(dataset_root); raw=pd.read_csv(csv_path)
        required={"dis_name","ref_name","det","seg","ret"}
        missing=required-set(raw.columns)
        if missing: raise ValueError(f"Missing MPD columns: {sorted(missing)}")
        rows=[]
        for row in raw.to_dict("records"):
            match=_PATTERN.match(str(row["dis_name"]))
            if not match: raise ValueError(f"Malformed MPD distorted filename: {row['dis_name']}")
            rows.append({"dis_name":row["dis_name"],"ref_name":row["ref_name"],
                         "ref_path":str(root/"ref"/row["ref_name"]),"dis_path":str(root/"dis"/row["dis_name"]),
                         "dist_type":match.group("dist"),"strength":int(match.group("strength")[1:]),
                         "det_drop":1-float(row["det"]),"seg_drop":1-float(row["seg"]),"ret_drop":1-float(row["ret"])})
        return cls(pd.DataFrame(rows))
