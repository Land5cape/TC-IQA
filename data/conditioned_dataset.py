from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ConditionedImagePairs(Dataset):
    def __init__(self, records, spec, image_size=224, reference_mode="fr"):
        self.reference_mode = reference_mode
        self.records=records.reset_index(drop=True); self.spec=spec
        self.transform=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.records)
    def __getitem__(self,index):
        row=self.records.iloc[index]; d,t,m=self.spec.encode(row["dataset"],row["task"],row["metric"])
        item = {'dist':self.transform(Image.open(Path(row["dist_path"])).convert('RGB')),'dataset_id':torch.tensor(d,dtype=torch.long),'task_id':torch.tensor(t,dtype=torch.long),'metric_id':torch.tensor(m,dtype=torch.long),'condition_id':torch.tensor(int(row["condition_id"]),dtype=torch.long),'robust_group_id':torch.tensor(int(row.get("robust_group_id", 0)),dtype=torch.long),'target':torch.tensor(float(row["target"]),dtype=torch.float32),'target_raw':torch.tensor(float(row.get("target_raw", row["target"])),dtype=torch.float32)}
        if self.reference_mode == "fr":
            item['ref'] = self.transform(Image.open(Path(row['ref_path'])).convert('RGB'))
        return item
