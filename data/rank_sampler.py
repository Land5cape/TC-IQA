import math

import numpy as np
from torch.utils.data import Sampler


class GlobalPairBatchSampler(Sampler):
    """Yield batches built from precomputed, same-condition ranking pairs."""

    def __init__(self, records, batch_size, pairs_per_epoch=3600, minimum_gap=0.1, seed=42, pool_size=20000):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("batch_size must be an even number >= 2")
        self.batch_size = batch_size
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.rng = np.random.default_rng(seed)
        records = records.reset_index(drop=True)
        target_column = "target" if "target" in records.columns else "target_raw"
        if target_column not in records.columns:
            raise ValueError("records must contain 'target' or 'target_raw'")
        self.pairs_by_condition = {}
        for condition, group in records.groupby("condition_id"):
            indices = group.index.to_numpy(dtype=np.int64)
            if len(indices) < 2:
                continue
            values = group[target_column].to_numpy(dtype=float)
            pairs = []
            attempts = max(pool_size, len(indices) * 4)
            for _ in range(attempts):
                left, right = self.rng.choice(len(indices), size=2, replace=False)
                if abs(values[left] - values[right]) >= minimum_gap:
                    pairs.append((int(indices[left]), int(indices[right])))
                    if len(pairs) >= pool_size:
                        break
            if pairs:
                self.pairs_by_condition[int(condition)] = pairs
        self.conditions = sorted(self.pairs_by_condition)
        self.eligible_indices = {
            index for pairs in self.pairs_by_condition.values() for pair in pairs for index in pair
        }
        self.last_stats = {"pairs_seen": 0, "coverage": 0.0, "conditions": {}}

    def __iter__(self):
        if not self.conditions:
            return iter(())
        per_condition = self.pairs_per_epoch // len(self.conditions)
        remainder = self.pairs_per_epoch % len(self.conditions)
        selected = []
        for position, condition in enumerate(self.conditions):
            count = per_condition + (position < remainder)
            pool = self.pairs_by_condition[condition]
            choices = self.rng.integers(0, len(pool), size=count)
            selected.extend((condition, pool[int(choice)]) for choice in choices)
        self.rng.shuffle(selected)
        flat = [index for _, pair in selected for index in pair]
        used = set(flat)
        self.last_stats = {
            "pairs_seen": len(selected),
            "coverage": len(used) / max(len(self.eligible_indices), 1),
            "conditions": {str(condition): sum(1 for current, _ in selected if current == condition) for condition in self.conditions},
        }
        return iter(flat[start : start + self.batch_size] for start in range(0, len(flat), self.batch_size))

    def __len__(self):
        return math.ceil(self.pairs_per_epoch * 2 / self.batch_size)
