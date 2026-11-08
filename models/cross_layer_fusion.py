"""Task-guided local evidence fusion before spatial pooling (NR only)."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossLayerLocalFusion(nn.Module):
    def __init__(self, channels, dim, condition_dim, output_dim, grid, heads=1):
        super().__init__()
        self.grid = grid
        self.heads = heads
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, dim, 1), nn.GroupNorm(1, dim), nn.GELU())
            for c in channels
        ])
        self.local_filters = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.GELU())
            for _ in range(2)
        ])
        self.detail_fusion = nn.Sequential(nn.Conv2d(2*dim, dim, 1), nn.GroupNorm(1, dim), nn.GELU())
        self.semantic_fusion = nn.Sequential(nn.Conv2d(2*dim, dim, 1), nn.GroupNorm(1, dim), nn.GELU())
        self.evidence = nn.Sequential(nn.Linear(3*dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.semantic_key = nn.Linear(dim, dim, bias=False)
        self.task_query = nn.Linear(condition_dim, dim*heads, bias=False)
        self.condition_value = nn.Linear(condition_dim, dim)
        self.value = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.output = nn.Linear(dim*heads, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _align(self, feature):
        if min(feature.shape[-2:]) >= self.grid:
            return F.adaptive_avg_pool2d(feature, (self.grid, self.grid))
        return F.interpolate(feature, size=(self.grid, self.grid), mode='bilinear', align_corners=False)

    def encode(self, levels):
        if len(levels) != 4:
            raise ValueError('cross_layer requires four feature levels')
        projected=[]
        for index,(projection,feature) in enumerate(zip(self.projections,levels)):
            feature=projection(feature)
            if index<2:
                feature=self.local_filters[index](feature)
            projected.append(self._align(feature))
        detail=self.detail_fusion(torch.cat(projected[:2],dim=1)).flatten(2).transpose(1,2)
        semantic=self.semantic_fusion(torch.cat(projected[2:],dim=1)).flatten(2).transpose(1,2)
        evidence=self.evidence(torch.cat([detail,semantic,detail*semantic],dim=-1))
        return evidence,self.semantic_key(semantic)

    def pool(self, encoded, task_condition, full_condition):
        evidence,keys=encoded
        # Task decides locations; metric conditions modulate evidence before pooling.
        query=self.task_query(task_condition)
        if self.heads == 1:
            logits=torch.einsum('bcd,bnd->bcn',query,keys)/math.sqrt(keys.shape[-1])
        else:
            query=query.reshape(*query.shape[:2],self.heads,keys.shape[-1])
            logits=torch.einsum('bchd,bnd->bchn',query,keys)/math.sqrt(keys.shape[-1])
        weights=torch.softmax(logits,dim=-1)
        values=self.value(F.gelu(evidence[:,None]+self.condition_value(full_condition)[:,:,None]))
        if self.heads == 1:
            pooled=(weights[...,None]*values).sum(-2)
        else:
            # Retain each region summary as a distinct block. Reuse values,
            # avoiding a B*C*heads*tokens*dim expanded activation tensor.
            pooled=torch.einsum('bchn,bcnd->bchd',weights,values).flatten(-2)
        return self.output(pooled),weights
