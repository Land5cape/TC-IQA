"""四层特征接口；预训练权重路径相对项目根目录解析。"""
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def load_backbone_weights(model, path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.is_file():
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)


class FrozenFeatureEncoder(nn.Module):
    def set_all_trainable(self, trainable):
        for parameter in self.parameters():
            parameter.requires_grad = trainable
        self.train(self.training)

    def set_last_stage_trainable(self, trainable):
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.last_stage.parameters():
            parameter.requires_grad = trainable
        self.train(self.training)

    def train(self, mode=True):
        # Even on partial unfreezing, fixed features must remain deterministic.
        if all(p.requires_grad for p in self.parameters()):
            return super().train(mode)
        super().train(False)
        self.training = mode
        if mode and any(p.requires_grad for p in self.last_stage.parameters()):
            self.last_stage.train(True)
        return self


class ConvNeXtTinyPyramid(FrozenFeatureEncoder):
    channels = (96, 192, 384, 768)

    def __init__(self, pretrained=True, freeze_backbone=True, weights_path=None):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained and not weights_path else None
        net = models.convnext_tiny(weights=weights)
        if weights_path:
            load_backbone_weights(net, weights_path)
        layers = list(net.features.children())
        self.stages = nn.ModuleList([
            nn.Sequential(*layers[i:i + 2]) for i in range(0, 8, 2)
        ])
        self.set_all_trainable(not freeze_backbone)

    @property
    def last_stage(self):
        return self.stages[-1]

    def forward(self, image):
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image)
        return outputs


class DINOv2SmallFeatures(FrozenFeatureEncoder):
    channels = (384, 384, 384, 384)

    def __init__(self, pretrained=True, freeze_backbone=True, weights_path=None):
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("DINOv2 requires timm: pip install -r requirements.txt") from error
        self.net = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=pretrained and not weights_path,
            dynamic_img_size=True,
        )
        if weights_path:
            load_backbone_weights(self.net, weights_path)
        self.set_all_trainable(not freeze_backbone)

    @property
    def last_stage(self):
        return self.net.blocks[-1]

    def forward(self, image):
        if image.shape[-2] % 14 or image.shape[-1] % 14:
            raise ValueError("DINOv2 input height and width must be multiples of 14 (e.g. 224)")
        # Four semantic depths, not four spatial resolutions; omit CLS/register tokens.
        return list(self.net.get_intermediate_layers(
            image, n=(2, 5, 8, 11), reshape=True, norm=True,
        ))
