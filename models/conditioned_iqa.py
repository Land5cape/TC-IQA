import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.cross_layer_fusion import CrossLayerLocalFusion
from models.alternative_backbones import (
    ConvNeXtTinyPyramid, DINOv2SmallFeatures, load_backbone_weights,
)


@dataclass(frozen=True)
class ConditionSpec:
    datasets: dict
    tasks: dict
    metrics: dict

    @classmethod
    def default(cls):
        return cls(
            {"mpd": 0, "miqd": 1},
            {"det": 0, "seg": 1, "ret": 2, "cls": 3, "ins": 4},
            {"utility": 0, "accuracy": 1, "consistency": 2, "composite": 3},
        )

    def encode(self, dataset, task, metric):
        return self.datasets[dataset], self.tasks[task], self.metrics[metric]


class ResNet50Pyramid(nn.Module):
    channels = (256, 512, 1024, 2048)

    def __init__(self, pretrained=True, freeze_backbone=True, weights_path=None):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained and not weights_path else None
        net = models.resnet50(weights=weights)
        if weights_path:
            load_backbone_weights(net, weights_path)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.set_all_trainable(not freeze_backbone)

    @property
    def last_stage(self):
        return self.stages[-1]

    def set_all_trainable(self, trainable):
        for parameter in self.parameters():
            parameter.requires_grad = trainable

    def set_last_stage_trainable(self, trainable):
        self.set_all_trainable(False)
        for parameter in self.last_stage.parameters():
            parameter.requires_grad = trainable

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, image):
        image = self.stem(image)
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image)
        return outputs


class VGG16Pyramid(nn.Module):
    channels = (128, 256, 512, 512)

    def __init__(self, pretrained=True, freeze_backbone=True, weights_path=None):
        super().__init__()
        weights = models.VGG16_Weights.DEFAULT if pretrained and not weights_path else None
        net = models.vgg16(weights=weights)
        if weights_path:
            load_backbone_weights(net, weights_path)
        layers = list(net.features.children())
        self.stem = nn.Identity()
        self.stages = nn.ModuleList(
            [nn.Sequential(*layers[:9]), nn.Sequential(*layers[9:16]), nn.Sequential(*layers[16:23]), nn.Sequential(*layers[23:])]
        )
        self.set_all_trainable(not freeze_backbone)

    @property
    def last_stage(self):
        return self.stages[-1]

    def set_all_trainable(self, trainable):
        for parameter in self.parameters():
            parameter.requires_grad = trainable

    def set_last_stage_trainable(self, trainable):
        self.set_all_trainable(False)
        for parameter in self.last_stage.parameters():
            parameter.requires_grad = trainable

    def forward(self, image):
        outputs = []
        for stage in self.stages:
            image = stage(image)
            outputs.append(image)
        return outputs


def failure_probability(mean, sigma, threshold):
    z = (threshold.to(mean.device) - mean) / sigma.clamp_min(1e-6)
    return 0.5 * torch.erfc(z / math.sqrt(2.0))


class SemanticBottleneckProjection(nn.Module):
    """Keep comparison features dense and factor only the semantic weights."""

    def __init__(self, channels, scale_dim, bottleneck_dim):
        super().__init__()
        self.comparison_dim = 2 * channels + 1
        self.comparison = nn.Linear(self.comparison_dim, scale_dim)
        self.semantic_down = nn.Linear(2 * channels, bottleneck_dim, bias=False)
        self.semantic_up = nn.Linear(bottleneck_dim, scale_dim, bias=False)
        self.norm = nn.LayerNorm(scale_dim)
        self.activation = nn.GELU()

    def forward(self, features):
        comparison = self.comparison(features[..., :self.comparison_dim])
        semantic = self.semantic_up(self.semantic_down(features[..., self.comparison_dim:]))
        return self.activation(self.norm(comparison + semantic))


class ReferenceDistortionInteraction(nn.Module):
    """Compare aligned pair tokens before condition-specific spatial pooling."""

    def __init__(self, channels, interaction_dim, scale_dim, condition_dim, grid_size):
        super().__init__()
        self.grid_size = int(grid_size)
        self.token_projection = nn.Sequential(
            nn.Linear(channels, interaction_dim),
            nn.LayerNorm(interaction_dim),
            nn.GELU(),
        )
        self.query = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.key = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.value = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.comparison = nn.Sequential(
            nn.Linear(3 * interaction_dim, scale_dim),
            nn.LayerNorm(scale_dim),
            nn.GELU(),
        )
        self.condition_attention = nn.Sequential(
            nn.Linear(scale_dim + condition_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.output = nn.Linear(scale_dim, scale_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        token_count = self.grid_size * self.grid_size
        self.register_buffer(
            "position_bias", 2.0 * torch.eye(token_count), persistent=False
        )

    def encode(self, ref_feature, dist_feature):
        output_size = (self.grid_size, self.grid_size)
        if ref_feature is None:
            dist = self.token_projection(F.adaptive_avg_pool2d(
                dist_feature, output_size).flatten(2).transpose(1, 2))
            logits = self.query(dist) @ self.key(dist).transpose(-1, -2)
            logits = logits / math.sqrt(self.query.out_features)
            logits = logits + self.position_bias.to(dtype=logits.dtype)[None]
            context = torch.softmax(logits, dim=-1) @ self.value(dist)
            return self.comparison(torch.cat([dist, context, dist * context], dim=-1))
        ref = F.adaptive_avg_pool2d(ref_feature, output_size).flatten(2).transpose(1, 2)
        dist = F.adaptive_avg_pool2d(dist_feature, output_size).flatten(2).transpose(1, 2)
        ref = self.token_projection(ref)
        dist = self.token_projection(dist)
        logits = torch.matmul(self.query(dist), self.key(ref).transpose(-1, -2))
        logits = logits / math.sqrt(self.query.out_features)
        logits = logits + self.position_bias.to(dtype=logits.dtype)[None]
        matched_ref = torch.matmul(torch.softmax(logits, dim=-1), self.value(ref))
        comparison = torch.cat(
            [(dist - ref).abs(), dist * ref, (dist - matched_ref).abs()], dim=-1
        )
        return self.comparison(comparison)

    def pool(self, tokens, condition):
        expanded_tokens = tokens[:, None].expand(-1, condition.shape[1], -1, -1)
        expanded_condition = condition[:, :, None].expand(
            -1, -1, tokens.shape[1], -1
        )
        attention = torch.softmax(
            self.condition_attention(
                torch.cat([expanded_tokens, expanded_condition], dim=-1)
            ).squeeze(-1),
            dim=-1,
        )
        pooled = (attention[..., None] * expanded_tokens).sum(-2)
        return self.output(pooled), attention


class ConditionedFRIQA(nn.Module):
    def __init__(
        self,
        pretrained=True,
        freeze_backbone=True,
        spec=None,
        backbone="resnet50",
        scale_dim=128,
        hidden_dim=512,
        adapter_dim=64,
        task_adapter_dim=16,
        condition_dim=32,
        sigma_min=0.03,
        pair_features="symmetric",
        semantic_bottleneck_dim=None,
        local_grid_size=None,
        spatial_task_adapter_dim=None,
        spatial_interaction_dim=None,
        spatial_interaction_grid=4,
        backbone_weights=None,
        reference_mode="fr",
        scale_fusion="attention_sum",
        cross_layer_dim=None,
        cross_layer_grid=28,
        cross_layer_heads=1,
        cross_layer_fusion="add",
    ):
        super().__init__()
        if reference_mode not in ("fr", "nr"):
            raise ValueError("reference_mode must be fr or nr")
        if reference_mode == "nr" and (local_grid_size is not None or spatial_task_adapter_dim is not None):
            raise ValueError("NR requires local_grid_size and spatial_task_adapter_dim to be null")
        self.reference_mode = reference_mode
        if scale_fusion not in ("attention_sum", "attention_concat"):
            raise ValueError("scale_fusion must be attention_sum or attention_concat")
        self.scale_fusion = scale_fusion
        if cross_layer_fusion not in ("add", "gated"):
            raise ValueError("cross_layer_fusion must be add or gated")
        if cross_layer_fusion == "gated" and cross_layer_dim is None:
            raise ValueError("cross_layer_fusion=gated requires cross_layer_dim")
        self.cross_layer_fusion = cross_layer_fusion
        if isinstance(cross_layer_heads, bool) or not isinstance(cross_layer_heads, int) or not 1 <= cross_layer_heads <= 8:
            raise ValueError("cross_layer_heads must be an integer in [1,8]")
        self.cross_layer_heads = cross_layer_heads
        if cross_layer_dim is not None:
            if isinstance(cross_layer_dim, bool) or not isinstance(cross_layer_dim, int) or cross_layer_dim < 1:
                raise ValueError("cross_layer_dim must be a positive integer or null")
            if isinstance(cross_layer_grid, bool) or not isinstance(cross_layer_grid, int) or not 2 <= cross_layer_grid <= 56:
                raise ValueError("cross_layer_grid must be an integer in [2,56]")
            if reference_mode != "nr" or scale_fusion != "attention_sum":
                raise ValueError("cross_layer requires NR and attention_sum")
            if any(v is not None for v in (local_grid_size, spatial_task_adapter_dim, spatial_interaction_dim)):
                raise ValueError("cross_layer cannot be combined with legacy spatial branches")
        self.cross_layer_dim = cross_layer_dim
        self.cross_layer_grid = cross_layer_grid
        if pair_features not in ("symmetric", "ordered_semantic"):
            raise ValueError(f"Unsupported pair_features: {pair_features}")
        if semantic_bottleneck_dim is not None:
            if (isinstance(semantic_bottleneck_dim, bool)
                    or not isinstance(semantic_bottleneck_dim, int)
                    or not 1 <= semantic_bottleneck_dim <= scale_dim):
                raise ValueError("semantic_bottleneck_dim must be null or an integer in [1, scale_dim]")
            if pair_features != "ordered_semantic":
                raise ValueError("semantic_bottleneck_dim requires pair_features='ordered_semantic'")
        if local_grid_size is not None:
            if (isinstance(local_grid_size, bool)
                    or not isinstance(local_grid_size, int)
                    or not 2 <= local_grid_size <= 4):
                raise ValueError("local_grid_size must be null or an integer in [2, 4]")
        if spatial_task_adapter_dim is not None:
            if (isinstance(spatial_task_adapter_dim, bool)
                    or not isinstance(spatial_task_adapter_dim, int)
                    or spatial_task_adapter_dim < 1):
                raise ValueError("spatial_task_adapter_dim must be null or a positive integer")
        if (isinstance(spatial_interaction_grid, bool)
                or not isinstance(spatial_interaction_grid, int)
                or not 2 <= spatial_interaction_grid <= 7):
            raise ValueError("spatial_interaction_grid must be an integer in [2, 7]")
        if spatial_interaction_dim is not None:
            if (isinstance(spatial_interaction_dim, bool)
                    or not isinstance(spatial_interaction_dim, int)
                    or spatial_interaction_dim < 1):
                raise ValueError("spatial_interaction_dim must be null or a positive integer")
        if spatial_task_adapter_dim is not None and spatial_interaction_dim is not None:
            raise ValueError(
                "spatial_task_adapter_dim and spatial_interaction_dim cannot be enabled together"
            )
        # Missing options in older checkpoint configurations retain their exact
        # projection shapes. Server configs explicitly enable ordered semantics.
        self.pair_features = pair_features
        self.semantic_bottleneck_dim = semantic_bottleneck_dim
        self.local_grid_size = local_grid_size
        self.spatial_task_adapter_dim = spatial_task_adapter_dim
        self.spatial_interaction_dim = spatial_interaction_dim
        self.spatial_interaction_grid = spatial_interaction_grid
        self.spec = spec or ConditionSpec.default()
        self.sigma_min = float(sigma_min)
        if backbone == "resnet50":
            self.encoder = ResNet50Pyramid(pretrained, freeze_backbone, backbone_weights)
        elif backbone == "vgg16":
            self.encoder = VGG16Pyramid(pretrained, freeze_backbone, backbone_weights)
        elif backbone == "convnext_tiny":
            self.encoder = ConvNeXtTinyPyramid(pretrained, freeze_backbone, backbone_weights)
        elif backbone == "dinov2_vits14":
            self.encoder = DINOv2SmallFeatures(pretrained, freeze_backbone, backbone_weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.dataset_embedding = nn.Embedding(len(self.spec.datasets), condition_dim)
        self.task_embedding = nn.Embedding(len(self.spec.tasks), condition_dim)
        self.metric_embedding = nn.Embedding(len(self.spec.metrics), condition_dim)
        feature_multiplier = 4 if pair_features == "ordered_semantic" else 2
        if reference_mode == "nr":
            # Project each spatial position before pooling; no fabricated pair differences.
            self.scale_projections = nn.ModuleList([
                nn.Sequential(nn.Conv2d(c, scale_dim, 1), nn.GELU(),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.LayerNorm(scale_dim))
                for c in self.encoder.channels
            ])
        elif semantic_bottleneck_dim is None:
            # Preserve legacy keys/shapes and initialization for old configs.
            self.scale_projections = nn.ModuleList(
                [nn.Sequential(nn.Linear(feature_multiplier * channels + 1, scale_dim), nn.LayerNorm(scale_dim), nn.GELU()) for channels in self.encoder.channels]
            )
        else:
            self.scale_projections = nn.ModuleList(
                [SemanticBottleneckProjection(channels, scale_dim, semantic_bottleneck_dim)
                 for channels in self.encoder.channels]
            )
        if spatial_task_adapter_dim is not None:
            spatial_channels = self.encoder.channels[-2:]
            self.spatial_task_down = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(channels, spatial_task_adapter_dim, kernel_size=1)
                    for _ in range(len(self.spec.tasks))
                ])
                for channels in spatial_channels
            ])
            self.spatial_task_up = nn.ModuleList([
                nn.ModuleList([
                    nn.Conv2d(spatial_task_adapter_dim, channels, kernel_size=1)
                    for _ in range(len(self.spec.tasks))
                ])
                for channels in spatial_channels
            ])
            for level in self.spatial_task_up:
                for layer in level:
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
        if local_grid_size is not None:
            local_channels = self.encoder.channels[-1]
            local_dim = min(32, scale_dim)
            self.local_projection = nn.Sequential(
                nn.Linear(2 * local_channels + 1, local_dim),
                nn.LayerNorm(local_dim),
                nn.GELU(),
                nn.Linear(local_dim, scale_dim),
                nn.LayerNorm(scale_dim),
                nn.GELU(),
            )
            self.local_attention = nn.Sequential(
                nn.Linear(scale_dim + condition_dim, 64), nn.GELU(), nn.Linear(64, 1)
            )
            self.local_output = nn.Linear(scale_dim, scale_dim)
            nn.init.zeros_(self.local_output.weight)
            nn.init.zeros_(self.local_output.bias)
        self.scale_attention = nn.Sequential(
            nn.Linear(scale_dim + condition_dim, 64), nn.GELU(), nn.Linear(64, 1)
        )
        fusion_dim = scale_dim * len(self.encoder.channels) if scale_fusion == "attention_concat" else scale_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.adapter_down = nn.Linear(hidden_dim, adapter_dim)
        self.adapter_gate = nn.Linear(condition_dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)
        self.task_adapter_down = nn.ModuleList(
            [nn.Linear(hidden_dim, task_adapter_dim) for _ in range(len(self.spec.tasks))]
        )
        self.task_adapter_up = nn.ModuleList(
            [nn.Linear(task_adapter_dim, hidden_dim) for _ in range(len(self.spec.tasks))]
        )
        self.mean_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(len(self.spec.tasks))]
        )
        self.sigma_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(len(self.spec.tasks))]
        )
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        for layer in self.task_adapter_up:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        # Targets are standardized per condition, so initialize sigma near one
        # rather than at the raw-label scale used by the earlier experiments.
        sigma_bias = math.log(math.expm1(max(1.0 - self.sigma_min, 1e-3)))
        for head in self.sigma_heads:
            nn.init.constant_(head.bias, sigma_bias)
        # Construct optional new branches after every legacy module so enabling
        # them does not change legacy initialization under the same random seed.
        if spatial_interaction_dim is not None:
            self.spatial_interactions = nn.ModuleList([
                ReferenceDistortionInteraction(
                    channels,
                    spatial_interaction_dim,
                    scale_dim,
                    condition_dim,
                    spatial_interaction_grid,
                )
                for channels in self.encoder.channels[-2:]
            ])

        if cross_layer_dim is not None:
            # Construct last to preserve common parameter initialization under a fixed seed.
            self.cross_layer = CrossLayerLocalFusion(
                self.encoder.channels, cross_layer_dim, condition_dim, scale_dim, cross_layer_grid,
                heads=cross_layer_heads)
        if cross_layer_fusion == "gated":
            self.global_branch_projection = nn.Linear(scale_dim, scale_dim)
            self.local_branch_projection = nn.Linear(scale_dim, scale_dim)
            for projection in (self.global_branch_projection, self.local_branch_projection):
                nn.init.eye_(projection.weight)
                nn.init.zeros_(projection.bias)
            gate_input_dim = 2 * scale_dim + condition_dim
            self.content_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim), nn.Linear(gate_input_dim, 32),
                nn.GELU(), nn.Linear(32, scale_dim))
            nn.init.zeros_(self.content_gate[-1].weight)
            nn.init.zeros_(self.content_gate[-1].bias)

    def set_last_stage_trainable(self, trainable):
        self.encoder.set_last_stage_trainable(trainable)

    @staticmethod
    def _compare(ref_feature, dist_feature, pair_features="symmetric"):
        difference = (ref_feature - dist_feature).abs().mean((2, 3))
        product = (ref_feature * dist_feature).mean((2, 3))
        ref_normalized = F.normalize(ref_feature, dim=1)
        dist_normalized = F.normalize(dist_feature, dim=1)
        cosine_distance = 1.0 - (ref_normalized * dist_normalized).sum(1).mean((1, 2))
        components = [difference, product, cosine_distance[:, None]]
        if pair_features == "ordered_semantic":
            # Preserve the original comparison statistics and the distinct
            # roles/content of each input without another backbone pass.
            components.extend([ref_feature.mean((2, 3)), dist_feature.mean((2, 3))])
        elif pair_features != "symmetric":
            raise ValueError(f"Unsupported pair_features: {pair_features}")
        return torch.cat(components, dim=1)

    @staticmethod
    def _local_compare(ref_feature, dist_feature, grid_size):
        output_size = (grid_size, grid_size)
        difference = F.adaptive_avg_pool2d((ref_feature - dist_feature).abs(), output_size)
        product = F.adaptive_avg_pool2d(ref_feature * dist_feature, output_size)
        ref_normalized = F.normalize(ref_feature, dim=1)
        dist_normalized = F.normalize(dist_feature, dim=1)
        cosine_map = 1.0 - (ref_normalized * dist_normalized).sum(1, keepdim=True)
        cosine_distance = F.adaptive_avg_pool2d(cosine_map, output_size)
        local = torch.cat([difference, product, cosine_distance], dim=1)
        return local.permute(0, 2, 3, 1).reshape(local.shape[0], grid_size * grid_size, -1)

    def _encode_pair_components(self, ref, dist):
        """Encode a pair once and retain every enabled shared comparison branch."""
        if self.reference_mode == "nr":
            levels = self.encoder(dist)
            scales = torch.stack([p(f) for p, f in zip(self.scale_projections, levels)], dim=1)
            spatial = None
            if self.spatial_interaction_dim is not None:
                spatial = [m.encode(None, f) for m, f in zip(self.spatial_interactions, levels[-2:])]
            return scales, None, spatial
        if ref is None:
            raise ValueError("FR mode requires a reference image")
        ref_levels = self.encoder(ref)
        dist_levels = self.encoder(dist)
        scale_features = torch.stack(
            [projection(self._compare(a, b, self.pair_features))
             for projection, a, b in zip(self.scale_projections, ref_levels, dist_levels)],
            dim=1,
        )
        local_features = None
        if self.local_grid_size is not None:
            local_features = self.local_projection(
                self._local_compare(ref_levels[-1], dist_levels[-1], self.local_grid_size)
            )
        spatial_features = None
        if self.spatial_interaction_dim is not None:
            spatial_features = [
                interaction.encode(ref_level, dist_level)
                for interaction, ref_level, dist_level in zip(
                    self.spatial_interactions, ref_levels[-2:], dist_levels[-2:]
                )
            ]
        return scale_features, local_features, spatial_features

    def encode_pair_with_local(self, ref, dist):
        """Backward-compatible pair encoding with the optional local branch."""

        scale_features, local_features, _ = self._encode_pair_components(ref, dist)
        return scale_features, local_features

    def _adapt_spatial_feature(self, feature, task_id, adapter_index):
        batch, conditions = task_id.shape
        channels, height, width = feature.shape[1:]
        residual = feature.new_zeros(batch * conditions, channels, height, width)
        for task in torch.unique(task_id).tolist():
            positions = torch.nonzero(task_id == task, as_tuple=False)
            rows, inverse = torch.unique(positions[:, 0], sorted=True, return_inverse=True)
            task_residual = self.spatial_task_up[adapter_index][task](
                F.gelu(self.spatial_task_down[adapter_index][task](feature[rows]))
            )
            flat_indices = positions[:, 0] * conditions + positions[:, 1]
            residual = residual.index_copy(0, flat_indices, task_residual[inverse])
        residual = residual.reshape(batch, conditions, channels, height, width)
        return feature[:, None] + residual

    def encode_pair_for_tasks(self, ref, dist, task_id):
        """Apply task adapters before pooling while sharing the frozen encoder."""

        if task_id.ndim == 1:
            task_id = task_id[:, None]
        ref_levels = self.encoder(ref)
        dist_levels = self.encoder(dist)
        projected = []
        adapted_start = len(ref_levels) - 2
        for level_index, (projection, ref_level, dist_level) in enumerate(
                zip(self.scale_projections, ref_levels, dist_levels)):
            if level_index < adapted_start:
                base = projection(self._compare(ref_level, dist_level, self.pair_features))
                projected.append(base[:, None].expand(-1, task_id.shape[1], -1))
                continue
            adapter_index = level_index - adapted_start
            adapted_ref = self._adapt_spatial_feature(ref_level, task_id, adapter_index)
            adapted_dist = self._adapt_spatial_feature(dist_level, task_id, adapter_index)
            batch, conditions, channels, height, width = adapted_ref.shape
            comparison = self._compare(
                adapted_ref.reshape(batch * conditions, channels, height, width),
                adapted_dist.reshape(batch * conditions, channels, height, width),
                self.pair_features,
            )
            projected.append(projection(comparison).reshape(batch, conditions, -1))
        return torch.stack(projected, dim=2)

    def encode_pair(self, ref, dist):
        """Encode an image pair once, independently of task/metric conditions."""

        if self.cross_layer_dim is not None:
            levels = self.encoder(dist)
            scales = torch.stack([p(f) for p, f in zip(self.scale_projections, levels)], dim=1)
            return {"scale_features": scales, "cross_layer_features": self.cross_layer.encode(levels)}
        scale_features, local_features, spatial_features = self._encode_pair_components(ref, dist)
        if spatial_features is not None:
            return {
                "scale_features": scale_features,
                "local_features": local_features,
                "spatial_features": spatial_features,
            }
        return scale_features

    def predict_conditions(
        self,
        scale_features,
        dataset_id,
        task_id,
        metric_id,
        local_features=None,
        spatial_features=None,
        cross_layer_features=None,
    ):
        """Predict one or more conditions from already-computed pair features.

        IDs may have shape ``[batch]`` for the original scalar interface or
        ``[batch, conditions]`` for MIQD's three metrics.  This changes only the
        computation schedule: every condition still passes through the same
        scalar conditioned predictor.
        """

        if isinstance(scale_features, dict):
            if local_features is not None or spatial_features is not None or cross_layer_features is not None:
                raise ValueError(
                    "local_features and spatial_features must not be passed twice"
                )
            encoded = scale_features
            scale_features = encoded["scale_features"]
            local_features = encoded.get("local_features")
            spatial_features = encoded.get("spatial_features")
            cross_layer_features = encoded.get("cross_layer_features")
        if (self.cross_layer_dim is not None) != (cross_layer_features is not None):
            raise ValueError("cross_layer features must match the enabled model; use encode_pair")
        scalar_output = dataset_id.ndim == 1
        if scalar_output:
            dataset_id = dataset_id[:, None]
            task_id = task_id[:, None]
            metric_id = metric_id[:, None]
        if dataset_id.shape != task_id.shape or task_id.shape != metric_id.shape:
            raise ValueError("dataset_id, task_id, and metric_id must have the same shape")
        if dataset_id.ndim != 2 or dataset_id.shape[0] != scale_features.shape[0]:
            raise ValueError("condition IDs must have shape [batch] or [batch, conditions]")
        if scale_features.ndim not in (3, 4):
            raise ValueError("scale_features must have shape [batch, scales, features] or [batch, conditions, scales, features]")
        if scale_features.ndim == 4 and scale_features.shape[1] != dataset_id.shape[1]:
            raise ValueError("conditioned scale features must match the number of conditions")
        if local_features is not None and (
                local_features.ndim != 3 or local_features.shape[0] != scale_features.shape[0]):
            raise ValueError("local_features must have shape [batch, regions, features]")
        if spatial_features is not None and (
                len(spatial_features) != 2
                or any(features.ndim != 3 or features.shape[0] != scale_features.shape[0]
                       for features in spatial_features)):
            raise ValueError("spatial_features must contain two [batch, tokens, features] tensors")

        condition = (
            self.dataset_embedding(dataset_id)
            + self.task_embedding(task_id)
            + self.metric_embedding(metric_id)
        )
        expanded_scales = (
            scale_features[:, None, :, :].expand(-1, condition.shape[1], -1, -1)
            if scale_features.ndim == 3 else scale_features
        )
        spatial_attention = None
        if spatial_features is not None:
            residuals, spatial_weights = [], []
            for interaction, features in zip(self.spatial_interactions, spatial_features):
                residual, weights = interaction.pool(features, condition)
                residuals.append(residual)
                spatial_weights.append(weights)
            spatial_residual = torch.stack(residuals, dim=2)
            prefix = expanded_scales.new_zeros(
                *expanded_scales.shape[:2], expanded_scales.shape[2] - 2,
                expanded_scales.shape[3]
            )
            expanded_scales = expanded_scales + torch.cat(
                [prefix, spatial_residual], dim=2
            )
            spatial_attention = torch.stack(spatial_weights, dim=2)
        expanded_condition = condition[:, :, None, :].expand(
            -1, -1, expanded_scales.shape[2], -1
        )
        attention = torch.softmax(
            self.scale_attention(
                torch.cat([expanded_scales, expanded_condition], dim=-1)
            ).squeeze(-1),
            dim=-1,
        )
        weighted_scales = attention[..., None] * expanded_scales
        if self.scale_fusion == "attention_concat":
            # Preserve scale identity; uniform attention leaves each block unscaled.
            pooled = (weighted_scales * expanded_scales.shape[-2]).flatten(-2)
        else:
            pooled = weighted_scales.sum(-2)
        local_attention = None
        if local_features is not None:
            expanded_local = local_features[:, None, :, :].expand(
                -1, condition.shape[1], -1, -1
            )
            local_condition = condition[:, :, None, :].expand(
                -1, -1, local_features.shape[1], -1
            )
            local_attention = torch.softmax(
                self.local_attention(
                    torch.cat([expanded_local, local_condition], dim=-1)
                ).squeeze(-1),
                dim=-1,
            )
            local_pooled = (local_attention[..., None] * expanded_local).sum(-2)
            local_residual = self.local_output(local_pooled)
            if self.scale_fusion == "attention_concat":
                local_residual = local_residual.repeat(1, 1, expanded_scales.shape[-2])
            pooled = pooled + local_residual
        cross_layer_attention = None
        content_gate = None
        if cross_layer_features is not None:
            task_condition = self.dataset_embedding(dataset_id) + self.task_embedding(task_id)
            residual, cross_layer_attention = self.cross_layer.pool(cross_layer_features, task_condition, condition)
            if self.cross_layer_fusion == "gated":
                global_feature = self.global_branch_projection(pooled)
                local_feature = self.local_branch_projection(residual)
                content_gate = torch.sigmoid(self.content_gate(torch.cat(
                    [global_feature, local_feature, condition], dim=-1)))
                pooled = global_feature + content_gate * local_feature
            else:
                pooled = pooled + residual
        hidden = self.fusion(pooled)
        gate = torch.sigmoid(self.adapter_gate(condition))
        adapted = hidden + self.adapter_up(gate * F.gelu(self.adapter_down(hidden)))
        task_residuals = torch.stack(
            [up(F.gelu(down(adapted))) for down, up in zip(self.task_adapter_down, self.task_adapter_up)],
            dim=2,
        )
        task_index = task_id[..., None, None].expand(-1, -1, 1, adapted.shape[-1])
        task_residual = task_residuals.gather(2, task_index).squeeze(2)
        adapted = adapted + task_residual
        mean_by_task = torch.stack(
            [head(adapted).squeeze(-1) for head in self.mean_heads], dim=-1
        )
        sigma_by_task = torch.stack(
            [head(adapted).squeeze(-1) for head in self.sigma_heads], dim=-1
        )
        head_index = task_id[..., None]
        mean = mean_by_task.gather(-1, head_index).squeeze(-1)
        sigma = self.sigma_min + F.softplus(
            sigma_by_task.gather(-1, head_index).squeeze(-1)
        )
        if scalar_output:
            mean = mean.squeeze(1)
            sigma = sigma.squeeze(1)
            attention = attention.squeeze(1)
            if local_attention is not None:
                local_attention = local_attention.squeeze(1)
            if spatial_attention is not None:
                spatial_attention = spatial_attention.squeeze(1)
            if cross_layer_attention is not None:
                cross_layer_attention = cross_layer_attention.squeeze(1)
            if content_gate is not None:
                content_gate = content_gate.squeeze(1)
        result = {"mean": mean, "sigma": sigma, "scale_attention": attention}
        if local_attention is not None:
            result["local_attention"] = local_attention
        if spatial_attention is not None:
            result["spatial_attention"] = spatial_attention
        if cross_layer_attention is not None:
            result["cross_layer_attention"] = cross_layer_attention
        if content_gate is not None:
            result["content_gate"] = content_gate
        return result

    def forward(self, ref, dist, dataset_id, task_id, metric_id):
        if self.cross_layer_dim is not None:
            return self.predict_conditions(self.encode_pair(ref, dist), dataset_id, task_id, metric_id)
        if self.spatial_task_adapter_dim is not None:
            scale_features = self.encode_pair_for_tasks(ref, dist, task_id)
            return self.predict_conditions(
                scale_features, dataset_id, task_id, metric_id
            )
        scale_features, local_features, spatial_features = self._encode_pair_components(ref, dist)
        return self.predict_conditions(
            scale_features,
            dataset_id,
            task_id,
            metric_id,
            local_features=local_features,
            spatial_features=spatial_features,
        )
