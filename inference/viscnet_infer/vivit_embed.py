try:
    import timm
except Exception:  # optional; only used by the (disabled) pattern branch
    timm = None
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vivit.configuration_vivit import VivitConfig
from .vivit.modeling_vivit import VivitModel

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class VivitEmbed(nn.Module):
    def __init__(
        self,
        dropout,
        output_size,
        class_bool,
        visc_class,
        gmm_num,
        rpm_bool,
        pat_bool,
        num_frames=50,
        image_size=224,
        pat_mode="legacy",
        pattern_gate_init=0.01,
        pattern_backbone="resnet18",
        pattern_backbone_pretrained=True,
        pattern_backbone_norm="none",
        hidden_size=256,
        num_hidden_layers=10,
        num_attention_heads=8,
        intermediate_size=1024,
    ):
        super(VivitEmbed, self).__init__()
        if pat_mode not in {"legacy", "embedding", "late_concat", "late_residual"}:
            raise ValueError(f"Unsupported pat_mode: {pat_mode}")
        early_pattern_bool = pat_bool and pat_mode in {"legacy", "embedding"}
        self.pattern_backbone_name = str(pattern_backbone)
        self.pattern_backbone_pretrained = bool(pattern_backbone_pretrained)
        self.pattern_backbone_norm = str(pattern_backbone_norm).lower()

        ##### for pretrained model
        # self.config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        # self.featureextractor = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=self.config)

        # Encoder dims are configurable (default = the 9.2M compact ViViT: hidden 256,
        # depth 10, 8 heads, MLP 1024). Smaller variants (e.g. the 7.85M MLP-768 build)
        # override these via the model config block; defaults preserve historical behaviour.
        self.config = VivitConfig(
            hidden_size=int(hidden_size),  # ViViT-L
            num_hidden_layers=int(num_hidden_layers),  # 20
            num_attention_heads=int(num_attention_heads),
            intermediate_size=int(intermediate_size),  # 1024
            tubelet_size=(2, 16, 16),
            image_size=int(image_size),
            num_frames=int(num_frames),
            num_channels=3,
            hidden_dropout_prob=float(dropout),
            attention_probs_dropout_prob=float(dropout),
            use_mean_pooling=False,
            rpm_bool=rpm_bool,
            pat_bool=early_pattern_bool,
            pat_mode=pat_mode,
            pattern_backbone=self.pattern_backbone_name,
            pattern_backbone_pretrained=self.pattern_backbone_pretrained,
            pattern_backbone_norm=self.pattern_backbone_norm,
        )

        self.featureextractor = VivitModel(self.config)
        self.hidden_size = self.config.hidden_size
        self.pat_bool = pat_bool
        self.pat_mode = pat_mode
        self.pattern_gate = (
            nn.Parameter(torch.tensor(float(pattern_gate_init)))
            if self.pat_bool and self.pat_mode in {"late_concat", "late_residual"}
            else None
        )

        if self.pat_bool and self.pat_mode in {"legacy", "late_concat", "late_residual"}:
            self.pat_backbone = timm.create_model(
                self.pattern_backbone_name,
                pretrained=self.pattern_backbone_pretrained,
                num_classes=0,
            )
            pat_feature_dim = int(getattr(self.pat_backbone, "num_features", 512))
            self.pat_proj = nn.Linear(pat_feature_dim, self.hidden_size)

            for p in self.pat_backbone.parameters():
                p.requires_grad = False
            self.pat_backbone.eval()
        else:
            self.pat_backbone = None
            self.pat_proj = None

        # self.pat_embed = nn.Sequential(
        #     nn.Conv2d(3, 256, kernel_size=16, stride=16),  # (B,256,Hp,Wp)
        #     nn.Flatten(2),                                                 # (B,256,N)
        # )

        # FC HEAD
        fc_input_size = self.hidden_size * 2 if self.pat_bool and self.pat_mode == "late_concat" else self.hidden_size
        self.fc_dropout = nn.Dropout(p=float(dropout))
        if class_bool:
            self.fc = nn.Sequential(nn.Linear(fc_input_size, 192), nn.SiLU(), nn.Linear(192, visc_class))
        else:
            # Regression head width follows output_size so the same encoder serves
            # 3-target dimensionless regression (Re/Ca/Fr) and 1-target cP regression.
            # Defaults to 3 to preserve historical behaviour when output_size is unset/<=0.
            regression_outputs = int(output_size) if int(output_size) and int(output_size) > 0 else 3
            self.fc = nn.Sequential(nn.Linear(fc_input_size, 192), nn.SiLU(), nn.Linear(192, regression_outputs))

    def _pattern_features(self, pattern):
        pattern = pattern.permute(0, 3, 1, 2).contiguous()
        _, _, height, width = pattern.shape
        top = max(0, (height - 224) // 2)
        left = max(0, (width - 224) // 2)
        pattern = pattern[:, :, top : top + min(height, 224), left : left + min(width, 224)]
        if pattern.shape[-2:] != (224, 224):
            pattern = F.interpolate(pattern, size=(224, 224), mode="bilinear", align_corners=False)
        if self.pattern_backbone_norm in {"imagenet", "timm_imagenet"}:
            pattern = ((pattern + 1.0) * 0.5).clamp(0.0, 1.0)
            mean = pattern.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
            std = pattern.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
            pattern = (pattern - mean) / std
        elif self.pattern_backbone_norm not in {"none", "", "identity"}:
            raise ValueError(f"Unsupported pattern_backbone_norm: {self.pattern_backbone_norm}")
        self.pat_backbone.eval()
        with torch.no_grad():
            features = self.pat_backbone(pattern)
        return self.pat_proj(features)

    def forward(self, video, rpm_idx, pattern):
        outputs = self.featureextractor(video, rpm_idx, pattern)
        video_features = outputs.last_hidden_state.mean(dim=1).contiguous()

        if self.pat_bool and self.pat_mode in {"legacy", "late_concat", "late_residual"}:
            pat_features = self._pattern_features(pattern)
            if self.pat_mode == "legacy":
                video_features = video_features - pat_features
            elif self.pat_mode == "late_concat":
                video_features = torch.cat([video_features, self.pattern_gate * pat_features], dim=1)
            elif self.pat_mode == "late_residual":
                video_features = video_features - self.pattern_gate * pat_features

        viscosity = self.fc(self.fc_dropout(video_features))

        return viscosity
