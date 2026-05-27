"""
DEIMv2-JDE: DINOv3STAs + HybridEncoder (DEIMv2 pretrained) + CenterNet JDE heads.

Adapter model that:
  - Uses DINOv3STAs backbone + HybridEncoder from DEIMv2 as pretrained feature extractor
  - Adds CenterNet JDE heads (hm, wh, reg, id) on top
  - Output format is IDENTICAL to DLA-34+CenterNet in AMOT → zero changes to
    decode.py / mot.py (trainer) / multitracker.py (tracker)

Architecture:
    Input (B,3,H,W)
     ↓
    DINOv3STAs (ViT-Tiny + STA)      pretrained DEIMv2-S COCO
     ↓  c2@s8, c3@s16, c4@s32        all 192-ch
    HybridEncoder (FPN + PAN)         pretrained DEIMv2-S COCO
     ↓  f0 @ stride-8  [B, 192, H/8, W/8]
    UpsampleNeck  (bilinear 2× + Conv1×1) trained on VisDrone
     ↓  [B, 192, H/4, W/4]            matches AMOT down_ratio=4
    CenterNet JDE Heads (InvertedBottleneckHead) trained on VisDrone
     hm  → [B, num_classes, H/4, W/4]
     wh  → [B, 2, H/4, W/4]
     reg → [B, 2, H/4, W/4]
     id  → [B, reid_dim, H/4, W/4]
     ↓
    [{'hm', 'wh', 'reg', 'id'}]       same as DLA-34 output (list of 1 dict)

Usage:
    model = get_deimv2_jde(
        heads={'hm': num_classes, 'wh': 2, 'reg': 2, 'id': 512},
        head_bottleneck=64,                                       # InvertedBottleneckHead
        deimv2_pretrained='/path/to/deimv2_dinov3_s_coco.pth',  # DEIMv2 COCO ckpt
    )
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import from local deimv2_engine (standalone copy, no tensorboard dep)
# ---------------------------------------------------------------------------
try:
    from .deimv2_engine import DINOv3STAs, HybridEncoder
    _DEIMV2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"deimv2_engine not importable: {e}")
    _DEIMV2_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers (mirrors DLASeg head init)
# ---------------------------------------------------------------------------
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Upsample Neck: stride-8 → stride-4  (to keep AMOT's down_ratio = 4)
# ---------------------------------------------------------------------------
class UpsampleNeck(nn.Module):
    """
    Bilinear 2× upsample with a lightweight Conv1×1 projection.

    The projection runs at the LOW-resolution input (stride-8, 80×80 for a
    640×640 image) BEFORE upsampling, keeping its cost ~4× cheaper than
    projecting after upsample.

    GFLOPs (640×640):
        Conv1×1(192→192) at 80×80  ≈ 0.47 GFLOPs
        Bilinear 2×                ≈ 0    GFLOPs
        Total ≈ 0.47 GFLOPs  (vs ~3 GFLOPs for the old ConvTranspose2d k=4)
    """
    def __init__(self, in_channels: int = 192, out_channels: int = 192):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Project at low resolution (cheap)
        x = self.proj(x)
        # 2. Bilinear upsample ×2: stride-8 → stride-4
        return F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)


# ---------------------------------------------------------------------------
# Inverted Bottleneck Head (Cách 3)
# ---------------------------------------------------------------------------
class InvertedBottleneckHead(nn.Module):
    """
    Lightweight CenterNet detection head using a squeeze → DW3×3 → project
    bottleneck structure (inspired by MobileNetV2 separable conv).

    Structure:
        Conv1×1(feat_ch → bottleneck_ch)   ← squeeze  (1×1, cheap)
        BN + ReLU6
        DWConv3×3(bottleneck_ch)           ← spatial mixing (depthwise, cheap)
        BN + ReLU6
        Conv1×1(bottleneck_ch → out_ch)    ← project to output (no bias in BN layers)

    GFLOPs comparison at stride-4 (160×160, feat_ch=192, bottleneck_ch=64):
        Standard Conv3×3(192→256): ≈ 22.6 GFLOPs
        This head:
            Conv1×1(192→64)    ≈  0.63 GFLOPs
            DWConv3×3(64)      ≈  0.24 GFLOPs
            Conv1×1(64→out)    ≈  small
            Total              ≈  0.9 GFLOPs  (~25× cheaper)

    Args:
        feat_ch:        input channels from the neck feature map
        out_ch:         output channels (num_classes, 2, reid_dim, …)
        bottleneck_ch:  internal channels for the squeeze/DW stage (default 64)
        hm_bias:        if not None, fill final conv bias with this value
                        (use -2.19 for heatmap heads per CenterNet convention)
    """

    def __init__(
        self,
        feat_ch: int,
        out_ch: int,
        bottleneck_ch: int = 64,
        hm_bias: float = None,
    ):
        super().__init__()
        self.seq = nn.Sequential(
            # --- squeeze ---
            nn.Conv2d(feat_ch, bottleneck_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU6(inplace=True),
            # --- spatial mixing (depthwise) ---
            nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3, padding=1,
                      groups=bottleneck_ch, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU6(inplace=True),
            # --- project to output ---
            nn.Conv2d(bottleneck_ch, out_ch, kernel_size=1, bias=True),
        )
        self._init_weights(hm_bias)

    def _init_weights(self, hm_bias):
        for m in self.seq.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # CenterNet heatmap bias: log(0.1 / 0.9) ≈ -2.19
        if hm_bias is not None:
            self.seq[-1].bias.data.fill_(hm_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


# ---------------------------------------------------------------------------
# DEIMv2JDE – the full wrapper model
# ---------------------------------------------------------------------------
class DEIMv2JDE(nn.Module):
    """
    DEIMv2 (DINOv3STAs + HybridEncoder) + CenterNet JDE heads.

    The model output is a *list* containing one dict, identical to DLA-34:
        [{'hm': ..., 'wh': ..., 'reg': ..., 'id': ...}]
    So the AMOT trainer (mot.py) and tracker (multitracker.py) need zero changes.
    """

    def __init__(
        self,
        heads: dict,
        head_conv: int = 256,               # kept for API compat; ignored when head_bottleneck > 0
        head_bottleneck: int = 64,          # bottleneck channels inside InvertedBottleneckHead
        # ---- DINOv3STAs params (mirrors deimv2_dinov3_s_coco.yml) --------
        vit_name: str = 'vit_tiny',
        embed_dim: int = 192,
        interaction_indexes: list = None,
        num_heads: int = 3,
        vit_weights_path: str = None,          # path to vitt_distill.pt
        # ---- HybridEncoder params -----------------------------------------
        enc_in_channels: list = None,          # default [192, 192, 192]
        enc_hidden_dim: int = 192,
        enc_depth_mult: float = 0.67,
        enc_expansion: float = 0.34,
        enc_dim_feedforward: int = 512,
        # ---- Neck params --------------------------------------------------
        neck_out_channels: int = 192,
    ):
        super().__init__()
        assert _DEIMV2_AVAILABLE, (
            "DEIMv2 engine not found. Add DEIMv2 to sys.path or place it at "
            f"{_DEIMV2_ROOT}"
        )

        if interaction_indexes is None:
            interaction_indexes = [3, 7, 11]
        if enc_in_channels is None:
            enc_in_channels = [192, 192, 192]

        # ---- 1. Backbone --------------------------------------------------
        self.backbone = DINOv3STAs(
            name=vit_name,
            embed_dim=embed_dim,
            interaction_indexes=interaction_indexes,
            num_heads=num_heads,
            weights_path=vit_weights_path,    # vit_tiny backbone weights
        )

        # ---- 2. Encoder ---------------------------------------------------
        self.encoder = HybridEncoder(
            in_channels=enc_in_channels,
            hidden_dim=enc_hidden_dim,
            depth_mult=enc_depth_mult,
            expansion=enc_expansion,
            dim_feedforward=enc_dim_feedforward,
            # keep defaults for version/csp_type/fuse_op/etc.
            feat_strides=[8, 16, 32],
            use_encoder_idx=[2],
            num_encoder_layers=1,
            nhead=8,
            version='deim',
            csp_type='csp2',
            fuse_op='sum',
        )
        # encoder.out_channels == [enc_hidden_dim] * 3

        # ---- 3. Upsample neck (stride-8 → stride-4) ----------------------
        self.neck = UpsampleNeck(enc_hidden_dim, neck_out_channels)

        # ---- 4. CenterNet JDE heads (Inverted Bottleneck) ----------------
        # Using InvertedBottleneckHead for all heads when head_bottleneck > 0.
        # Falls back to a plain Conv1×1 when head_bottleneck == 0 (ablation).
        feat_ch = neck_out_channels
        self.heads = heads      # keep as attribute for reference
        for head_name, out_ch in heads.items():
            hm_bias = -2.19 if 'hm' in head_name else None   # CenterNet standard init

            if head_bottleneck > 0:
                # Inverted-bottleneck head: ~25× cheaper than standard Conv3×3
                head_layer = InvertedBottleneckHead(
                    feat_ch=feat_ch,
                    out_ch=out_ch,
                    bottleneck_ch=head_bottleneck,
                    hm_bias=hm_bias,
                )
            else:
                # Minimal fallback: single Conv1×1 (e.g. for ablation)
                head_layer = nn.Conv2d(feat_ch, out_ch, kernel_size=1, bias=True)
                if hm_bias is not None:
                    head_layer.bias.data.fill_(hm_bias)
                else:
                    fill_fc_weights(head_layer)

            self.__setattr__(head_name, head_layer)

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Returns [output_dict] to match AMOT's list-based API:
            output = model(img)[-1]   → {'hm', 'wh', 'reg', 'id'}
        """
        # 1. Backbone: x → (c2@s8, c3@s16, c4@s32)
        c2, c3, c4 = self.backbone(x)

        # 2. Encoder: multi-scale feature fusion
        #    encoder_outs[0] = finest feature @ stride-8  [B, 192, H/8, W/8]
        encoder_outs = self.encoder([c2, c3, c4])
        f0 = encoder_outs[0]   # finest scale

        # 3. Upsample: stride-8 → stride-4  [B, 192, H/4, W/4]
        feat = self.neck(f0)

        # 4. CenterNet heads
        output = {}
        for head_name in self.heads:
            output[head_name] = self.__getattr__(head_name)(feat)

        return [output]     # list of 1 dict, matching DLASeg.forward() return value


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def load_deimv2_pretrained(model: DEIMv2JDE, ckpt_path: str, strict_encoder: bool = False):
    """
    Load backbone + encoder weights from a DEIMv2 COCO checkpoint.
    CenterNet heads (hm/wh/reg/id) and UpsampleNeck are intentionally left
    randomly initialized for domain-specific fine-tuning on VisDrone.

    Args:
        model:          DEIMv2JDE instance (already constructed)
        ckpt_path:      path to DEIMv2 COCO checkpoint (.pth)
        strict_encoder: if True, raise on missing encoder keys (default False)

    Returns:
        model with backbone+encoder weights loaded
    """
    if not os.path.isfile(ckpt_path):
        logger.warning(f"DEIMv2 checkpoint not found at {ckpt_path}. "
                       f"Training backbone+encoder from scratch.")
        return model

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # DEIMv2 checkpoints store EMA weights under 'ema', fallback to 'model'
    state_dict = ckpt.get('ema', ckpt.get('model', ckpt))
    # strip leading 'module.' (DataParallel)
    state_dict = {
        k[7:] if k.startswith('module.') else k: v
        for k, v in state_dict.items()
    }

    # We only want backbone.* and encoder.* keys
    pretrained_keys = {k: v for k, v in state_dict.items()
                       if k.startswith('backbone.') or k.startswith('encoder.')}

    missing, unexpected = model.load_state_dict(pretrained_keys, strict=False)

    backbone_loaded = sum(1 for k in pretrained_keys if k.startswith('backbone.'))
    encoder_loaded  = sum(1 for k in pretrained_keys if k.startswith('encoder.'))

    logger.info(
        f"DEIMv2 pretrained loaded: backbone={backbone_loaded} params, "
        f"encoder={encoder_loaded} params"
    )
    neck_and_head_missing = [k for k in missing
                             if not (k.startswith('backbone.') or k.startswith('encoder.'))]
    logger.info(
        f"Randomly initialized (to be trained on VisDrone): "
        f"neck + {len(neck_and_head_missing)} head params"
    )
    return model


def freeze_pretrained_params(model: DEIMv2JDE):
    """
    Freeze backbone + encoder. Only neck + heads are trainable.
    Call during warm-up phase (first ~10 epochs on VisDrone).
    """
    for name, param in model.named_parameters():
        if name.startswith('backbone.') or name.startswith('encoder.'):
            param.requires_grad = False
    logger.info("Backbone + encoder frozen. Only neck+heads will be trained.")


def unfreeze_all_params(model: DEIMv2JDE):
    """
    Unfreeze everything for end-to-end fine-tuning.
    Use a very small lr for backbone/encoder relative to heads.
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All parameters unfrozen for end-to-end fine-tuning.")


# ---------------------------------------------------------------------------
# Factory function (matches DLA-34 pattern: get_pose_net → get_deimv2_jde)
# ---------------------------------------------------------------------------

def get_deimv2_jde(
    heads: dict,
    head_conv: int = 256,            # legacy param, kept for API compat
    head_bottleneck: int = 64,       # bottleneck channels in InvertedBottleneckHead
    deimv2_pretrained: str = None,   # path to DEIMv2 COCO .pth checkpoint
    vit_weights_path: str = None,    # path to vitt_distill.pt (ViT-Tiny backbone)
    freeze_backbone: bool = False,   # False = end-to-end fine-tuning (recommended)
):
    """
    Build DEIMv2-JDE model and optionally load DEIMv2 COCO pretrained weights.

    Args:
        heads:               dict mapping head name → output channels
                             e.g. {'hm': 5, 'wh': 2, 'reg': 2, 'id': 512}
        head_conv:           legacy parameter, no longer used (kept for API compat)
        head_bottleneck:     bottleneck channels in the InvertedBottleneckHead.
                             Default 64 gives ~25× cheaper heads vs standard
                             Conv3×3(192→256). Set to 0 for plain Conv1×1 ablation.
        deimv2_pretrained:   path to the full DEIMv2-S COCO checkpoint
                             (backbone+encoder keys will be extracted)
        vit_weights_path:    path to vitt_distill.pt (ViT-Tiny distilled weights)
                             used only when deimv2_pretrained is None
        freeze_backbone:     if True, freeze backbone+encoder after loading
                             so only neck+heads are trained initially

    Returns:
        DEIMv2JDE model
    """
    model = DEIMv2JDE(
        heads=heads,
        head_conv=head_conv,
        head_bottleneck=head_bottleneck,
        vit_weights_path=vit_weights_path,
    )

    if deimv2_pretrained is not None:
        model = load_deimv2_pretrained(model, deimv2_pretrained)
        if freeze_backbone:
            freeze_pretrained_params(model)

    return model
