"""DR grading model from `grading model notebook.ipynb`.

The notebook used ConvNeXt Base from timm, modified so the first stem
convolution accepts 9 channels:

RGB + vessel + MA + HE + EX + OD + CW
"""

import torch
from torch import nn

GRADE_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
GRADER_MASK_CHANNELS = ["vessel", "MA", "HE", "EX", "OD", "CW"]


def build_convnext_base_9ch(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for the grading model. Install with: pip install -e \".[dev,ml]\""
        ) from exc

    model = timm.create_model("convnext_base", pretrained=pretrained, num_classes=num_classes)
    old_stem = model.stem[0]

    new_stem_conv = nn.Conv2d(
        in_channels=9,
        out_channels=old_stem.out_channels,
        kernel_size=old_stem.kernel_size,
        stride=old_stem.stride,
        padding=old_stem.padding,
        bias=old_stem.bias is not None,
    )

    with torch.no_grad():
        new_stem_conv.weight[:, :3, :, :] = old_stem.weight
        new_stem_conv.weight[:, 3:, :, :] = 0.0
        if old_stem.bias is not None and new_stem_conv.bias is not None:
            new_stem_conv.bias.copy_(old_stem.bias)

    model.stem[0] = new_stem_conv
    return model


class GraderCNN(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = False):
        super().__init__()
        self.model = build_convnext_base_9ch(num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)


def build_grader_model(num_classes: int = 5, pretrained: bool = False) -> nn.Module:
    return build_convnext_base_9ch(num_classes=num_classes, pretrained=pretrained)
