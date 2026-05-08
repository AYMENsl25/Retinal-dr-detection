"""Lesion segmentation model from `lession-unetps-ps.ipynb`.

The checkpoint was trained with segmentation_models_pytorch.Unet:
ResNet34 encoder, RGB input, 5 lesion output channels.
"""

from torch import nn

LESION_TYPES = ["MA", "HE", "EX", "OD", "CW"]
NUM_LESION_CLASSES = len(LESION_TYPES)


def build_lesion_model(
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    classes: int = NUM_LESION_CLASSES,
    encoder_weights: str | None = None,
) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "segmentation_models_pytorch is required for the lesion model. "
            'Install with: pip install -e ".[dev,ml]"'
        ) from exc

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )


class LesionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = build_lesion_model()

    def forward(self, x):
        return self.model(x)
