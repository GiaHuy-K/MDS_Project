"""
model_utils.py
Shared module for scripts/03_train.py / 04_evaluate.py / 05_gradcam.py.

Defines the 3 LIGHTWEIGHT models compared in this study:
  - EfficientNet-Lite3   (the study's main model, via timm)
  - MobileNetV3-Small    (lightweight baseline, via torchvision)
  - ShuffleNetV2-x1.0    (lightweight baseline, via torchvision)

Each model has its own entry in MODEL_CONFIGS (img_size, mean, std). EfficientNet-Lite3
was originally trained at 280x280 resolution with [0.5, 0.5, 0.5] normalization (TF/Google
convention), while the torchvision models use the standard ImageNet setup (224x224,
ImageNet mean/std). Using the correct per-model config makes transfer learning more
effective (it matches how the pretrained weights were produced).
"""

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError as e:
    raise ImportError(
        "Missing the 'timm' package. Install it with: pip install timm"
    ) from e

NUM_CLASSES = 4
CLASS_NAMES = ["Level_0", "Level_1", "Level_2", "Level_3"]

# ==========================================
# PER-MODEL CONFIGURATION
# ==========================================
MODEL_CONFIGS = {
    "efficientnet_lite3": {
        "display_name": "EfficientNet-Lite3",
        "img_size": 280,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    },
    "mobilenet_small": {
        "display_name": "MobileNetV3-Small",
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "shufflenet": {
        "display_name": "ShuffleNetV2-x1.0",
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}

ALL_MODELS = list(MODEL_CONFIGS.keys())


def get_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    """Return a model whose final classifier is replaced to match this task's class count."""
    if model_name == "efficientnet_lite3":
        model = timm.create_model(
            "tf_efficientnet_lite3", pretrained=pretrained, num_classes=num_classes
        )

    elif model_name == "mobilenet_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    elif model_name == "shufflenet":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Invalid model '{model_name}'. Choose one of: {ALL_MODELS}"
        )

    return model


def get_target_layers(model, model_name):
    """The last convolutional layer - where Grad-CAM computes gradients/activations."""
    if model_name == "efficientnet_lite3":
        # timm EfficientNet: conv_head is the final 1x1 conv before global pooling
        return [model.conv_head]

    elif model_name == "mobilenet_small":
        return [model.features[-1]]

    elif model_name == "shufflenet":
        # torchvision ShuffleNetV2: conv5 is the last conv block before global pooling
        return [model.conv5]

    else:
        raise ValueError(f"Invalid model '{model_name}'.")


def get_param_groups(model, model_name, lr_backbone, lr_head):
    """Return optimizer param_groups: a small LR for the pretrained backbone and a
    larger LR for the freshly initialized classifier head. This reduces overfitting
    when fine-tuning the whole network on a small dataset (see log.md - Round 3)."""
    if model_name == "efficientnet_lite3":
        head_module = model.classifier
    elif model_name == "mobilenet_small":
        head_module = model.classifier
    elif model_name == "shufflenet":
        head_module = model.fc
    else:
        raise ValueError(f"Invalid model '{model_name}'.")

    head_params = list(head_module.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]


def count_params(model):
    """Count parameters (in millions) and estimate size (MB, float32)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # 4 bytes per parameter (float32)
    return total, trainable, size_mb


@torch.no_grad()
def measure_inference_time(model, model_name, device, n_warmup=10, n_runs=50):
    """Measure average inference time (ms) for a single image on the given `device`."""
    import time

    img_size = MODEL_CONFIGS[model_name]["img_size"]
    model.eval().to(device)
    dummy = torch.randn(1, 3, img_size, img_size, device=device)

    for _ in range(n_warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / n_runs) * 1000
    return ms_per_image
