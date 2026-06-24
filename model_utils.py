"""
model_utils.py
Module dung chung cho 03_train.py / 04_evaluate.py / 05_gradcam.py.

Chua dinh nghia 3 model NHE dung de so sanh trong de tai:
  - EfficientNet-Lite3   (model chinh cua de tai, qua timm)
  - MobileNetV3-Small    (model nhe doi chung, qua torchvision)
  - ShuffleNetV2-x1.0    (model nhe doi chung, qua torchvision)

Moi model co "MODEL_CONFIGS" rieng (img_size, mean, std) vi EfficientNet-Lite3
duoc huan luyen goc o do phan giai 280x280 voi chuan hoa [0.5,0.5,0.5] (theo TF gov),
trong khi cac model torchvision dung chuan ImageNet thong thuong (224x224, mean/std ImageNet).
Dung dung config nay giup transfer learning hieu qua hon (khop voi pretrained weights).
"""

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError as e:
    raise ImportError(
        "Thieu thu vien 'timm'. Cai dat bang: pip install timm"
    ) from e

NUM_CLASSES = 4
CLASS_NAMES = ["Level_0", "Level_1", "Level_2", "Level_3"]

# ==========================================
# CAU HINH RIENG CHO TUNG MODEL
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
    """Tra ve model da thay classifier cuoi cho phu hop so class cua bai toan."""
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
            f"Model '{model_name}' khong hop le. Chon trong: {ALL_MODELS}"
        )

    return model


def get_target_layers(model, model_name):
    """Layer conv cuoi cung - noi Grad-CAM tinh gradient/activation."""
    if model_name == "efficientnet_lite3":
        # timm EfficientNet: conv_head la conv 1x1 cuoi truoc global pooling
        return [model.conv_head]

    elif model_name == "mobilenet_small":
        return [model.features[-1]]

    elif model_name == "shufflenet":
        # torchvision ShuffleNetV2: conv5 la block conv cuoi truoc global pool
        return [model.conv5]

    else:
        raise ValueError(f"Model '{model_name}' khong hop le.")


def count_params(model):
    """Dem so tham so (M) va uoc luong dung luong (MB, float32)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # 4 byte / tham so (float32)
    return total, trainable, size_mb


@torch.no_grad()
def measure_inference_time(model, model_name, device, n_warmup=10, n_runs=50):
    """Do thoi gian inference trung binh (ms) cho 1 anh, tren thiet bi `device`."""
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
