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
from torchvision import transforms

import cv2
import numpy as np
from PIL import ImageOps

try:
    import timm
except ImportError as e:
    raise ImportError(
        "Missing the 'timm' package. Install it with: pip install timm"
    ) from e

NUM_CLASSES = 4
CLASS_NAMES = ["Level_0", "Level_1", "Level_2", "Level_3"]
ORDINAL_OUTPUTS = NUM_CLASSES - 1


def encode_ordinal_targets(labels, num_classes=NUM_CLASSES):
    """Encode class indices into ordinal binary decomposition targets.

    Each label k is converted to a binary vector of length (num_classes - 1)
    where positions 0..k-1 are 1 and positions k..end are 0.

    NOTE: This is threshold-based ordinal encoding (Frank & Hall / Niu et al.),
    NOT strict CORAL (Cao, Mirjalili & Raschka, 2020), which requires a shared
    weight vector across all thresholds (only biases differ) to guarantee rank
    monotonicity by architecture.  Our model uses nn.Linear(in_features, K-1)
    with independent per-threshold weights.
    """
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels, dtype=torch.long)
    labels = labels.long().view(-1)
    thresholds = torch.arange(num_classes - 1, device=labels.device).unsqueeze(0)
    return (labels.unsqueeze(1) > thresholds).float()


def ordinal_logits_to_class_indices(logits):
    """Decode ordinal logits into class indices by thresholding at 0.5."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


def ordinal_logits_to_class_probabilities(logits):
    """Convert ordinal logits into a class-probability tensor for evaluation."""
    threshold_probs = torch.sigmoid(logits)
    class_probs = []
    class_probs.append(1.0 - threshold_probs[:, 0])
    for idx in range(1, threshold_probs.size(1)):
        class_probs.append(threshold_probs[:, idx - 1] - threshold_probs[:, idx])
    class_probs.append(threshold_probs[:, -1])
    probs = torch.stack(class_probs, dim=1)
    # clamp(min=0) + renormalize: because our model uses independent per-threshold
    # weights (not shared as in true CORAL), the sigmoid outputs are NOT guaranteed
    # to be monotonically decreasing.  This can produce negative "probabilities"
    # when P(Y > k-1) < P(Y > k).  Clamping and renormalizing is a pragmatic fix.
    probs = torch.clamp(probs, min=0.0)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs


class OrdinalClassTarget:
    """Grad-CAM target that scores a specific ordinal class probability."""

    def __init__(self, class_idx, num_classes=NUM_CLASSES):
        self.class_idx = class_idx
        self.num_classes = num_classes

    def __call__(self, model_output):
        probs = ordinal_logits_to_class_probabilities(model_output)
        return probs[:, self.class_idx].sum()

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


class FaceROICrop:
    # Class-level counters for face-detection statistics (not thread-safe,
    # intended for diagnostic logging only — see print_stats()).
    _detected_count = 0
    _fallback_count = 0

    def __init__(self, padding=0.15):
        self.padding = padding
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ------------------------------------------------------------------
    # Pickle support: cv2.CascadeClassifier is not picklable, which breaks
    # DataLoader(num_workers>0) on Windows (spawn).  We exclude it from
    # the pickle state and lazy-load it back the first time __call__ runs
    # in a new process.
    # ------------------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("cascade", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # cascade will be lazy-loaded in __call__

    def _ensure_cascade(self):
        """Lazy-load the Haar cascade if it hasn't been loaded yet (e.g. after unpickling)."""
        if not hasattr(self, "cascade"):
            self.cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

    def _center_square_crop(self, img):
        width, height = img.size
        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return img.crop((left, top, left + side, top + side))

    def _expand_box(self, width, height, x, y, w, h):
        side = int(max(w, h) * (1 + self.padding))
        side = max(1, min(side, width, height))
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        left = int(round(center_x - (side / 2.0)))
        top = int(round(center_y - (side / 2.0)))
        left = max(0, min(left, width - side))
        top = max(0, min(top, height - side))
        right = left + side
        bottom = top + side
        return left, top, right, bottom

    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")

        self._ensure_cascade()

        if self.cascade.empty():
            FaceROICrop._fallback_count += 1
            return self._center_square_crop(img)

        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        if len(faces) == 0:
            FaceROICrop._fallback_count += 1
            return self._center_square_crop(img)

        FaceROICrop._detected_count += 1
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        return img.crop(self._expand_box(img.size[0], img.size[1], x, y, w, h))

    @classmethod
    def print_stats(cls):
        """Print face-detection vs. fallback statistics."""
        total = cls._detected_count + cls._fallback_count
        if total == 0:
            print("[FaceROICrop] No images processed yet.")
            return
        det_pct = cls._detected_count / total * 100
        fb_pct = cls._fallback_count / total * 100
        print(f"[FaceROICrop] Total: {total} | "
              f"Detected: {cls._detected_count} ({det_pct:.1f}%) | "
              f"Fallback: {cls._fallback_count} ({fb_pct:.1f}%)")

    @classmethod
    def reset_stats(cls):
        """Reset detection counters."""
        cls._detected_count = 0
        cls._fallback_count = 0


def build_image_transform(
    model_name,
    train=False,
    enable_color_jitter=True,
    random_erasing_p=0.25,
):
    cfg = MODEL_CONFIGS[model_name]
    transform_list = list(build_pil_transform(model_name).transforms)

    if train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        if enable_color_jitter:
            transform_list.append(
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1)
            )

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    if train:
        transform_list.append(
            transforms.RandomErasing(p=random_erasing_p, scale=(0.02, 0.15))
        )

    return transforms.Compose(transform_list)


def build_pil_transform(model_name):
    cfg = MODEL_CONFIGS[model_name]
    return transforms.Compose([
        FaceROICrop(),
        transforms.Resize(
            cfg["img_size"],
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.Lambda(lambda img: ImageOps.autocontrast(img)),
    ])


def get_model(model_name, num_classes=NUM_CLASSES, pretrained=True, ordinal=False):
    """Return a model whose final classifier is replaced to match this task's class count."""
    output_units = num_classes - 1 if ordinal else num_classes

    if model_name == "efficientnet_lite3":
        model = timm.create_model(
            "tf_efficientnet_lite3", pretrained=pretrained, num_classes=output_units
        )

    elif model_name == "mobilenet_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, output_units)

    elif model_name == "shufflenet":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, output_units)

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
