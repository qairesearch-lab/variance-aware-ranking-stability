"""Versioned image/model recipe for the JIIM reviewer extension.

This is deliberately separate from the public v1.0-submission runner. The
secondary variant changes Swin validation and test only; training augmentation
is held fixed, so the sensitivity has a precise, narrow interpretation.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
VARIANTS = {
    "shared": CONFIG_DIR / "preprocessing_shared_v1.yaml",
    "swin_weight_eval": CONFIG_DIR / "preprocessing_swin_weight_eval_v1.yaml",
}
DATASETS = ("isic2019", "mura")
MODELS = ("resnet18", "resnet50", "densenet121", "efficientnet_b0", "swin_t")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class PadSquare:
    """Symmetric black padding, with the one-pixel remainder at right/bottom."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        return ImageOps.expand(
            image,
            border=(left, top, side - width - left, side - height - top),
            fill=0,
        )


def load_variant(name: str) -> tuple[dict, Path]:
    if name not in VARIANTS:
        raise ValueError(f"Unknown preprocessing variant: {name}")
    path = VARIANTS[name]
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["design_lock"] != "extension_protocol_design_locked_v0.3.yaml":
        raise ValueError("Variant does not refer to the approved design lock")
    return config, path


def decode_mode(dataset: str) -> str:
    if dataset == "isic2019":
        return "RGB"
    if dataset == "mura":
        return "L"
    raise ValueError(dataset)


def decode_image(image: Image.Image, dataset: str) -> Image.Image:
    # MURA is explicitly decoded as luminance and then repeated into 3 channels.
    # This differs from PIL RGB decoding of an arbitrary source color image.
    return image.convert(decode_mode(dataset)).convert("RGB")


def build_transforms(dataset: str, model: str, variant: str):
    if dataset not in DATASETS or model not in MODELS:
        raise ValueError((dataset, model))
    config, _ = load_variant(variant)
    normalize = transforms.Normalize(MEAN, STD)
    if dataset == "isic2019":
        train_ops = [
            transforms.RandomResizedCrop(
                224, scale=(0.8, 1.0), ratio=(0.75, 4 / 3),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10, interpolation=InterpolationMode.NEAREST, fill=0),
        ]
        eval_prefix = []
    else:
        train_ops = [
            PadSquare(),
            transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomRotation(10, interpolation=InterpolationMode.NEAREST, fill=0),
        ]
        eval_prefix = [PadSquare()]
    train_transform = transforms.Compose(train_ops + [transforms.ToTensor(), normalize])
    if model == "swin_t" and config["swin_weight_specific_eval"]:
        weight_transform = models.Swin_T_Weights.IMAGENET1K_V1.transforms()
        if (
            weight_transform.resize_size != [232]
            or weight_transform.crop_size != [224]
            or weight_transform.interpolation != InterpolationMode.BICUBIC
            or tuple(weight_transform.mean) != MEAN
            or tuple(weight_transform.std) != STD
        ):
            raise RuntimeError("Installed torchvision Swin-T V1 inference recipe changed")
        eval_transform = transforms.Compose(eval_prefix + [weight_transform])
    else:
        eval_transform = transforms.Compose(
            eval_prefix
            + [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return train_transform, eval_transform


def build_model(model_name: str, classes: int, *, pretrained: bool = True) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(model_name)
    weights = getattr(models.get_model_weights(model_name), "IMAGENET1K_V1") if pretrained else None
    model = models.get_model(model_name, weights=weights)
    if model_name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, classes)
        trainable_modules = (model.layer4, model.fc)
    elif model_name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, classes)
        trainable_modules = (model.features.denseblock4, model.features.norm5, model.classifier)
    elif model_name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, classes)
        trainable_modules = (model.features[-1], model.classifier)
    else:
        model.head = nn.Linear(model.head.in_features, classes)
        trainable_modules = (model.features[-1], model.norm, model.head)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return model


def rule_a_update(validation_loss: float, best_loss: float, non_improving: int) -> tuple[bool, float, int, bool]:
    improved = validation_loss < best_loss - 0.001
    if improved:
        return True, validation_loss, 0, False
    non_improving += 1
    return False, best_loss, non_improving, non_improving >= 7
