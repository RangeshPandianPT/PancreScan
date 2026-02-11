import base64
import io
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    model_name: str
    checkpoint_path: str


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, score: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=input_size, mode="bilinear", align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.squeeze(0).squeeze(0)


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "densenet121":
        return model.features
    if model_name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"Unsupported model: {model_name}")


def load_model(config: ModelConfig, device: torch.device) -> nn.Module:
    model = build_model(config.model_name).to(device)
    if config.checkpoint_path and os.path.exists(config.checkpoint_path):
        state = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(state)
    else:
        print(
            f"Warning: checkpoint not found at {config.checkpoint_path}. "
            "Using ImageNet weights only."
        )
    model.eval()
    return model


def build_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_class_names(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else ["normal", "pancreatic_tumor"]


def make_overlay(image: Image.Image, cam: torch.Tensor) -> Image.Image:
    cam_np = cam.detach().cpu().numpy()
    cam_np = np.clip(cam_np, 0.0, 1.0)
    base = np.array(image).astype(np.float32) / 255.0
    heat = np.zeros_like(base)
    heat[..., 0] = cam_np
    overlay = np.clip(base * 0.6 + heat * 0.4, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded


def parse_weights(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("ENSEMBLE_WEIGHTS must be two comma-separated values")
    weights = [float(parts[0]), float(parts[1])]
    total = weights[0] + weights[1]
    if total <= 0:
        raise ValueError("ENSEMBLE_WEIGHTS must sum to a positive value")
    return weights[0] / total, weights[1] / total


app = FastAPI(title="PancreScan 2.0 API", version="0.1.0")


class ModelBundle:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = int(os.getenv("IMAGE_SIZE", "224"))
        self.class_names = parse_class_names(os.getenv("CLASS_NAMES", "normal,pancreatic_tumor"))
        self.positive_name = os.getenv("POSITIVE_CLASS", "pancreatic_tumor")
        self.positive_index = (
            self.class_names.index(self.positive_name)
            if self.positive_name in self.class_names
            else 1
        )
        self.pos_threshold = float(os.getenv("POSITIVE_THRESHOLD", "0.4"))
        self.preprocess = build_preprocess(self.image_size)

        primary = ModelConfig(
            model_name=os.getenv("PRIMARY_MODEL", "efficientnet_b0"),
            checkpoint_path=os.getenv("PRIMARY_CHECKPOINT", "outputs/efficientnet_b0_best.pt"),
        )
        secondary_path = os.getenv("SECONDARY_CHECKPOINT")
        secondary_name = os.getenv("SECONDARY_MODEL", "densenet121")
        self.primary = load_model(primary, self.device)
        self.secondary = None
        self.ensemble_weights = (0.5, 0.5)
        if secondary_path:
            secondary = ModelConfig(model_name=secondary_name, checkpoint_path=secondary_path)
            self.secondary = load_model(secondary, self.device)
            self.ensemble_weights = parse_weights(os.getenv("ENSEMBLE_WEIGHTS", "0.5,0.5"))

        target_layer = get_target_layer(self.primary, primary.model_name)
        self.grad_cam = GradCAM(self.primary, target_layer)

    def predict_logits(self, image_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.primary(image_tensor)
        if self.secondary is None:
            return logits
        logits_secondary = self.secondary(image_tensor)
        return logits * self.ensemble_weights[0] + logits_secondary * self.ensemble_weights[1]


bundle = ModelBundle()


def prepare_image(file: UploadFile) -> Image.Image:
    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image upload") from exc
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Query(default=False, description="Return Grad-CAM overlay when positive."),
) -> dict:
    image = prepare_image(file)
    input_tensor = bundle.preprocess(image).unsqueeze(0).to(bundle.device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = bundle.predict_logits(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pos_prob = probs[0, bundle.positive_index].item()

    inference_ms = (time.perf_counter() - start) * 1000.0
    diagnosis = (
        bundle.positive_name if pos_prob >= bundle.pos_threshold else bundle.class_names[1 - bundle.positive_index]
    )

    heatmap_b64: Optional[str] = None
    if heatmap and diagnosis == bundle.positive_name:
        score = logits[0, bundle.positive_index]
        cam = bundle.grad_cam.generate(score, (bundle.image_size, bundle.image_size))
        overlay = make_overlay(image.resize((bundle.image_size, bundle.image_size)), cam)
        heatmap_b64 = image_to_base64(overlay)

    return {
        "diagnosis": diagnosis,
        "confidence": pos_prob,
        "inference_ms": inference_ms,
        "positive_class": bundle.positive_name,
        "positive_threshold": bundle.pos_threshold,
        "heatmap_png_base64": heatmap_b64,
    }
