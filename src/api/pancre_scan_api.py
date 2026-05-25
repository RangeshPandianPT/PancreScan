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
    if model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    if model_name == "unet":
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from src.models.unet import UNetMultiTask
        model = UNetMultiTask(n_channels=3, n_classes=1, num_cls_classes=1)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "densenet121":
        return model.features
    if model_name == "efficientnet_v2_s":
        return model.features[-1]
    if model_name == "convnext_tiny":
        return model.features[-1]
    if model_name == "unet":
        return model.features
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

        self.models_cache = {}
        self.grad_cam_cache = {}
        self.last_segment_mask = None
        self.last_grad_cam = None

    def _map_display_name_to_internal(self, display_name: str) -> str:
        if not display_name:
            return os.getenv("PRIMARY_MODEL", "efficientnet_v2_s")
        lower_name = display_name.lower()
        if "efficientnet" in lower_name: return "efficientnet_v2_s"
        if "densenet" in lower_name: return "densenet121"
        if "convnext" in lower_name: return "convnext_tiny"
        if "unet" in lower_name: return "unet"
        return os.getenv("PRIMARY_MODEL", "efficientnet_v2_s")

    def get_model_and_cam(self, display_name: str):
        internal_name = self._map_display_name_to_internal(display_name)
        if internal_name not in self.models_cache:
            # Determine path
            path = os.getenv(f"{internal_name.upper()}_CHECKPOINT", f"outputs/{internal_name}_best.pt")
            config = ModelConfig(model_name=internal_name, checkpoint_path=path)
            model = load_model(config, self.device)
            # Ensure params require grad if we need GradCAM
            for param in model.parameters():
                param.requires_grad = True
            self.models_cache[internal_name] = model
            target_layer = get_target_layer(model, internal_name)
            self.grad_cam_cache[internal_name] = GradCAM(model, target_layer)
        return self.models_cache[internal_name], self.grad_cam_cache[internal_name]

    def predict_logits(self, image_tensor: torch.Tensor, model_name: str) -> torch.Tensor:
        model, grad_cam = self.get_model_and_cam(model_name)
        self.last_grad_cam = grad_cam
        
        out = model(image_tensor)
        if isinstance(out, tuple) and len(out) == 2:
            self.last_segment_mask, logits = out
        else:
            self.last_segment_mask = None
            logits = out
            
        return logits


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
    model_name: Optional[str] = Query(default=None, description="Model architecture to use."),
) -> dict:
    image = prepare_image(file)
    input_tensor = bundle.preprocess(image).unsqueeze(0).to(bundle.device)
    input_tensor.requires_grad_(heatmap)

    start = time.perf_counter()
    if heatmap:
        # We need gradients for GradCAM, so no torch.no_grad()
        logits = bundle.predict_logits(input_tensor, model_name)
    else:
        with torch.no_grad():
            logits = bundle.predict_logits(input_tensor, model_name)

    if logits.shape[1] == 1:
        pos_prob = torch.sigmoid(logits).item()
    else:
        probs = torch.softmax(logits, dim=1)
        pos_prob = probs[0, bundle.positive_index].item()

    inference_ms = (time.perf_counter() - start) * 1000.0
    diagnosis = (
        bundle.positive_name if pos_prob >= bundle.pos_threshold else bundle.class_names[1 - bundle.positive_index]
    )

    heatmap_b64: Optional[str] = None
    mask_b64: Optional[str] = None
    
    if heatmap and diagnosis == bundle.positive_name and bundle.last_grad_cam is not None:
        score = logits[0, 0] if logits.shape[1] == 1 else logits[0, bundle.positive_index]
        cam = bundle.last_grad_cam.generate(score, (bundle.image_size, bundle.image_size))
        overlay = make_overlay(image.resize((bundle.image_size, bundle.image_size)), cam)
        heatmap_b64 = image_to_base64(overlay)

    # Inconclusive Logic for UNet
    if bundle.last_segment_mask is not None:
        mask_prob = torch.sigmoid(bundle.last_segment_mask[0, 0])
        mask_area = (mask_prob > 0.5).sum().item()
        
        # Create visual mask overlay
        mask_np = mask_prob.detach().cpu().numpy()
        base_img = np.array(image.resize((bundle.image_size, bundle.image_size))).astype(np.float32) / 255.0
        mask_heat = np.zeros_like(base_img)
        mask_heat[..., 0] = mask_np # Red tint for mask
        
        mask_overlay = np.clip(base_img * 0.6 + mask_heat * 0.4, 0.0, 1.0)
        mask_b64 = image_to_base64(Image.fromarray((mask_overlay * 255).astype(np.uint8)))
        
        if mask_area < 10 or (0.45 < pos_prob < 0.55):
            diagnosis = "Inconclusive"

    return {
        "diagnosis": diagnosis,
        "confidence": pos_prob,
        "inference_ms": inference_ms,
        "positive_class": bundle.positive_name,
        "positive_threshold": bundle.pos_threshold,
        "heatmap_png_base64": heatmap_b64,
        "mask_png_base64": mask_b64,
    }
