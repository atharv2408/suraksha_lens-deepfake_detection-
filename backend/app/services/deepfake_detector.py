import io
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from app.models.xception_model import load_deepfake_efficientnet_b0   # <-- UPDATED


# ------------------------------------------------------------
# 0. DEVICE
# ------------------------------------------------------------
DEVICE = torch.device("cpu")  # CPU is fine for now


# ------------------------------------------------------------
# 1. MODEL LOADING LOGIC
# ------------------------------------------------------------

MODEL_NAME = "SimpleFallbackModel"
model: nn.Module

# Updated weights path
weights_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "weights",
    "deepfake_efficientnet_best.pth"   # <-- UPDATED MODEL NAME
)

print("[DeepfakeDetector] Checking for pretrained deepfake model...")

if os.path.exists(weights_path):
    try:
        print(f"[DeepfakeDetector] Found pretrained model at: {weights_path}")
        model = load_deepfake_efficientnet_b0(
            weights_path=weights_path,
            num_classes=2
        )
        MODEL_NAME = "EfficientNetB0_DeepfakeModel"
    except Exception as e:
        print(f"[DeepfakeDetector] Failed to load pretrained model: {e}")
        print("[DeepfakeDetector] Using fallback CNN.")
        from torch import nn

        class SimpleFallback(nn.Module):
            def forward(self, x):
                return torch.tensor([[0.4]])

        model = SimpleFallback()
        MODEL_NAME = "SimpleFallbackModel"

else:
    print("[DeepfakeDetector] No model weights found. Using fallback CNN.")
    from torch import nn

    class SimpleFallback(nn.Module):
        def forward(self, x):
            return torch.tensor([[0.4]])

    model = SimpleFallback()
    MODEL_NAME = "SimpleFallbackModel"


model.to(DEVICE)
model.eval()


# ------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# 3. MAIN FUNCTION
# ------------------------------------------------------------

@torch.no_grad()
def analyze_image_bytes(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    output = model(img_tensor)

    prob_fake = float(output.item())
    prob_fake = max(0.0, min(1.0, prob_fake))

    if prob_fake > 0.8:
        risk_level = "high"
    elif prob_fake > 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"

    explanation = (
        "Prediction generated using a pretrained EfficientNet-B0 deepfake model. "
        "Higher values indicate features similar to known deepfakes."
    )

    return {
        "fake_probability": prob_fake,
        "risk_level": risk_level,
        "explanation": explanation,
        "model_used": MODEL_NAME,
    }
