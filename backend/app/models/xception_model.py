import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def load_deepfake_efficientnet_b0(weights_path: str, num_classes: int = 2):
    """
    Load the REAL pretrained EfficientNet-B0 deepfake model.
    The .pth file you uploaded will be loaded here.
    """
    print("[MODEL] Loading EfficientNet-B0 backbone...")

    base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )

    print(f"[MODEL] Loading weights from: {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")

    # Some training scripts store "state_dict" inside dict
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state = {}
    for k, v in state_dict.items():
        # Remove prefixes like module.model.
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_state[k] = v

    missing, unexpected = base.load_state_dict(new_state, strict=False)
    print("[MODEL] Missing keys:", missing)
    print("[MODEL] Unexpected keys:", unexpected)

    class DeepfakeClassifier(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            logits = self.m(x)
            return torch.softmax(logits, dim=1)[:, 1:2]  # prob(fake)

    print("[MODEL] Loaded successfully!")
    return DeepfakeClassifier(base)
