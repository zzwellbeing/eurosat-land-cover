"""Compact baseline CNN and checkpoint helpers."""
from pathlib import Path

import torch
from torch import nn

from src.data import CLASSES, ROOT

MODEL_NAME = "EuroSAT CNN"
CHECKPOINT = ROOT / "models" / "best_model.pth"


class LandCoverCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            self.block(3, 24), self.block(24, 48), self.block(48, 80),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(80 * 4, 96), nn.ReLU(),
            nn.Dropout(0.25), nn.Linear(96, len(CLASSES)),
        )

    @staticmethod
    def block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2),
        )

    def forward(self, pixels):
        return self.head(self.features(pixels))


def load_model(path: Path = CHECKPOINT, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["labels"] != list(CLASSES):
        raise ValueError("Checkpoint labels do not match the current dataset.")
    model = LandCoverCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
