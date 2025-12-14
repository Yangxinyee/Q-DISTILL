"""CNN baseline models for chest X-ray classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Conv block: Conv→BN→ReLU→Conv→BN→ReLU→MaxPool"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.pool(x)


class CNNBaseline(nn.Module):
    """Pure CNN for classification (4 conv blocks + MLP)."""
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        
        self.conv_blocks = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class CheXzeroCNNBaseline(nn.Module):
    """CheXzero encoder + MLP classifier."""
    
    def __init__(
        self,
        chexzero_path: str,
        num_classes: int = 2,
        dropout: float = 0.5,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        from .chexzero_vision_encoder import load_chexzero_vision_encoder
        self.vision_encoder = load_chexzero_vision_encoder(
            checkpoint_path=chexzero_path,
            freeze=freeze_encoder,
            device='cpu'
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.vision_encoder(x).last_hidden_state[:, 0, :]
        return self.classifier(features)


def create_cnn_baseline(
    num_classes: int = 2,
    dropout: float = 0.5,
    use_chexzero: bool = False,
    chexzero_path: Optional[str] = None,
    device: str = 'cuda'
) -> nn.Module:
    """Create CNN baseline model."""
    if use_chexzero:
        if chexzero_path is None:
            raise ValueError("chexzero_path required when use_chexzero=True")
        model = CheXzeroCNNBaseline(chexzero_path, num_classes, dropout)
    else:
        model = CNNBaseline(num_classes, dropout)
    
    return model.to(device)


if __name__ == '__main__':
    model = create_cnn_baseline(num_classes=2, device='cpu')
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
