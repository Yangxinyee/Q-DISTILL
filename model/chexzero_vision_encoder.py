"""CheXzero vision encoder wrapper for CLIP model."""

import torch
import torch.nn as nn
import os
from typing import Optional


class CheXzeroVisionEncoder(nn.Module):
    """CheXzero Vision Encoder (ViT-B/32)."""

    def __init__(self, checkpoint_path: str, image_resolution: int = 320, freeze: bool = True):
        super().__init__()
        self.image_resolution = image_resolution
        self.hidden_size = 768

        self.clip_model = self._load_clip(checkpoint_path)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def _load_clip(self, model_path: str):
        """Load CLIP model from checkpoint."""
        try:
            from .clip_model import CLIP
        except ImportError:
            from clip_model import CLIP

        params = {
            'embed_dim': 512,
            'image_resolution': 224,
            'vision_layers': 12,
            'vision_width': 768,
            'vision_patch_size': 32,
            'context_length': 77,
            'vocab_size': 49408,
            'transformer_width': 512,
            'transformer_heads': 8,
            'transformer_layers': 12
        }

        model = CLIP(**params)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint)
        model.eval()

        return model

    def forward(self, pixel_values: torch.Tensor):
        """Extract patch features from images."""
        visual = self.clip_model.visual

        # Patch embedding
        x = visual.conv1(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # Add CLS token and positional embeddings
        cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        class Outputs:
            def __init__(self, hidden_states):
                self.last_hidden_state = hidden_states

        return Outputs(x)

    @property
    def config(self):
        """Config for compatibility with HuggingFace-style code."""
        class Config:
            hidden_size = 768
            num_hidden_layers = 12
            num_attention_heads = 12
        return Config()


def load_chexzero_vision_encoder(
    checkpoint_path: str = "checkpt/chexzero_best.pt",
    image_resolution: int = 320,
    freeze: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Load CheXzero vision encoder from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    encoder = CheXzeroVisionEncoder(
        checkpoint_path=checkpoint_path,
        image_resolution=image_resolution,
        freeze=freeze
    )
    return encoder.to(device)


if __name__ == "__main__":
    checkpoint_path = "checkpt/chexzero_best.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
    else:
        encoder = load_chexzero_vision_encoder(checkpoint_path, device='cpu')
        dummy = torch.randn(2, 3, 224, 224)
        
        with torch.no_grad():
            out = encoder(dummy)
        
        print(f"Input: {dummy.shape}, Output: {out.last_hidden_state.shape}")
