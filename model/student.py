"""Lightweight student model with Q-Former architecture for knowledge distillation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pathlib import Path
import json


class LightweightQFormer(nn.Module):
    """Lightweight Q-Former with reduced parameters."""

    def __init__(
        self,
        num_queries: int = 8,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        self.queries = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

        self.ln_self_attn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ln_self_ffn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.cross_attn_image = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ln_cross_attn = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def create_unimodal_mask(self, num_queries: int, num_text: int, device: torch.device) -> torch.Tensor:
        """Unimodal mask: blocks cross-modal attention."""
        total = num_queries + num_text
        mask = torch.full((total, total), float('-inf'), device=device)
        mask[:num_queries, :num_queries] = 0.0
        mask[num_queries:, num_queries:] = 0.0
        return mask

    def create_bidirectional_mask(self, num_queries: int, num_text: int, device: torch.device) -> torch.Tensor:
        """Bidirectional mask: allows all attention."""
        return torch.zeros(num_queries + num_text, num_queries + num_text, device=device)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        use_bidirectional: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Q-Former layers."""
        batch_size = image_features.shape[0]
        device = image_features.device
        queries = self.queries.expand(batch_size, -1, -1)

        if text_features is not None:
            text_features = torch.clamp(
                torch.where(torch.isfinite(text_features), text_features, torch.zeros_like(text_features)),
                -50.0, 50.0
            )
            if text_attention_mask is not None:
                text_features = text_features * text_attention_mask.unsqueeze(-1).float()

            attn_mask = (self.create_bidirectional_mask if use_bidirectional else self.create_unimodal_mask)(
                self.num_queries, text_features.shape[1], device
            )
        else:
            attn_mask = None

        for i in range(len(self.self_attn_layers)):
            # Self-attention
            if text_features is not None:
                combined = torch.clamp(torch.cat([queries, text_features], dim=1), -10.0, 10.0)
                attn_out, _ = self.self_attn_layers[i](combined, combined, combined, attn_mask=attn_mask)
                combined = self.ln_self_attn[i](combined + torch.clamp(attn_out, -50.0, 50.0))
                queries, text_features = combined[:, :self.num_queries], combined[:, self.num_queries:]
            else:
                queries = torch.clamp(queries, -10.0, 10.0)
                attn_out, _ = self.self_attn_layers[i](queries, queries, queries)
                queries = self.ln_self_attn[i](queries + torch.clamp(attn_out, -50.0, 50.0))

            # Cross-attention with image
            img_clamped = torch.clamp(image_features, -10.0, 10.0)
            cross_out, _ = self.cross_attn_image[i](torch.clamp(queries, -10.0, 10.0), img_clamped, img_clamped)
            queries = self.ln_cross_attn[i](queries + torch.clamp(cross_out, -50.0, 50.0))

            # FFN
            ffn_out = self.ffn_layers[i](torch.clamp(queries, -10.0, 10.0))
            queries = self.ln_self_ffn[i](queries + torch.clamp(ffn_out, -50.0, 50.0))

        return queries, text_features


class StudentQFormerModel(nn.Module):
    """Student model with lightweight Q-Former for distillation."""

    def __init__(
        self,
        chexzero_encoder: nn.Module,
        text_encoder: nn.Module,
        num_queries: int = 8,
        hidden_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        freeze_text_encoder: bool = True
    ):
        super().__init__()

        self.chexzero_encoder = chexzero_encoder
        self.chexzero_encoder.eval()
        for p in self.chexzero_encoder.parameters():
            p.requires_grad = False

        self.text_encoder = text_encoder
        if freeze_text_encoder:
            self.text_encoder.eval()
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.qformer = LightweightQFormer(num_queries, hidden_dim, num_heads, num_layers, dropout)

        self.num_classes = num_classes
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_report: bool = False,
        return_logits: bool = True
    ) -> torch.Tensor:
        """Forward pass through student model."""
        with torch.no_grad():
            image_features = self.chexzero_encoder(images).last_hidden_state

        if mask_report or (attention_mask is not None and attention_mask.sum() == 0):
            text_features, text_attention_mask = None, None
        else:
            with torch.no_grad():
                text_features = self.text_encoder(input_ids, attention_mask)
            text_attention_mask = attention_mask

        queries, _ = self.qformer(image_features, text_features, text_attention_mask, use_bidirectional=True)
        features = queries.mean(dim=1)

        if return_logits:
            return self.classification_head(features)
        return features


def create_student_model(
    chexzero_path: str,
    teacher_checkpoint: str,
    num_classes: int = 2,
    device: str = "cuda",
    freeze_text_encoder: bool = True,
    num_queries: int = 8,
    num_layers: int = 4,
    num_heads: int = 12
) -> StudentQFormerModel:
    """Create student model with Q-Former."""
    teacher_dir = Path(teacher_checkpoint).parent
    config_path = teacher_dir / 'config.json'

    # Load vocab_size from teacher config
    vocab_size = 10000
    if config_path.exists():
        with open(config_path, 'r') as f:
            teacher_config = json.load(f)
        vocab_size = teacher_config.get('vocab_size', 10000)

    # Load encoders
    from model.chexzero_vision_encoder import load_chexzero_vision_encoder
    chexzero_encoder = load_chexzero_vision_encoder(chexzero_path, freeze=True, device=device)

    from model.teacher import SimpleTextEncoder
    checkpoint = torch.load(teacher_checkpoint, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    
    text_encoder = SimpleTextEncoder(vocab_size=vocab_size, hidden_dim=768)
    text_state = {k.replace('text_encoder.', ''): v for k, v in state.items() if k.startswith('text_encoder.')}
    if text_state:
        text_encoder.load_state_dict(text_state)

    model = StudentQFormerModel(
        chexzero_encoder=chexzero_encoder,
        text_encoder=text_encoder,
        num_queries=num_queries,
        hidden_dim=768,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
        num_classes=num_classes,
        freeze_text_encoder=freeze_text_encoder
    ).to(device)

    return model
