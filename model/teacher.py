"""Teacher model with Q-Former for multimodal vision-language fusion."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import re
from collections import Counter


class QFormer(nn.Module):
    """Q-Former module for vision-language fusion using learnable queries."""

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
        
        self.itm_head = nn.Linear(hidden_dim, 2)

        self.self_attn_ffn_layers = nn.ModuleList([
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

    def _sanitize(self, x: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
        """Sanitize tensor: replace NaN/Inf and clamp."""
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        return torch.clamp(x, -clamp, clamp)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        use_bidirectional_mask: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through Q-Former layers."""
        batch_size = image_features.shape[0]
        device = image_features.device
        
        image_features = self._sanitize(image_features)
        queries = self.queries.expand(batch_size, -1, -1)

        if text_features is not None:
            text_features = self._sanitize(text_features, 50.0)
            num_text = text_features.shape[1]
            attn_mask = (self.create_bidirectional_mask if use_bidirectional_mask else self.create_unimodal_mask)(
                self.num_queries, num_text, device
            )
        else:
            attn_mask = None

        for i in range(len(self.self_attn_layers)):
            if text_features is not None:
                combined = self._sanitize(torch.cat([queries, text_features], dim=1))
                attn_out, _ = self.self_attn_layers[i](combined, combined, combined, attn_mask=attn_mask)
                combined = self.ln_self_attn[i](combined + self._sanitize(attn_out))
                queries, text_features = combined[:, :self.num_queries], combined[:, self.num_queries:]
            else:
                queries = self._sanitize(queries)
                attn_out, _ = self.self_attn_layers[i](queries, queries, queries)
                queries = self.ln_self_attn[i](queries + self._sanitize(attn_out))

            cross_out, _ = self.cross_attn_image[i](self._sanitize(queries), image_features, image_features)
            queries = self.ln_cross_attn[i](queries + self._sanitize(cross_out))

            ffn_out = self.self_attn_ffn_layers[i](self._sanitize(queries))
            queries = self.ln_self_ffn[i](queries + self._sanitize(ffn_out))

        queries = self._sanitize(queries, 50.0)
        if text_features is not None:
            text_features = self._sanitize(text_features, 50.0)

        return queries, text_features
    
    def compute_itm_logits(self, queries: torch.Tensor) -> torch.Tensor:
        """Compute ITM logits for image-text matching."""
        queries = self._sanitize(queries, 50.0)
        logits = self.itm_head(queries).mean(dim=1)
        return self._sanitize(logits, 50.0)


class SimpleTextEncoder(nn.Module):
    """Simple text encoder using embeddings and positional encoding."""
    
    def __init__(self, vocab_size: int = 10000, hidden_dim: int = 768, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        nn.init.xavier_uniform_(self.token_embedding.weight)
        self.token_embedding.weight.data *= 0.1
        
        self.positional_embedding = nn.Parameter(torch.empty(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode tokens to feature vectors."""
        seq_len = input_ids.shape[1]
        input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        
        x = self.token_embedding(input_ids)
        x = torch.clamp(x + self.positional_embedding[:, :seq_len], -10.0, 10.0)
        x = self.dropout(self.ln(x))
        
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        
        return torch.clamp(x, -50.0, 50.0)


class ClassificationHead(nn.Module):
    """MLP classification head."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(torch.where(torch.isfinite(x), x, torch.zeros_like(x)), -50.0, 50.0)
        if x.dim() == 3:
            x = x.mean(dim=1)
        return torch.clamp(self.net(x), -50.0, 50.0)


class SimpleTokenizer:
    """Simple tokenizer for text encoding."""
    
    def __init__(self, vocab_size: int = 10000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.vocab = {}
        self.unk_token_id = 0
        self.pad_token_id = 1
        
    def build_vocab(self, texts: list):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            word_counts.update(re.findall(r'\b\w+\b', text.lower()))
        
        most_common = word_counts.most_common(self.vocab_size - 2)
        self.vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
        self.vocab['<UNK>'] = self.unk_token_id
        self.vocab['<PAD>'] = self.pad_token_id
    
    def load_vocab(self, vocab_path: str):
        """Load vocabulary from JSON file."""
        import json
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.unk_token_id = data.get('unk_token_id', 0)
        self.pad_token_id = data.get('pad_token_id', 1)
        self.vocab_size = data.get('vocab_size', len(self.vocab))
        
    def encode(self, texts: list, padding: bool = True, truncation: bool = True) -> dict:
        """Encode texts to token IDs."""
        input_ids_list, attention_mask_list = [], []
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            ids = [self.vocab.get(w, self.unk_token_id) for w in words]
            
            if truncation and len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            if padding:
                pad_len = self.max_length - len(ids)
                mask = [1] * len(ids) + [0] * pad_len
                ids = ids + [self.pad_token_id] * pad_len
            else:
                mask = [1] * len(ids)
            
            input_ids_list.append(ids)
            attention_mask_list.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
        }


class TeacherModel(nn.Module):
    """Teacher model with CheXzero vision encoder and trainable Q-Former."""

    def __init__(
        self,
        vision_encoder: nn.Module,
        vocab_size: int = 10000,
        num_queries: int = 8,
        hidden_dim: int = 768,
        num_layers: int = 3,
        num_heads: int = 12,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()

        # CheXzero vision encoder (frozen)
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Text encoder and tokenizer (trainable)
        self.text_encoder = SimpleTextEncoder(vocab_size, hidden_dim, 512, dropout)
        self.text_tokenizer = SimpleTokenizer(vocab_size, 512)

        # Q-Former (trainable)
        self.qformer = QFormer(num_queries, hidden_dim, num_heads, num_layers, dropout)
        
        # Classification head (trainable)
        self.num_classes = num_classes
        self.classification_head = ClassificationHead(hidden_dim, 512, num_classes, dropout)
        self.hidden_dim = hidden_dim

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using frozen CheXzero encoder."""
        images = torch.clamp(torch.where(torch.isfinite(images), images, torch.zeros_like(images)), -10.0, 10.0)
        with torch.no_grad():
            features = self.vision_encoder(images).last_hidden_state
        return torch.clamp(features, -50.0, 50.0)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using trainable text encoder."""
        return self.text_encoder(
            torch.clamp(input_ids, 0, self.text_encoder.vocab_size - 1),
            torch.clamp(attention_mask, 0, 1)
        )
    
    def get_text_cls_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get CLS token embedding."""
        return self.encode_text(input_ids, attention_mask)[:, 0]

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_dropout_rate: float = 0.0,
        compute_itm: bool = False
    ) -> Tuple:
        """Forward pass through teacher model."""
        image_features = self.encode_image(images)
        
        if input_ids is None:
            text_features = torch.zeros(images.shape[0], 32, self.hidden_dim, device=images.device)
        else:
            text_features = self.encode_text(input_ids, attention_mask)
        
        image_features = torch.clamp(image_features, -10.0, 10.0)
        text_features = torch.clamp(text_features, -10.0, 10.0)
        
        # Unimodal pass (for contrastive)
        mm_uni, text_uni = self.qformer(image_features, text_features, use_bidirectional_mask=False)
        
        # Bidirectional pass (for ITM and classification)
        mm_bi, text_bi, itm_logits = None, None, None
        if compute_itm:
            mm_bi, text_bi = self.qformer(image_features, text_features, use_bidirectional_mask=True)
            itm_logits = self.qformer.compute_itm_logits(mm_bi)

        query_mean = (mm_bi if mm_bi is not None else mm_uni).mean(dim=1)
        query_mean = torch.clamp(torch.where(torch.isfinite(query_mean), query_mean, torch.zeros_like(query_mean)), -50.0, 50.0)
        class_logits = self.classification_head(query_mean)

        return mm_uni, image_features, text_uni, class_logits, mm_bi, text_bi, itm_logits


def create_teacher_model(
    chexzero_path: str = "checkpt/chexzero_best.pt",
    vocab_size: int = 10000,
    num_queries: int = 12,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_layers: int = 4,
    num_heads: int = 12,
    dropout: float = 0.1,
    num_classes: int = 2
) -> TeacherModel:
    """Create teacher model with CheXzero vision encoder."""
    from .chexzero_vision_encoder import load_chexzero_vision_encoder
    vision_encoder = load_chexzero_vision_encoder(chexzero_path, freeze=True, device='cpu')
    
    model = TeacherModel(
        vision_encoder=vision_encoder,
        vocab_size=vocab_size,
        num_queries=num_queries,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes
    )

    return model.to(device)


if __name__ == "__main__":
    model = create_teacher_model(device="cpu")
    images = torch.randn(2, 3, 224, 224)
    texts = ["Bilateral pneumonia.", "No abnormality."]
    inputs = model.text_tokenizer.encode(texts)
    out = model(images, inputs['input_ids'], inputs['attention_mask'])
    print(f"Q-Former output: {out[0].shape}, Classification: {out[3].shape}")
