"""Loss functions for teacher-student distillation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss for image-text alignment."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute bidirectional contrastive loss between image and text features."""
        # Sanitize inputs
        image_features = torch.where(torch.isfinite(image_features), image_features, torch.zeros_like(image_features))
        text_features = torch.where(torch.isfinite(text_features), text_features, torch.zeros_like(text_features))

        # Normalize features
        image_norm = torch.clamp(torch.norm(image_features, dim=-1, keepdim=True), min=1e-6)
        text_norm = torch.clamp(torch.norm(text_features, dim=-1, keepdim=True), min=1e-6)
        image_features = torch.clamp(image_features / image_norm, -2.0, 2.0)
        text_features = torch.clamp(text_features / text_norm, -2.0, 2.0)

        # Compute similarity matrix
        logits = torch.clamp(torch.matmul(image_features, text_features.t()), -2.0, 2.0)
        raw_similarities = logits.clone()
        
        # Scale by temperature
        logits = torch.clamp(logits / self.temperature, -50.0, 50.0)

        # Labels are diagonal (matching pairs)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Bidirectional cross-entropy loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        loss = (loss_i2t + loss_t2i) / 2

        if not torch.isfinite(loss):
            loss = torch.tensor(0.01, device=loss.device, requires_grad=True)

        # Store statistics for monitoring
        positive_sims = torch.diag(raw_similarities)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=raw_similarities.device)
        negative_sims = raw_similarities[mask]
        
        self._last_stats = {
            'batch_size': batch_size,
            'positive_sim_mean': positive_sims.mean().item(),
            'negative_sim_mean': negative_sims.mean().item() if len(negative_sims) > 0 else 0.0,
            'separation': (positive_sims.mean() - negative_sims.mean()).item() if len(negative_sims) > 0 else 0.0
        }

        return loss


class ClassificationLoss(nn.Module):
    """Classification loss with optional Focal Loss for class imbalance."""

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        focal_alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.reduction = reduction

        if focal_gamma > 0:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        if self.focal_gamma > 0:
            ce_loss = self.ce_loss(logits, targets)
            probs = F.softmax(logits, dim=1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - p_t) ** self.focal_gamma

            if self.focal_alpha is not None:
                alpha_t = self.focal_alpha.gather(0, targets)
                focal_loss = alpha_t * focal_weight * ce_loss
            else:
                focal_loss = focal_weight * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss
        else:
            return self.ce_loss(logits, targets)


class TeacherLoss(nn.Module):
    """Combined loss for teacher model: contrastive + ITM + classification."""

    def __init__(
        self,
        lambda_contrast: float = 0.45,
        lambda_itm: float = 0.45,
        lambda_classification: float = 0.1,
        temperature: float = 0.07,
        query_selection_temp: float = 0.1,
        use_softmax_weighting: bool = True,
        num_classes: int = 4,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
        focal_alpha: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.lambda_contrast = lambda_contrast
        self.lambda_itm = lambda_itm
        self.lambda_classification = lambda_classification
        self.query_selection_temp = query_selection_temp
        self.use_softmax_weighting = use_softmax_weighting
        self.num_classes = num_classes

        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.itm_loss = nn.CrossEntropyLoss()
        self.classification_loss = ClassificationLoss(
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha
        )

    def forward(
        self,
        multimodal_features: torch.Tensor,
        processed_text_features: torch.Tensor,
        itm_logits: Optional[torch.Tensor] = None,
        classification_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs  # Accept unused args for compatibility
    ) -> dict:
        """Compute combined teacher loss."""
        batch_size = multimodal_features.shape[0]
        
        # Sanitize and normalize inputs
        multimodal_features = torch.clamp(
            torch.where(torch.isfinite(multimodal_features), multimodal_features, torch.zeros_like(multimodal_features)),
            -50.0, 50.0
        )
        processed_text_features = torch.clamp(
            torch.where(torch.isfinite(processed_text_features), processed_text_features, torch.zeros_like(processed_text_features)),
            -50.0, 50.0
        )
        
        # Extract and normalize text CLS token
        text_cls = processed_text_features[:, 0, :]
        text_norm = torch.clamp(torch.norm(text_cls, dim=-1, keepdim=True), min=1e-6)
        normalized_text = torch.clamp(text_cls / text_norm, -2.0, 2.0)
        
        # Normalize queries
        query_norm = torch.clamp(torch.norm(multimodal_features, dim=-1, keepdim=True), min=1e-6)
        normalized_queries = torch.clamp(multimodal_features / query_norm, -2.0, 2.0)
        
        # Select best queries via softmax weighting
        similarities = torch.clamp(torch.einsum('bqd,bd->bq', normalized_queries, normalized_text), -1.5, 1.5)
        
        if self.use_softmax_weighting:
            scaled_sim = torch.clamp(similarities / self.query_selection_temp, -50.0, 50.0)
            weights = F.softmax(scaled_sim - scaled_sim.max(dim=1, keepdim=True)[0], dim=1)
            best_queries = torch.einsum('bq,bqd->bd', weights, normalized_queries)
            max_similarities = torch.sum(weights * similarities, dim=1)
            best_query_indices = torch.argmax(weights, dim=1)
        else:
            max_similarities, best_query_indices = torch.max(similarities, dim=1)
            batch_idx = torch.arange(batch_size, device=multimodal_features.device)
            best_queries = normalized_queries[batch_idx, best_query_indices, :]
        
        # Compute losses
        loss_contrast = self.contrastive_loss(best_queries, normalized_text)
        if not torch.isfinite(loss_contrast):
            loss_contrast = torch.tensor(0.0, device=loss_contrast.device, requires_grad=True)
        
        loss_itm = torch.tensor(0.0, device=multimodal_features.device, requires_grad=True)
        if itm_logits is not None and self.lambda_itm > 0:
            pos_labels = torch.ones(batch_size, dtype=torch.long, device=multimodal_features.device)
            loss_itm = self.itm_loss(itm_logits, pos_labels)
        
        loss_class = torch.tensor(0.0, device=multimodal_features.device, requires_grad=True)
        if classification_logits is not None and labels is not None and self.lambda_classification > 0:
            logits = torch.clamp(classification_logits, -50.0, 50.0)
            cls_labels = torch.clamp(labels.long().squeeze(-1) if labels.dim() > 1 else labels.long(), 0, self.num_classes - 1)
            loss_class = self.classification_loss(logits, cls_labels)
        
        total_loss = (self.lambda_contrast * loss_contrast +
                     self.lambda_itm * loss_itm +
                     self.lambda_classification * loss_class)
        
        if not torch.isfinite(total_loss):
            total_loss = torch.tensor(0.01, device=total_loss.device, requires_grad=True)

        return {
            'total_loss': total_loss,
            'contrastive_loss': loss_contrast,
            'itm_loss': loss_itm,
            'classification_loss': loss_class,
            'max_similarity_mean': max_similarities.mean().item() if torch.isfinite(max_similarities.mean()) else 0.0,
            'best_query_indices': best_query_indices.cpu().numpy()
        }


class StudentLoss(nn.Module):
    """Combined distillation loss: MSE + Cosine + Classification."""

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_cosine: float = 1.0,
        lambda_classification: float = 0.5,
        num_classes: int = 2,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_cosine = lambda_cosine
        self.lambda_classification = lambda_classification
        
        self.classification_loss = ClassificationLoss(
            num_classes=num_classes,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha
        )

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """Compute distillation loss between student and teacher features."""
        teacher_features = teacher_features.detach()
        
        # Pool teacher features if needed
        if teacher_features.dim() == 3:
            teacher_features = teacher_features.mean(dim=1)
        
        # MSE loss
        mse_loss = F.mse_loss(student_features, teacher_features)
        
        # Cosine similarity loss
        cosine_sim = F.cosine_similarity(student_features, teacher_features, dim=-1)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Classification loss (optional)
        if student_logits is not None and labels is not None and self.lambda_classification > 0:
            class_loss = self.classification_loss(student_logits, labels)
        else:
            class_loss = torch.tensor(0.0, device=student_features.device)
        
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_cosine * cosine_loss +
            self.lambda_classification * class_loss
        )
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'cosine_loss': cosine_loss,
            'cosine_similarity': cosine_sim.mean(),
            'classification_loss': class_loss
        }
