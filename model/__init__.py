"""Q-DISTILL: Teacher-Student Knowledge Distillation for CXR Triage"""

from .teacher import TeacherModel, QFormer, ClassificationHead, create_teacher_model
from .losses import ContrastiveLoss, ClassificationLoss
from .dataset import CXRDataset, create_dataloaders

__version__ = "1.0.0"

__all__ = [
    "TeacherModel",
    "QFormer", 
    "ClassificationHead",
    "create_teacher_model",
    "ContrastiveLoss",
    "ClassificationLoss",
    "CXRDataset",
    "create_dataloaders",
]
