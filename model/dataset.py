"""Dataset and DataLoader for CXR image-text model training."""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, Callable, Tuple, Dict
import numpy as np


class CXRDataset(Dataset):
    """Dataset for CXR images, reports, and labels."""

    def __init__(
        self,
        metadata_path: str,
        root_dir: str,
        split: str = 'train',
        image_transform: Optional[Callable] = None,
        max_text_length: int = 512,
        use_priority_labels: bool = True
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.max_text_length = max_text_length
        self.use_priority_labels = use_priority_labels

        self.metadata = pd.read_csv(metadata_path)

        if use_priority_labels and 'priority_label' not in self.metadata.columns:
            raise ValueError("priority_label column not found in metadata")

        if split is not None and split != 'all':
            self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)

        self.num_classes = self.metadata['priority_label'].nunique() if use_priority_labels else None

        # Image transforms
        if image_transform is None:
            if split == 'train':
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])
        else:
            self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.metadata.iloc[idx]

        image_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)

        report_path = os.path.join(self.root_dir, row['report_path'])
        with open(report_path, 'r', encoding='utf-8') as f:
            report_text = f.read().strip()

        result = {
            'image': image,
            'report_text': report_text,
            'study_id': row['study_id'],
            'idx': idx
        }

        if self.use_priority_labels:
            result['priority_label'] = int(row['priority_label'])
        else:
            raise ValueError("use_priority_labels must be True")

        return result


def collate_fn_teacher(batch: list, tokenizer: Callable, use_priority_labels: bool = False) -> Dict[str, torch.Tensor]:
    """Collate function for teacher model training."""
    images = torch.stack([item['image'] for item in batch])
    report_texts = [item['report_text'] for item in batch]
    study_ids = [item['study_id'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch], dtype=torch.long)

    if hasattr(tokenizer, 'encode'):
        text_inputs = tokenizer.encode(report_texts, padding=True, truncation=True)
    else:
        text_inputs = tokenizer(report_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')

    result = {
        'images': images,
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'study_ids': study_ids,
        'indices': indices
    }

    if use_priority_labels:
        result['priority_labels'] = torch.tensor([item['priority_label'] for item in batch], dtype=torch.long)
    else:
        raise ValueError("use_priority_labels must be True")
    return result


def collate_fn_student(batch: list, use_priority_labels: bool = False) -> Dict[str, torch.Tensor]:
    """Collate function for student model (image-only)."""
    images = torch.stack([item['image'] for item in batch])
    study_ids = [item['study_id'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch], dtype=torch.long)

    result = {
        'images': images,
        'study_ids': study_ids,
        'indices': indices
    }

    if use_priority_labels:
        result['priority_labels'] = torch.tensor([item['priority_label'] for item in batch], dtype=torch.long)
    else:
        raise ValueError("use_priority_labels must be True")
    return result


def create_dataloaders(
    metadata_path: str,
    root_dir: str,
    tokenizer: Optional[Callable] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    for_student: bool = False,
    use_priority_labels: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_dataset = CXRDataset(metadata_path, root_dir, 'train', use_priority_labels=use_priority_labels)
    val_dataset = CXRDataset(metadata_path, root_dir, 'validation', use_priority_labels=use_priority_labels)
    test_dataset = CXRDataset(metadata_path, root_dir, 'test', use_priority_labels=use_priority_labels)

    if for_student:
        collate_func = lambda batch: collate_fn_student(batch, use_priority_labels=use_priority_labels)
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer required for teacher model dataloaders")
        collate_func = lambda batch: collate_fn_teacher(batch, tokenizer, use_priority_labels=use_priority_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, collate_fn=collate_func, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, collate_fn=collate_func, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_func, pin_memory=True)

    return train_loader, val_loader, test_loader


class CXRDatasetMedGemma(Dataset):
    """Dataset for student training with MedGemma-generated reports."""
    
    def __init__(
        self,
        metadata_path: str,
        medgemma_metadata_path: str,
        root_dir: str,
        medgemma_report_dir: str,
        split: str = 'train',
        image_transform: Optional[Callable] = None,
        max_text_length: int = 512,
        use_priority_labels: bool = False
    ):
        super().__init__()
        self.root_dir = root_dir
        self.medgemma_report_dir = medgemma_report_dir
        self.split = split
        self.max_text_length = max_text_length
        self.use_priority_labels = use_priority_labels
        
        self.metadata = pd.read_csv(metadata_path)
        self.medgemma_metadata = pd.read_csv(medgemma_metadata_path)
        
        self.merged_data = pd.merge(
            self.metadata,
            self.medgemma_metadata[['study_id', 'medgemma_report_path']],
            on='study_id',
            how='inner'
        )
        
        if split is not None and split != 'all':
            self.merged_data = self.merged_data[self.merged_data['split'] == split].reset_index(drop=True)
        
        if use_priority_labels and 'priority_label' not in self.merged_data.columns:
            raise ValueError("priority_label column not found")
        
        # Image transforms
        if image_transform is None:
            if split == 'train':
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])
        else:
            self.image_transform = image_transform
    
    def __len__(self) -> int:
        return len(self.merged_data)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        row = self.merged_data.iloc[idx]
        
        image_path = os.path.join(self.root_dir, row['image_path'])
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        medgemma_report_path = os.path.join(self.medgemma_report_dir, row['medgemma_report_path'])
        with open(medgemma_report_path, 'r', encoding='utf-8') as f:
            medgemma_report = f.read().strip()
        
        result = {
            'image': image,
            'medgemma_report': medgemma_report,
            'study_id': row['study_id'],
            'idx': idx
        }
        
        # Load real report if available
        if 'report_path' in row and pd.notna(row['report_path']):
            report_path = os.path.join(self.root_dir, row['report_path'])
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    result['real_report'] = f.read().strip()
            else:
                result['real_report'] = medgemma_report
        else:
            result['real_report'] = medgemma_report
        
        if self.use_priority_labels:
            result['priority_label'] = int(row['priority_label'])
        else:
            raise ValueError("use_priority_labels must be True")
        return result


def collate_fn_student_medgemma(batch: list, tokenizer: Callable, use_real_reports_for_teacher: bool = False) -> Dict[str, torch.Tensor]:
    """Collate function for student model with MedGemma reports."""
    images = torch.stack([item['image'] for item in batch])
    medgemma_reports = [item['medgemma_report'] for item in batch]
    study_ids = [item['study_id'] for item in batch]
    indices = torch.tensor([item['idx'] for item in batch], dtype=torch.long)
    
    if hasattr(tokenizer, 'encode'):
        text_inputs = tokenizer.encode(medgemma_reports, padding=True, truncation=True)
    else:
        text_inputs = tokenizer(medgemma_reports, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    result = {
        'images': images,
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'study_ids': study_ids,
        'indices': indices
    }
    
    if use_real_reports_for_teacher:
        real_reports = [item.get('real_report', item['medgemma_report']) for item in batch]
        if hasattr(tokenizer, 'encode'):
            teacher_inputs = tokenizer.encode(real_reports, padding=True, truncation=True)
        else:
            teacher_inputs = tokenizer(real_reports, padding=True, truncation=True, max_length=512, return_tensors='pt')
        result['teacher_input_ids'] = teacher_inputs['input_ids']
        result['teacher_attention_mask'] = teacher_inputs['attention_mask']
    
    if 'priority_label' in batch[0]:
        result['priority_labels'] = torch.tensor([item['priority_label'] for item in batch], dtype=torch.long)
    else:
        raise ValueError("use_priority_labels must be True")
    return result


if __name__ == "__main__":
    metadata_path = "data/metadata.csv"
    root_dir = "data"

    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}")
    else:
        dataset = CXRDataset(metadata_path, root_dir, split='train')
        sample = dataset[0]
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample: image={sample['image'].shape}, label={sample['priorty_label']:d}")
