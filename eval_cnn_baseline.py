"""Evaluation script for CNN baseline model on chest X-rays."""

import argparse
import torch
import numpy as np
from pathlib import Path
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from model.cnn_baseline import create_cnn_baseline
from model.dataset import CXRDataset


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['priority_label'] for item in batch], dtype=torch.long)
    return {'images': images, 'priority_labels': labels}


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        labels = batch['priority_labels'].to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return all_preds, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN Baseline Model')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Data
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to metadata CSV')
    parser.add_argument('--root_dir', type=str, default='data',
                       help='Root directory for images')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'validation', 'test'])

    # Model config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_chexzero', action='store_true',
                       help='Use CheXzero as feature extractor')
    parser.add_argument('--chexzero_path', type=str,
                       default='checkpt/chexzero_best.pt',
                       help='Path to CheXzero checkpoint')

    # Output
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save evaluation results')

    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_cnn_baseline(
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_chexzero=args.use_chexzero,
        chexzero_path=args.chexzero_path if args.use_chexzero else None,
        device=device
    )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = CXRDataset(
        metadata_path=args.csv_path,
        root_dir=args.root_dir,
        split=args.split
    )
    print(f"  {len(dataset)} samples")

    # Dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Evaluate
    print("\nEvaluating...")
    preds, labels, probs = evaluate(model, dataloader, device)

    # Compute metrics
    accuracy = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds)

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.split} set)")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    class_names = ['Non-urgent', 'Urgent']
    print(classification_report(labels, preds, target_names=class_names))

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    for i, name in enumerate(class_names):
        class_mask = labels == i
        class_acc = (preds[class_mask] == i).mean() * 100
        print(f"  {name}: {class_acc:.2f}% ({class_mask.sum()} samples)")

    # Save results
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_path = checkpoint_dir / f'eval_results_{args.split}.json'

    results = {
        'split': args.split,
        'num_samples': len(dataset),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'use_chexzero': args.use_chexzero,
        'per_class': {
            name: {
                'count': int((labels == i).sum()),
                'accuracy': float((preds[labels == i] == i).mean() * 100)
            }
            for i, name in enumerate(class_names)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
