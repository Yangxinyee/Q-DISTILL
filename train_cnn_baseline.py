"""Training script for CNN baseline model on chest X-rays."""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from model.cnn_baseline import create_cnn_baseline
from model.losses import ClassificationLoss
from model.dataset import CXRDataset


class CNNBaselineTrainer:
    """Trainer for CNN baseline model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: str,
        output_dir: Path,
        use_amp: bool = True,
        patience: int = 15,
        min_delta: float = 0.0001
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.patience = patience
        self.min_delta = min_delta

        self.scaler = GradScaler('cuda') if use_amp else None

        # Tracking metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['priority_labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.1f}%'})

        return total_loss / len(self.train_loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['priority_labels'].to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Progress bar
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.1f}%'})

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        return total_loss / len(self.val_loader), 100.0 * correct / total, all_preds, all_labels

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }

        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')
            print(f"  ✓ Saved best checkpoint (epoch {epoch+1})")

    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting CNN Baseline Training")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_preds, val_labels = self.validate(epoch)

            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # Scheduler step
            self.scheduler.step()

            # Check improvement
            is_first = (epoch == 0)
            improvement = val_acc - self.best_val_acc
            is_best = is_first or (improvement >= self.min_delta)

            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n{'='*60}")
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                print(f"{'='*60}")
                break

            # Save history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'best_val_acc': self.best_val_acc,
                'best_val_loss': self.best_val_loss
            }
            with open(self.output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

        print(f"\n✓ Training complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")


def collate_fn(batch):
    """Collate function for dataloader."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['priority_label'] for item in batch], dtype=torch.long)
    return {'images': images, 'priority_labels': labels}


def main():
    parser = argparse.ArgumentParser(description='Train CNN Baseline Model')

    # Data
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to metadata CSV')
    parser.add_argument('--root_dir', type=str, default='data',
                       help='Root directory for images')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--use_chexzero', action='store_true',
                       help='Use CheXzero as feature extractor')
    parser.add_argument('--chexzero_path', type=str,
                       default='checkpt/chexzero_best.pt',
                       help='Path to CheXzero checkpoint')

    # Loss
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (auto-generated if not specified)')
    parser.add_argument('--no_amp', action='store_true')

    # Early stopping
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.0001)

    args = parser.parse_args()

    # Auto-generate output dir
    if args.output_dir is None:
        if args.use_chexzero:
            args.output_dir = 'outputs/cnn_baseline_chexzero'
        else:
            args.output_dir = 'outputs/cnn_baseline'

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create model
    print("\nCreating model...")
    model = create_cnn_baseline(
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_chexzero=args.use_chexzero,
        chexzero_path=args.chexzero_path if args.use_chexzero else None,
        device=device
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = CXRDataset(
        metadata_path=args.csv_path,
        root_dir=args.root_dir,
        split='train',
        use_priority_labels=True
    )
    val_dataset = CXRDataset(
        metadata_path=args.csv_path,
        root_dir=args.root_dir,
        split='validation',
        use_priority_labels=True
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Loss function
    criterion = ClassificationLoss(
        num_classes=args.num_classes,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    # Create trainer
    trainer = CNNBaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        use_amp=not args.no_amp,
        patience=args.patience,
        min_delta=args.min_delta
    )

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
