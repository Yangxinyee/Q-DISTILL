"""Training script for teacher model."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from model.teacher import create_teacher_model
from model.losses import TeacherLoss
from model.dataset import CXRDataset, collate_fn_teacher


class Trainer:
    """Trainer for teacher model."""

    def __init__(self, model, train_loader, eval_loader, criterion, optimizer, device, args):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.args = args

        self.scaler = GradScaler('cuda') if args.use_amp else None
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

        self.best_train_loss = float('inf')
        self.train_history = []
        self.patience = args.patience
        self.patience_counter = 0
        self.max_grad_norm = args.max_grad_norm

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

    def _save_config(self):
        """Save config."""
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(self.args), f, indent=2)

    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss, total_contrast, total_itm, total_cls = 0.0, 0.0, 0.0, 0.0
        total_pos_sim, total_neg_sim, total_sep = 0.0, 0.0, 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        for batch in pbar:
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['priority_labels'].to(self.device)

            self.optimizer.zero_grad()

            if self.args.use_amp:
                with autocast('cuda'):
                    outputs = self.model(images, input_ids, attention_mask, compute_itm=True)
                    mm_uni, img_feat, text_uni, cls_logits, mm_bi, text_bi, itm_logits = outputs
                    
                    loss_dict = self.criterion(
                        multimodal_features=mm_uni,
                        processed_text_features=text_uni,
                        itm_logits=itm_logits,
                        classification_logits=cls_logits,
                        labels=labels
                    )
                    loss = loss_dict['total_loss']
                
                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                outputs = self.model(images, input_ids, attention_mask, compute_itm=True)
                mm_uni, img_feat, text_uni, cls_logits, mm_bi, text_bi, itm_logits = outputs
                
                loss_dict = self.criterion(
                    multimodal_features=mm_uni,
                    processed_text_features=text_uni,
                    itm_logits=itm_logits,
                    classification_logits=cls_logits,
                    labels=labels
                )
                loss = loss_dict['total_loss']
                
                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            if torch.isfinite(loss):
                total_loss += loss.item()
                total_contrast += loss_dict.get('contrastive_loss', torch.tensor(0.0)).item()
                total_itm += loss_dict.get('itm_loss', torch.tensor(0.0)).item()
                total_cls += loss_dict.get('classification_loss', torch.tensor(0.0)).item()
                
                # Collect contrastive stats
                contrast_stats = self.criterion.contrastive_loss._last_stats
                total_pos_sim += contrast_stats.get('positive_sim_mean', 0.0)
                total_neg_sim += contrast_stats.get('negative_sim_mean', 0.0)
                total_sep += contrast_stats.get('separation', 0.0)
                num_batches += 1

            pbar.set_postfix({'loss': f'{total_loss/(num_batches or 1):.4f}'})

        avg_loss = total_loss / max(num_batches, 1)
        return {
            'loss': avg_loss,
            'contrastive_loss': total_contrast / max(num_batches, 1),
            'itm_loss': total_itm / max(num_batches, 1),
            'classification_loss': total_cls / max(num_batches, 1),
            'positive_sim_mean': total_pos_sim / max(num_batches, 1),
            'negative_sim_mean': total_neg_sim / max(num_batches, 1),
            'separation': total_sep / max(num_batches, 1)
        }

    @torch.no_grad()
    def evaluate(self, epoch):
        """Evaluate on test set with I2T recall metrics."""
        self.model.eval()
        total_loss, num_batches = 0.0, 0
        all_img_embeds, all_text_embeds = [], []

        for batch in tqdm(self.eval_loader, desc="Evaluating"):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self.model(images, input_ids, attention_mask, compute_itm=False)
            mm_uni = outputs[0]
            text_uni = outputs[2]

            # Extract embeddings
            img_embed = mm_uni.mean(dim=1)
            text_embed = text_uni[:, 0]
            img_embed = nn.functional.normalize(img_embed, dim=-1)
            text_embed = nn.functional.normalize(text_embed, dim=-1)
            
            # Store for recall calculation
            all_img_embeds.append(img_embed.cpu())
            all_text_embeds.append(text_embed.cpu())
            
            # Compute batch loss
            logits = torch.matmul(img_embed, text_embed.t()) / 0.07
            labels = torch.arange(len(logits), device=self.device)
            loss = (nn.functional.cross_entropy(logits, labels) + nn.functional.cross_entropy(logits.t(), labels)) / 2

            total_loss += loss.item()
            num_batches += 1

        # Compute I2T Recall@1 and Recall@10
        all_img_embeds = torch.cat(all_img_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        sim_matrix = torch.matmul(all_img_embeds, all_text_embeds.t())
        n = len(sim_matrix)
        
        i2t_r1, i2t_r10 = 0, 0
        for i in range(n):
            ranks = torch.argsort(sim_matrix[i], descending=True)
            if i in ranks[:1]: i2t_r1 += 1
            if i in ranks[:10]: i2t_r10 += 1
        
        return {
            'eval_loss': total_loss / max(num_batches, 1),
            'val_i2t_recall@1': i2t_r1 / n,
            'val_i2t_recall@10': i2t_r10 / n
        }

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_train_loss': self.best_train_loss,
            'train_history': self.train_history
        }
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def train(self):
        """Main training loop."""
        print(f"Training for {self.args.epochs} epochs")
        print(f"Samples: {len(self.train_loader.dataset)}, Batch: {self.args.batch_size}")

        for epoch in range(1, self.args.epochs + 1):
            metrics = self.train_epoch(epoch)
            self.scheduler.step()

            print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Contrast={metrics['contrastive_loss']:.4f}, "
                  f"ITM={metrics['itm_loss']:.4f}, Cls={metrics['classification_loss']:.4f}, "
                  f"PosSim={metrics['positive_sim_mean']:.4f}, NegSim={metrics['negative_sim_mean']:.4f}")

            # Evaluate every 10 epochs
            if epoch % 10 == 0:
                eval_metrics = self.evaluate(epoch)
                metrics.update(eval_metrics)  # Merge eval metrics into this epoch
                print(f"  Eval Loss: {eval_metrics['eval_loss']:.4f}, "
                      f"I2T R@1: {eval_metrics['val_i2t_recall@1']:.4f}, "
                      f"I2T R@10: {eval_metrics['val_i2t_recall@10']:.4f}")
            
            self.train_history.append(metrics)

            # Save checkpoint
            is_best = metrics['loss'] < self.best_train_loss
            if is_best:
                self.best_train_loss = metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % self.args.save_freq == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save history
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump({'train': self.train_history, 'best_loss': self.best_train_loss}, f, indent=2)
        print(f"Training complete. Best loss: {self.best_train_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    
    # Data
    parser.add_argument('--metadata_path', type=str, default='data/metadata.csv')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--vocab_path', type=str, default='vocab/vocabulary.json')
    
    # Model
    parser.add_argument('--chexzero_path', type=str, default='checkpt/chexzero_best.pt')
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--num_queries', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--qformer_dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=2)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true')
    
    # Loss
    parser.add_argument('--lambda_contrast', type=float, default=0.45)
    parser.add_argument('--lambda_itm', type=float, default=0.45)
    parser.add_argument('--lambda_classification', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--use_class_weights', action='store_true', default=True)
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='outputs/teacher')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Datasets
    train_dataset = CXRDataset(args.metadata_path, args.root_dir, split='train', use_priority_labels=True)
    val_dataset = CXRDataset(args.metadata_path, args.root_dir, split='validation', use_priority_labels=True)
    test_dataset = CXRDataset(args.metadata_path, args.root_dir, split='test', use_priority_labels=True)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f"Train+Val: {len(combined_dataset)}, Test: {len(test_dataset)}")

    # Model
    model = create_teacher_model(
        chexzero_path=args.chexzero_path,
        vocab_size=args.vocab_size,
        num_queries=args.num_queries,
        device='cpu',
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.qformer_dropout,
        num_classes=args.num_classes
    )
    
    # Load vocabulary
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    model.text_tokenizer.load_vocab(str(vocab_path))

    # Multi-GPU
    if num_gpus > 1:
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
    model_unwrapped = model.module if isinstance(model, nn.DataParallel) else model

    # Dataloaders
    collate = lambda b: collate_fn_teacher(b, model_unwrapped.text_tokenizer, use_priority_labels=True)
    train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    # Class weights for focal loss
    focal_alpha = None
    if args.focal_gamma > 0 and args.use_class_weights:
        train_labels = np.concatenate([d.metadata['priority_label'].values for d in [train_dataset, val_dataset]])
        unique, counts = np.unique(train_labels, return_counts=True)
        weights = len(train_labels) / (len(unique) * counts)
        alpha = np.ones(args.num_classes)
        for l, w in zip(unique, weights):
            if 0 <= l < args.num_classes:
                alpha[int(l)] = w
        focal_alpha = torch.FloatTensor(alpha).to(device)

    # Loss
    criterion = TeacherLoss(
        lambda_contrast=args.lambda_contrast,
        lambda_itm=args.lambda_itm,
        lambda_classification=args.lambda_classification,
        temperature=args.temperature,
        num_classes=args.num_classes,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    trainer = Trainer(model, train_loader, eval_loader, criterion, optimizer, device, args)
    trainer.train()


if __name__ == '__main__':
    main()
