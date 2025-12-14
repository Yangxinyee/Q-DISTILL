"""Training script for lightweight student model with Q-Former."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime

from model.student import create_student_model
from model.losses import StudentLoss
from model.dataset import CXRDatasetMedGemma, collate_fn_student_medgemma
from model.teacher import create_teacher_model, SimpleTokenizer


class StudentTrainer:
    """Trainer for Q-Former student model."""

    def __init__(self, student_model, teacher_model, train_loader, val_loader,
                 criterion, optimizer, scheduler, device, output_dir,
                 mask_report=False, use_amp=False, lambda_classification=1.0,
                 args=None, distill_target='bi'):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.mask_report = mask_report
        self.use_amp = use_amp
        self.lambda_classification = lambda_classification
        self.args = args
        self.distill_target = distill_target

        self.scaler = GradScaler('cuda') if use_amp else None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses, self.val_losses = [], []
        self.val_accuracies = []
        self.patience = getattr(args, 'patience', 15) if args else 15
        self.patience_counter = 0
        self.min_delta = getattr(args, 'min_delta', 0.0001) if args else 0.0001

    def _get_teacher_features(self, images, input_ids, attention_mask):
        """Get teacher features based on distill_target."""
        use_bi = (self.distill_target == 'bi')
        outputs = self.teacher_model(images, input_ids, attention_mask, compute_itm=use_bi)
        queries = outputs[4] if use_bi else outputs[0]
        return queries.mean(dim=1)

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.student_model.train()
        self.teacher_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        cls_enabled = self.lambda_classification > 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch in pbar:
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['priority_labels'].to(self.device)
            teacher_ids = batch.get('teacher_input_ids', input_ids).to(self.device)
            teacher_mask = batch.get('teacher_attention_mask', attention_mask).to(self.device)

            if self.mask_report:
                input_ids = torch.zeros_like(input_ids)
                attention_mask = torch.zeros_like(attention_mask)

            self.optimizer.zero_grad()

            with torch.no_grad():
                teacher_features = self._get_teacher_features(images, teacher_ids, teacher_mask)

            if self.use_amp:
                with autocast('cuda'):
                    student_features = self.student_model(images, input_ids, attention_mask,
                                                          mask_report=self.mask_report, return_logits=False)
                    loss_inputs = {'student_features': student_features, 'teacher_features': teacher_features}
                    if cls_enabled:
                        logits = self.student_model.classification_head(student_features)
                        loss_inputs.update({'student_logits': logits, 'labels': labels})
                    loss_dict = self.criterion(**loss_inputs)
                    loss = loss_dict['total_loss']
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                student_features = self.student_model(images, input_ids, attention_mask,
                                                      mask_report=self.mask_report, return_logits=False)
                loss_inputs = {'student_features': student_features, 'teacher_features': teacher_features}
                if cls_enabled:
                    logits = self.student_model.classification_head(student_features)
                    loss_inputs.update({'student_logits': logits, 'labels': labels})
                loss_dict = self.criterion(**loss_inputs)
                loss = loss_dict['total_loss']
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            if cls_enabled:
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}',
                            'acc': f'{100*correct/total:.1f}%' if total > 0 else 'N/A'})

        return total_loss / len(self.train_loader), 100 * correct / total if total > 0 else 0.0

    @torch.no_grad()
    def validate(self, epoch):
        """Validate model."""
        self.student_model.eval()
        self.teacher_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        cls_enabled = self.lambda_classification > 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
        for batch in pbar:
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['priority_labels'].to(self.device)
            teacher_ids = batch.get('teacher_input_ids', input_ids).to(self.device)
            teacher_mask = batch.get('teacher_attention_mask', attention_mask).to(self.device)

            if self.mask_report:
                input_ids = torch.zeros_like(input_ids)
                attention_mask = torch.zeros_like(attention_mask)

            teacher_features = self._get_teacher_features(images, teacher_ids, teacher_mask)
            student_features = self.student_model(images, input_ids, attention_mask,
                                                  mask_report=self.mask_report, return_logits=False)

            loss_inputs = {'student_features': student_features, 'teacher_features': teacher_features}
            if cls_enabled:
                logits = self.student_model.classification_head(student_features)
                loss_inputs.update({'student_logits': logits, 'labels': labels})

            loss_dict = self.criterion(**loss_inputs)
            total_loss += loss_dict['total_loss'].item()

            if cls_enabled:
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix({'loss': f'{total_loss/(pbar.n+1):.4f}',
                            'acc': f'{100*correct/total:.1f}%' if total > 0 else 'N/A'})

        return total_loss / len(self.val_loader), 100 * correct / total if total > 0 else 0.0

    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pt')
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pt')

    def train(self, num_epochs):
        """Main training loop."""
        print(f"Training for {num_epochs} epochs, output: {self.output_dir}")

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            self.scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.1f}% | "
                  f"Val Loss={val_loss:.4f}, Acc={val_acc:.1f}%")

            # Check improvement
            if self.lambda_classification > 0:
                is_best = val_acc > self.best_val_acc + self.min_delta
                if is_best:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            else:
                is_best = val_loss < self.best_val_loss - self.min_delta
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

            # Save history
            with open(self.output_dir / 'training_history.json', 'w') as f:
                json.dump({'train_losses': self.train_losses, 'val_losses': self.val_losses,
                          'val_accuracies': self.val_accuracies}, f)

        print(f"Training complete. Best: Loss={self.best_val_loss:.4f}, Acc={self.best_val_acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Train Student Q-Former')
    
    # Required
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--chexzero_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--medgemma_metadata_path', type=str, required=True)
    
    # Data
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--medgemma_report_dir', type=str, default='data/medgemma_reports')
    parser.add_argument('--vocab_path', type=str, default='vocab/vocabulary.json')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=0.0001)
    
    # Model
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--mask_report', action='store_true')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True)
    parser.add_argument('--student_num_queries', type=int, default=None)
    parser.add_argument('--student_num_layers', type=int, default=None)
    
    # Loss
    parser.add_argument('--lambda_cosine', type=float, default=1.0)
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    parser.add_argument('--lambda_classification', type=float, default=1.0)
    parser.add_argument('--focal_alpha', type=str, default=None)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--distill_target', type=str, default='bi', choices=['uni', 'bi'])
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/student')
    parser.add_argument('--use_amp', action='store_true')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'timestamp': datetime.now().isoformat()}, f, indent=2)

    # Load teacher config
    teacher_dir = Path(args.teacher_checkpoint).parent
    config_path = teacher_dir / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            tc = json.load(f)
        vocab_size = tc.get('vocab_size', 10000)
        num_queries_t = tc.get('num_queries', 12)
        num_layers_t = tc.get('num_layers', 4)
        num_heads_t = tc.get('num_heads', 12)
        num_classes_t = tc.get('num_classes', 2)
    else:
        vocab_size, num_queries_t, num_layers_t, num_heads_t, num_classes_t = 10000, 12, 4, 12, 2

    # Create models
    student_model = create_student_model(
        chexzero_path=args.chexzero_path,
        teacher_checkpoint=args.teacher_checkpoint,
        num_classes=args.num_classes,
        device=device,
        freeze_text_encoder=args.freeze_text_encoder,
        num_queries=args.student_num_queries or 8,
        num_layers=args.student_num_layers or 4
    )

    teacher_model = create_teacher_model(
        chexzero_path=args.chexzero_path,
        vocab_size=vocab_size,
        num_queries=num_queries_t,
        device=device,
        num_layers=num_layers_t,
        num_heads=num_heads_t,
        num_classes=num_classes_t
    )

    # Load teacher weights
    checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
    state = checkpoint.get('model_state_dict', checkpoint)
    teacher_model.load_state_dict(state, strict=False)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Load vocabulary
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=512)
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    tokenizer.load_vocab(str(vocab_path))

    # Create datasets
    train_dataset = CXRDatasetMedGemma(args.csv_path, args.medgemma_metadata_path,
                                       args.root_dir, args.medgemma_report_dir,
                                       split='train', use_priority_labels=True)
    val_dataset = CXRDatasetMedGemma(args.csv_path, args.medgemma_metadata_path,
                                     args.root_dir, args.medgemma_report_dir,
                                     split='validation', use_priority_labels=True)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    collate = lambda b: collate_fn_student_medgemma(b, tokenizer, use_real_reports_for_teacher=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    # Loss
    focal_alpha = None
    if args.focal_alpha:
        focal_alpha = torch.tensor([float(x) for x in args.focal_alpha.split(',')]).to(device)
    
    criterion = StudentLoss(
        lambda_mse=args.lambda_distill,
        lambda_cosine=args.lambda_cosine,
        lambda_classification=args.lambda_classification,
        num_classes=args.num_classes,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha
    )

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    trainer = StudentTrainer(
        student_model, teacher_model, train_loader, val_loader,
        criterion, optimizer, scheduler, device, output_dir,
        mask_report=args.mask_report, use_amp=args.use_amp,
        lambda_classification=args.lambda_classification, args=args,
        distill_target=args.distill_target
    )
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
