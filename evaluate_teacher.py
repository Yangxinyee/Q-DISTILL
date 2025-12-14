"""Evaluation script for teacher model."""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from torch.utils.data import DataLoader

from model.teacher import create_teacher_model
from model.dataset import CXRDataset, collate_fn_teacher
from model.losses import ContrastiveLoss


class TeacherEvaluator:
    """Evaluator for teacher model."""

    def __init__(self, model, dataloader, device='cuda', save_dir='outputs/evaluation', temperature=0.07):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.model.eval()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)

    @torch.no_grad()
    def evaluate(self):
        """Run evaluation."""
        all_image_embeds, all_text_embeds = [], []
        all_preds, all_labels = [], []
        total_loss, num_batches = 0.0, 0

        for batch in tqdm(self.dataloader, desc="Evaluating"):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            outputs = self.model(images, input_ids, attention_mask, compute_itm=True)
            mm_feat, _, text_feat, cls_logits = outputs[0], outputs[1], outputs[2], outputs[3]

            # Extract embeddings
            text_embed = F.normalize(text_feat[:, 0, :], dim=-1)
            query_norm = F.normalize(mm_feat, dim=-1)
            sim = torch.einsum('bqd,bd->bq', query_norm, text_embed)
            weights = F.softmax(sim / 0.1, dim=1)
            image_embed = torch.einsum('bq,bqd->bd', weights, query_norm)

            # Loss
            loss = self.contrastive_loss(image_embed, text_embed)
            total_loss += loss.item()
            num_batches += 1

            all_image_embeds.append(image_embed.cpu())
            all_text_embeds.append(text_embed.cpu())

            # Classification
            if cls_logits is not None and 'priority_labels' in batch:
                all_preds.append(cls_logits.argmax(dim=1).cpu())
                all_labels.append(batch['priority_labels'])

        # Aggregate
        image_embeds = torch.cat(all_image_embeds, dim=0)
        text_embeds = torch.cat(all_text_embeds, dim=0)

        # Retrieval metrics
        sim_matrix = torch.matmul(image_embeds, text_embeds.t())
        n = len(sim_matrix)
        
        i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10 = 0, 0, 0, 0, 0, 0
        for i in range(n):
            i2t_ranks = torch.argsort(sim_matrix[i], descending=True)
            t2i_ranks = torch.argsort(sim_matrix[:, i], descending=True)
            if i in i2t_ranks[:1]: i2t_r1 += 1
            if i in i2t_ranks[:5]: i2t_r5 += 1
            if i in i2t_ranks[:10]: i2t_r10 += 1
            if i in t2i_ranks[:1]: t2i_r1 += 1
            if i in t2i_ranks[:5]: t2i_r5 += 1
            if i in t2i_ranks[:10]: t2i_r10 += 1

        metrics = {
            'contrastive_loss': total_loss / num_batches,
            'i2t_recall@1': i2t_r1 / n,
            'i2t_recall@5': i2t_r5 / n,
            'i2t_recall@10': i2t_r10 / n,
            't2i_recall@1': t2i_r1 / n,
            't2i_recall@5': t2i_r5 / n,
            't2i_recall@10': t2i_r10 / n,
            'num_samples': n
        }

        # Classification metrics
        if len(all_preds) > 0:
            preds = torch.cat(all_preds).numpy()
            labels = torch.cat(all_labels).numpy()
            from sklearn.metrics import accuracy_score, f1_score
            metrics['accuracy'] = accuracy_score(labels, preds)
            metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)

        # Save and visualize
        self._save_results(metrics)
        self._create_visualizations(sim_matrix, metrics)
        return metrics

    def _save_results(self, metrics):
        """Save results."""
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {self.save_dir / 'metrics.json'}")

    def _create_visualizations(self, sim_matrix, metrics):
        """Create visualizations with 3 subplots."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        n = len(sim_matrix)

        # 1. Similarity matrix (random 100 samples)
        ax = axes[0]
        sample_size = min(100, n)
        if n > 100:
            indices = np.random.choice(n, size=100, replace=False)
            indices = np.sort(indices)
            sampled_matrix = sim_matrix[indices][:, indices].numpy()
        else:
            sampled_matrix = sim_matrix.numpy()
        sns.heatmap(sampled_matrix, ax=ax, cmap='Blues', cbar_kws={'shrink': 0.8})
        ax.set_title(f'Similarity Matrix (n={sample_size} random samples)')
        ax.set_xlabel('Text Index')
        ax.set_ylabel('Image Index')

        # 2. Similarity distribution (positive vs negative)
        ax = axes[1]
        positive_sims = torch.diag(sim_matrix).numpy()
        mask = ~np.eye(n, dtype=bool)
        negative_sims = sim_matrix.numpy()[mask]
        
        ax.hist(negative_sims, bins=50, alpha=0.7, label='Negative', color='red', density=True)
        ax.hist(positive_sims, bins=30, alpha=0.7, label='Positive', color='green', density=True)
        
        # Mean lines
        pos_mean = np.mean(positive_sims)
        neg_mean = np.mean(negative_sims)
        ax.axvline(pos_mean, color='darkgreen', linestyle='--', linewidth=2, label=f'Pos Mean: {pos_mean:.3f}')
        ax.axvline(neg_mean, color='darkred', linestyle='--', linewidth=2, label=f'Neg Mean: {neg_mean:.3f}')
        
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Density')
        ax.set_title('Similarity Distribution')
        ax.legend(loc='upper right')

        # 3. Retrieval metrics bar chart (with @10)
        ax = axes[2]
        metric_names = ['I2T@1', 'I2T@5', 'I2T@10', 'T2I@1', 'T2I@5', 'T2I@10']
        values = [
            metrics['i2t_recall@1'], metrics['i2t_recall@5'], metrics['i2t_recall@10'],
            metrics['t2i_recall@1'], metrics['t2i_recall@5'], metrics['t2i_recall@10']
        ]
        colors = ['#1f77b4', '#6baed6', '#c6dbef', '#2ca02c', '#74c476', '#c7e9c0']
        bars = ax.bar(metric_names, values, color=colors)
        ax.set_ylabel('Recall')
        ax.set_title('Retrieval Metrics')
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', fontsize=9)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plots saved to {self.save_dir / 'evaluation_plots.png'}")


def plot_training_curves(history_path, output_dir=None):
    """Plot all training curves from training_history.json in a single figure."""
    # Close any existing figures to avoid memory issues
    plt.close('all')
    
    history_path = Path(history_path)
    if output_dir is None:
        output_dir = history_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(history_path) as f:
        data = json.load(f)
    
    train_history = data.get('train', [])
    if not train_history:
        print("No training history found.")
        return
    
    epochs = list(range(1, len(train_history) + 1))
    
    # Define which metrics to plot and their display names
    metric_configs = [
        ('loss', 'Total Loss', 'blue'),
        ('contrastive_loss', 'Contrastive Loss', 'green'),
        ('itm_loss', 'ITM Loss', 'orange'),
        ('classification_loss', 'Cls Loss', 'red'),
        ('positive_sim_mean', 'Pos Sim Mean', 'purple'),
        ('negative_sim_mean', 'Neg Sim Mean', 'brown'),
        ('separation', 'Separation', 'teal'),
        ('val_i2t_recall@1', 'Val I2T R@1', 'magenta'),
        ('val_i2t_recall@10', 'Val I2T R@10', 'cyan'),
    ]
    
    # Create figure with 3x3 subplots using constrained_layout
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    
    for idx, (metric_key, title, color) in enumerate(metric_configs):
        ax = axes[idx]
        values = [ep.get(metric_key) for ep in train_history]
        
        # Handle sparse metrics (those only recorded every N epochs)
        valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            marker = 'o' if len(valid_epochs) < 30 else None
            ax.plot(valid_epochs, valid_values, color=color, linewidth=1.5, marker=marker, markersize=4)
            ax.set_xlabel('Epoch', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=8)
            
            # Add min/max annotation for recall metrics
            if 'recall' in metric_key.lower() and len(valid_values) > 1:
                max_val = max(valid_values)
                max_epoch = valid_epochs[valid_values.index(max_val)]
                ax.annotate(f'Max: {max_val:.3f}', xy=(max_epoch, max_val), 
                           fontsize=7, ha='center', color='black',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(title, fontsize=10, fontweight='bold')
    
    output_path = output_dir / 'training_curves.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Teacher Model')
    
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint (required for evaluation)')
    parser.add_argument('--metadata_path', type=str, default='data/metadata.csv')
    parser.add_argument('--root_dir', type=str, default='data')
    parser.add_argument('--chexzero_path', type=str, default='checkpt/chexzero_best.pt')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation')
    
    # Training curves plotting options
    parser.add_argument('--plot_curves', action='store_true')
    parser.add_argument('--history_path', type=str, default='outputs/teacher/training_history.json')
    
    args = parser.parse_args()
    
    # If plot_curves mode, only plot training curves
    if args.plot_curves:
        if args.history_path is None:
            print("Error: --history_path is required when using --plot_curves")
            return
        plot_training_curves(args.history_path, args.output_dir if args.output_dir != 'outputs/evaluation' else None)
        return
    
    # Otherwise, require checkpoint for evaluation
    if args.checkpoint is None:
        print("Error: --checkpoint is required for evaluation")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load config
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = checkpoint_dir / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    vocab_size = config.get('vocab_size', 10000)
    num_queries = config.get('num_queries', 12)
    num_layers = config.get('num_layers', 4)
    num_heads = config.get('num_heads', 12)
    dropout = config.get('qformer_dropout', 0.1)
    num_classes = config.get('num_classes', 2)
    temperature = config.get('temperature', 0.07)

    # Create model
    model = create_teacher_model(
        chexzero_path=args.chexzero_path,
        vocab_size=vocab_size,
        num_queries=num_queries,
        device=device,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes
    )

    # Load vocabulary
    vocab_path = checkpoint_dir / 'vocabulary.json'
    if not vocab_path.exists():
        vocab_path = Path('vocab/vocabulary.json')
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    model.text_tokenizer.load_vocab(str(vocab_path))

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Dataset
    dataset = CXRDataset(args.metadata_path, args.root_dir, split=args.split, use_priority_labels=True)
    collate = lambda b: collate_fn_teacher(b, model.text_tokenizer, use_priority_labels=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate, pin_memory=True)

    print(f"Evaluating on {args.split} split ({len(dataset)} samples)")

    evaluator = TeacherEvaluator(model, dataloader, device, args.output_dir, temperature)
    metrics = evaluator.evaluate()

    print(f"\n{'='*50}")
    print("Results:")
    print(f"  Contrastive Loss: {metrics['contrastive_loss']:.4f}")
    print(f"  I2T Recall@1: {metrics['i2t_recall@1']:.4f}")
    print(f"  I2T Recall@5: {metrics['i2t_recall@5']:.4f}")
    print(f"  I2T Recall@10: {metrics['i2t_recall@10']:.4f}")
    print(f"  T2I Recall@1: {metrics['t2i_recall@1']:.4f}")
    print(f"  T2I Recall@5: {metrics['t2i_recall@5']:.4f}")
    print(f"  T2I Recall@10: {metrics['t2i_recall@10']:.4f}")
    if 'accuracy' in metrics:
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
